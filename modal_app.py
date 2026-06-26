"""Modal app: benchmark attention backends for tabular foundation models on CUDA.

Usage (from repo root):

    uv run modal run modal_app.py                      # full sweep on H100
    uv run modal run modal_app.py --quick              # fast smoke test
    uv run modal run modal_app.py --gpu A100           # different GPU
    uv run modal run modal_app.py --backward           # also time fwd+bwd
    uv run modal run modal_app.py --iters 100 --warmup 20

Results are written locally to results.json / results.csv and summarized to stdout.

Pinned, mutually-compatible versions (torch 2.5.1 cu12 / cxx11abi=FALSE):
  - flash-attn prebuilt wheel matches torch 2.5 to avoid a slow source build.
  - xformers 0.0.28.post3 is built for torch 2.5.1.
If an install is incompatible at runtime, the backend reports itself unavailable
and the benchmark still runs the others (see availability table in output).
"""
import modal

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.5.1")
    .uv_pip_install("xformers==0.0.28.post3")
    .uv_pip_install(FLASH_ATTN_WHEEL)
    # torch ships libnvrtc.so.12 (as a dep) but doesn't put it on the loader
    # path; cuDNN's runtime-fusion attention engine needs it. Register its dir
    # with ldconfig so the sdpa-cudnn backend can build kernels.
    .run_commands(
        "d=$(dirname $(find / -name 'libnvrtc.so.12' 2>/dev/null | head -1)); "
        "if [ -n \"$d\" ]; then echo \"$d\" > /etc/ld.so.conf.d/nvidia-nvrtc.conf "
        "&& ldconfig && echo \"registered nvrtc dir: $d\"; "
        "else echo 'WARNING: libnvrtc.so.12 not found'; fi"
    )
    .add_local_python_source("bench")
)

app = modal.App("tab-attn-bench", image=image)

# Results are persisted here from inside the container after every config, so a
# local network/heartbeat drop (or a --detach run) never loses work. Retrieve
# with `modal run modal_app.py::fetch` or the CLI `modal volume get`.
results_vol = modal.Volume.from_name("tab-attn-bench-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(gpu="H100", timeout=60 * 45, volumes={RESULTS_DIR: results_vol})
def run_bench(configs, warmup: int, iters: int, backward: bool,
              correctness: bool, run_id: str):
    import json
    from bench.runner import run_all

    def checkpoint(partial):
        with open(f"{RESULTS_DIR}/{run_id}.json", "w") as f:
            json.dump(partial, f, indent=2)
        results_vol.commit()

    return run_all(configs, warmup=warmup, iters=iters, backward=backward,
                   do_correctness=correctness, progress_cb=checkpoint)


def _read_volume_json(run_id: str) -> dict:
    import json
    data = b"".join(results_vol.read_file(f"{run_id}.json"))
    return json.loads(data)


@app.local_entrypoint()
def main(
    gpu: str = "H100",
    iters: int = 50,
    warmup: int = 10,
    backward: bool = False,
    quick: bool = False,
    correctness: bool = True,
    out: str = "results",
    run_id: str = "",
):
    import time
    from bench.config import default_sweep
    from bench.report import write_outputs, print_summary

    rid = run_id or f"{gpu.lower()}-{'quick' if quick else 'full'}-{int(time.time())}"
    configs = default_sweep(quick=quick)
    print(f"Submitting {len(configs)} shape configs to a {gpu} on Modal "
          f"(iters={iters}, warmup={warmup}, backward={backward})")
    print(f"run_id={rid}  (checkpointed to volume 'tab-attn-bench-results')")

    fn = run_bench if gpu == "H100" else run_bench.with_options(gpu=gpu)
    try:
        payload = fn.remote(configs, warmup, iters, backward, correctness, rid)
    except Exception as e:  # noqa: BLE001 - recover partial/final from the volume
        print(f"\nLost connection to the run ({type(e).__name__}: {e}).")
        print(f"Recovering checkpointed results from the volume (run_id={rid})...")
        results_vol.reload()
        payload = _read_volume_json(rid)

    json_path, csv_path = write_outputs(payload, out)
    print_summary(payload)
    print(f"Wrote {json_path} and {csv_path}  (status={payload['meta'].get('status')})")


@app.local_entrypoint()
def fetch(run_id: str = "", out: str = "results"):
    """Download a (possibly in-progress) run's results from the Modal volume."""
    from bench.report import write_outputs, print_summary
    results_vol.reload()
    if not run_id:
        files = sorted(results_vol.listdir("/"), key=lambda f: f.mtime)
        if not files:
            print("No runs found in volume 'tab-attn-bench-results'.")
            return
        run_id = files[-1].path.removesuffix(".json")
        print(f"Latest run_id: {run_id}")
    payload = _read_volume_json(run_id)
    json_path, csv_path = write_outputs(payload, out)
    print_summary(payload)
    print(f"Wrote {json_path} and {csv_path}  (status={payload['meta'].get('status')})")
