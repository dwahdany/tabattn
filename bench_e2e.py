"""Instrumented end-to-end TabPFN-3 latency harness with full trace capture.

Objective = end-to-end `predict` latency (clean wall, median) gated on
output-equivalence. Around that, it captures the systems-level observability
that the latency number hides -- copies, casts, syncs, launch overhead, stream
concurrency -- via a PyTorch chrome trace analyzed by e2e_analysis, plus a
module-level component breakdown. Chrome traces are saved to a volume for
manual inspection in perfetto / chrome://tracing.

    uv run modal run bench_e2e.py::main                  # full basket
    uv run modal run bench_e2e.py::main --points typical # one point
    uv run modal run bench_e2e.py::fetch_traces          # download traces

This baselines the stock model. Next step is to install TabPFN from an editable
fork and let an autoresearch loop optimize against this same harness.
"""
import modal

import e2e_analysis  # noqa: F401  (ensure it's mounted)

CACHE = "/vol"
TRACES = "/traces"
HF_REPO = "Prior-Labs/tabpfn_3"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("tabpfn", "scikit-learn")
    .env({
        "HF_HOME": f"{CACHE}/hf",
        "HF_HUB_CACHE": f"{CACHE}/hf/hub",
        "TABPFN_MODEL_CACHE_DIR": f"{CACHE}/tabpfn",
    })
    .add_local_python_source("e2e_analysis", "bench_e2e")
)

app = modal.App("tabpfn-e2e-bench", image=image)
cache_vol = modal.Volume.from_name("tabpfn-cache", create_if_missing=True)
traces_vol = modal.Volume.from_name("tabpfn-traces", create_if_missing=True)

# operating points spanning the regimes: small=overhead/launch-bound,
# typical=mixed, sample=cross-row attention, feature=cross-column/embedder.
POINTS = {
    "tiny":    dict(n_train=1024,  n_test=512,  n_features=32),
    "typical": dict(n_train=4096,  n_test=2048, n_features=64),
    "sample":  dict(n_train=16384, n_test=2048, n_features=64),
    "feature": dict(n_train=8192,  n_test=2048, n_features=512),
}


def _log(msg):
    print(msg, flush=True)


def _load_clf(n_estimators, filename):
    from huggingface_hub import list_repo_files, hf_hub_download
    from tabpfn import TabPFNClassifier
    if not filename:
        files = list_repo_files(HF_REPO)
        cands = [f for f in files if f.endswith(".ckpt") and "classifier" in f.lower()]
        filename = ([f for f in cands if "multiclass" in f.lower()] or cands)[0]
    local_path = hf_hub_download(HF_REPO, filename)
    cache_vol.commit()
    return filename, local_path


def _component_hooks(model):
    """Hook top-level components to attribute GPU time by model stage."""
    import torch
    cat_of = {}
    for n, m in model.named_modules():
        if n == "feature_distribution_embedder":
            cat_of[m] = "feature_embedder(ISAB)"
        elif n == "column_aggregator":
            cat_of[m] = "column_agg(cross-col)"
        elif n.endswith(".icl_attention") and n.startswith("icl_blocks."):
            cat_of[m] = "icl_attention(cross-row)"
        elif n.endswith(".mlp") and n.startswith("icl_blocks."):
            cat_of[m] = "icl_mlp"
    events, starts, handles = [], {}, []

    def pre(m, inp):
        e = torch.cuda.Event(enable_timing=True); e.record(); starts[id(m)] = e

    def mk_post(cat):
        def post(m, inp, out):
            s = starts.get(id(m))
            if s is None:
                return
            e = torch.cuda.Event(enable_timing=True); e.record()
            events.append((cat, s, e))
        return post

    for m, cat in cat_of.items():
        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(mk_post(cat)))
    return events, handles


def _dev_us(ev):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(ev, attr, None)
        if v:
            return v
    return 0.0


def _callsite_attribution(filename, n_train=2048, n_test=256, n_features=64):
    """Attribute GPU time to operators (by op name + bucket), once, on a small
    model. Python-line stacks aren't available (TabPFN-3 runs through compiled /
    C++ paths, so with_stack records nothing), so we attribute by aten op name,
    which is robust and still actionable ('aten::copy_ = X ms')."""
    import torch
    from sklearn.datasets import make_classification
    from tabpfn import TabPFNClassifier
    from torch.profiler import profile, ProfilerActivity
    from e2e_analysis import classify_kernel
    try:
        _, local_path = _load_clf(1, filename)
        X, y = make_classification(n_samples=n_train + n_test,
                                   n_features=n_features,
                                   n_informative=max(2, n_features // 2),
                                   n_classes=2, random_state=0)
        clf = TabPFNClassifier(device="cuda", n_estimators=1,
                               model_path=local_path,
                               ignore_pretraining_limits=True)
        clf.fit(X[:n_train], y[:n_train])
        Xte = X[n_train:]
        clf.predict(Xte)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            clf.predict(Xte)
        torch.cuda.synchronize()

        # attribute GPU self-time by op name; keep the aten:: dispatcher ops
        # (readable) and the raw kernels separately so totals aren't conflated.
        ops = []
        for e in prof.key_averages():
            dt = _dev_us(e)
            if dt <= 0:
                continue
            key = e.key
            kind = "aten" if key.startswith("aten::") else "kernel"
            ops.append({"op": key[:70], "kind": kind,
                        "bucket": classify_kernel(key, ""),
                        "ms": round(dt / 1e3, 3), "count": e.count})
        ops.sort(key=lambda r: -r["ms"])
        aten = [o for o in ops if o["kind"] == "aten"][:20]
        kernels = [o for o in ops if o["kind"] == "kernel"][:20]
        # bucket totals from the aten view (1 op -> its dispatched device time)
        bucket_ms = {}
        for o in aten:
            bucket_ms[o["bucket"]] = round(
                bucket_ms.get(o["bucket"], 0.0) + o["ms"], 3)
        return {"config": f"{n_train}x{n_test}x{n_features}",
                "by_aten_op": aten, "by_kernel": kernels,
                "bucket_ms_aten": dict(sorted(bucket_ms.items(),
                                              key=lambda kv: -kv[1]))}
    except Exception as e:
        import traceback
        return {"error": repr(e), "traceback": traceback.format_exc()}


def _point_body(cfg, repeats, warmup, n_estimators, filename, save_trace):
    import gzip, json, time
    import numpy as np
    import torch
    from sklearn.datasets import make_classification
    from tabpfn import TabPFNClassifier
    from e2e_analysis import analyze

    name = cfg["name"]
    nt, nte, nf = cfg["n_train"], cfg["n_test"], cfg["n_features"]
    key = f"{name}_{nt}x{nte}x{nf}"
    _log(f"\n===== {key} =====")

    fn, local_path = _load_clf(n_estimators, filename)
    X, y = make_classification(n_samples=nt + nte, n_features=nf,
                               n_informative=max(2, nf // 2), n_classes=2,
                               random_state=0)
    Xtr, ytr, Xte = X[:nt], y[:nt], X[nt:]
    clf = TabPFNClassifier(device="cuda", n_estimators=n_estimators,
                           model_path=local_path, ignore_pretraining_limits=True)
    clf.fit(Xtr, ytr)

    # ---- clean wall-clock latency (objective) ----
    torch.cuda.synchronize()
    for _ in range(warmup):
        clf.predict(Xte)
    torch.cuda.synchronize()
    walls = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        clf.predict(Xte)
        torch.cuda.synchronize(); walls.append((time.perf_counter() - t0) * 1e3)
    walls.sort()
    wall_ms = walls[len(walls) // 2]
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    _log(f"  wall median={wall_ms:.2f} ms  (min={walls[0]:.2f})  peak={peak_mb:.0f} MB")

    # ---- correctness gate (output-equivalence vs stored baseline) ----
    proba = clf.predict_proba(Xte)
    base_path = f"{TRACES}/baseline_{key}.npy"
    import os
    correctness = {"baseline": "stored-now"}
    if os.path.exists(base_path):
        base = np.load(base_path)
        if base.shape == proba.shape:
            correctness = {
                "max_abs_dproba": float(np.abs(base - proba).max()),
                "argmax_agreement": float((base.argmax(1) == proba.argmax(1)).mean()),
            }
    else:
        np.save(base_path, proba)
        traces_vol.commit()

    # ---- module-level component breakdown (CUDA events) ----
    model = clf.models_[0]
    ev, handles = _component_hooks(model)
    ev.clear()
    clf.predict(Xte)
    torch.cuda.synchronize()
    comp = {}
    for cat, s, e in ev:
        comp[cat] = comp.get(cat, 0.0) + s.elapsed_time(e)
    for h in handles:
        h.remove()
    comp = {k: round(v, 3) for k, v in sorted(comp.items(), key=lambda x: -x[1])}

    # ---- chrome trace + systems analysis ----
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        clf.predict(Xte)
    torch.cuda.synchronize()
    trace_path = f"/tmp/{key}.chrome.json"
    prof.export_chrome_trace(trace_path)
    with open(trace_path) as f:
        trace = json.load(f)
    analysis = analyze(trace, wall_ms=wall_ms)
    _log(f"  gpu_busy(union)={analysis['gpu_busy_union_ms']:.2f} ms  "
         f"busy_frac={analysis['busy_fraction']}  "
         f"exposed_overhead={analysis['exposed_overhead_ms']:.2f} ms")
    _log(f"  launches={analysis['launches']} syncs={analysis['syncs']} "
         f"memcpy={analysis['memcpy_calls']} streams={analysis['n_streams']} "
         f"kernels={analysis['n_gpu_kernels']}")
    _log(f"  buckets(ms): " + ", ".join(
        f"{k}={v['ms']:.1f}" for k, v in list(analysis["buckets"].items())[:8]))

    report = {
        "point": name, "key": key, "ckpt": fn,
        "params": dict(n_train=nt, n_test=nte, n_features=nf,
                       n_estimators=n_estimators, repeats=repeats),
        "wall_ms_median": round(wall_ms, 3),
        "wall_ms_min": round(walls[0], 3),
        "peak_mem_mb": round(peak_mb, 1),
        "correctness": correctness,
        "component_breakdown_ms": comp,
        "systems": analysis,
    }

    if save_trace:
        gz = f"{TRACES}/{key}.chrome.json.gz"
        with open(trace_path, "rb") as fin, gzip.open(gz, "wb") as fout:
            fout.writelines(fin)
        with open(f"{TRACES}/{key}.summary.json", "w") as f:
            json.dump(report, f, indent=2)
        traces_vol.commit()
        report["trace_file"] = gz
    return report


@app.function(gpu="H100", volumes={CACHE: cache_vol, TRACES: traces_vol},
              timeout=60 * 40)
def bench(points, repeats=5, warmup=2, n_estimators=1, filename="",
          save_trace=True):
    import traceback
    import torch
    runs = []
    for name in points:
        cfg = dict(name=name, **POINTS[name])
        try:
            runs.append(_point_body(cfg, repeats, warmup, n_estimators,
                                    filename, save_trace))
        except Exception as e:
            runs.append({"point": name, "error": repr(e),
                         "traceback": traceback.format_exc()})
        torch.cuda.empty_cache()
    _log("\n===== operator attribution (small model) =====")
    hotspots = _callsite_attribution(filename)
    for h in hotspots.get("by_aten_op", [])[:10]:
        _log(f"  [{h['bucket']:>10}] {h['ms']:>7.1f}ms x{h['count']:<5} {h['op']}")
    return {"gpu": str(torch.cuda.get_device_name(0)),
            "torch": str(torch.__version__), "runs": runs,
            "hotspots": hotspots}


def _print_report(payload):
    print(f"\nGPU {payload['gpu']} | torch {payload['torch']}")
    hdr = (f'{"point":>8} {"wall_ms":>8} {"gpu_busy":>9} {"overhd":>7} '
           f'{"busy%":>6} {"launch":>7} {"sync":>5} {"DtoH":>5} | top buckets(ms)')
    print(hdr); print("-" * len(hdr))
    for r in payload["runs"]:
        if "error" in r:
            print(f'{r["point"]:>8}  ERROR {r["error"][:60]}'); continue
        s = r["systems"]
        dtoh = s.get("memcpy_by_dir", {}).get("DtoH", {}).get("count", 0)
        bk = ", ".join(f"{k}={v['ms']:.0f}" for k, v in
                       list(s["buckets"].items())[:5])
        print(f'{r["point"]:>8} {r["wall_ms_median"]:>8.1f} '
              f'{s["gpu_busy_union_ms"]:>9.1f} {s["exposed_overhead_ms"]:>7.1f} '
              f'{(s["busy_fraction"] or 0) * 100:>5.0f}% {s["launches"]:>7} '
              f'{s["syncs"]:>5} {dtoh:>5} | {bk}')
    # component attribution
    print("\ncomponent breakdown (ms, CUDA events):")
    for r in payload["runs"]:
        if "error" in r:
            continue
        comp = r["component_breakdown_ms"]
        cs = ", ".join(f"{k}={v:.1f}" for k, v in comp.items())
        print(f'  {r["point"]:>8}: {cs}')
    # operator attribution (global, small model)
    hs = payload.get("hotspots", {})
    if "by_aten_op" in hs:
        print(f'\ntop GPU operators (ms)  [small model {hs.get("config")}]:')
        for h in hs["by_aten_op"][:14]:
            print(f'  [{h["bucket"]:>10}] {h["ms"]:>7.1f}ms x{h["count"]:<6} {h["op"]}')
        print(f'  bucket totals (aten view): {hs.get("bucket_ms_aten")}')


@app.local_entrypoint()
def main(points: str = "", repeats: int = 5, warmup: int = 2,
         n_estimators: int = 1, filename: str = "", out: str = "e2e_results"):
    import json
    names = points.split(",") if points else list(POINTS)
    payload = bench.remote(names, repeats, warmup, n_estimators, filename, True)
    with open(f"{out}.json", "w") as f:
        json.dump(payload, f, indent=2)
    _print_report(payload)
    print(f"\nWrote {out}.json   (traces in volume 'tabpfn-traces')")


@app.function(gpu="H100", volumes={CACHE: cache_vol, TRACES: traces_vol},
              timeout=60 * 15)
def hotspots_only(filename: str = ""):
    import torch
    return {"gpu": str(torch.cuda.get_device_name(0)),
            "hotspots": _callsite_attribution(filename)}


@app.local_entrypoint()
def hotspots(filename: str = ""):
    import json
    print(json.dumps(hotspots_only.remote(filename), indent=2, default=str))


@app.local_entrypoint()
def fetch_traces(dest: str = "traces"):
    """Download all chrome traces + summaries from the volume."""
    import os
    os.makedirs(dest, exist_ok=True)
    n = 0
    for entry in traces_vol.iterdir("/"):
        data = b"".join(traces_vol.read_file(entry.path))
        with open(os.path.join(dest, os.path.basename(entry.path)), "wb") as f:
            f.write(data)
        n += 1
        print(f"  {entry.path}  ({len(data)/1e6:.1f} MB)")
    print(f"Downloaded {n} files to {dest}/")
