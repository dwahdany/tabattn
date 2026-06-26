# tab-attn-bench

Benchmark of **attention backends** for tabular foundation models (TabPFN,
RPT-OSS, …), run on CUDA GPUs via [Modal](https://modal.com).

Tabular FMs embed a table to `(rows, cols, dim)` and run attention along **two
axes**, which are very different kernel regimes:

| regime | tokens attend across | sequence length | batch | character |
|--------|----------------------|-----------------|-------|-----------|
| **cross-row** | all rows (samples) | `n_rows` — long (≤ ~10k) | `n_cols` — modest | classic long-sequence; FlashAttention territory |
| **cross-column** | all columns (features) | `n_cols` — short (16–128) | `n_rows` — large (thousands) | many tiny attentions; launch/overhead bound |

The benchmark measures which attention implementation is fastest in **each**
regime. A different backend can (and does) win each one.

## What it compares

- **eager** — naive `matmul + softmax + matmul` (materializes the S×S matrix; OOMs on long sequences — the baseline that motivates fused kernels)
- **sdpa-math / sdpa-flash / sdpa-mem-eff / sdpa-cudnn** — `torch.nn.functional.scaled_dot_product_attention` forced to each kernel via `torch.nn.attention.sdpa_kernel`
- **flash-attn** — Dao-AILab `flash_attn_func`
- **xformers** — `xformers.ops.memory_efficient_attention`

Each backend runs in its own native tensor layout (SDPA/eager: `BHSD`;
flash-attn/xformers: `BSHD`). **Timing covers only the attention call** —
layout conversion happens outside the timed region — so we measure the kernel,
not glue code.

Metrics per (backend × shape × dtype): forward latency (median / p10 / p90 /
min, via CUDA events), achieved TFLOP/s, peak memory. Optional forward+backward
(training-step) timing with `--backward`. A correctness pass checks every
backend against an fp32 eager reference.

## Setup

Local machine only needs `modal` (the GPU deps — torch, flash-attn, xformers —
live in the Modal image). Authenticate Modal once: `modal setup`.

## Run

There are two local entrypoints (`main`, `fetch`), so the entrypoint must be
named explicitly with `::main`:

```bash
uv run modal run modal_app.py::main                  # full sweep on H100
uv run modal run modal_app.py::main --quick          # 2-config smoke test
uv run modal run modal_app.py::main --gpu A100       # different GPU
uv run modal run modal_app.py::main --backward       # also time fwd+bwd
uv run modal run modal_app.py::main --iters 100 --warmup 20
uv run modal run modal_app.py::main --out runs/h100  # stem -> runs/h100.{json,csv}
```

First run builds the image (~minutes; cached afterwards). Results are written to
`results.json` (full payload incl. metadata, availability, correctness) and
`results.csv` (one row per backend×shape), and a summary table prints to stdout.

### Long runs / flaky networks

Results are checkpointed to a Modal **Volume** (`tab-attn-bench-results`) after
every config. For long sweeps, run **detached** so a local network blip can't
kill the run, then fetch the results:

```bash
uv run modal run --detach modal_app.py::main --run-id h100-full --out results_h100
# ...if the client disconnects, recover the (partial or final) results:
uv run modal run modal_app.py::fetch --run-id h100-full --out results_h100
uv run modal run modal_app.py::fetch                 # newest run if --run-id omitted
```

`main` also auto-recovers from the volume if it loses the connection mid-run.

## Plots

```bash
uv run python plots.py [results_h100.json] [figures_dir]   # -> figures/*.png
```

Four figures, each making one argument:

- **`scaling.png`** — forward latency vs sequence length, 2×2 (head config ×
  regime). The hero: cuDNN/flash-attn dominate long cross-row sequences, while
  for short cross-column sequences the fused kernels lose to plain eager and the
  winner *varies with size*.
- **`ranking.png`** — heatmap of every config, each cell = latency relative to
  the fastest backend (1.0× = winner), so you can see who wins everywhere at a
  glance (OOM cells marked).
- **`utilization.png`** — achieved TFLOP/s vs the H100's 989 TFLOP/s bf16 peak.
  Cross-row is compute-bound (~45% of peak); cross-column is overhead-bound
  (never escapes launch latency).
- **`memory.png`** — peak memory vs sequence length, with S² extrapolation
  showing exactly where eager/math cross the 80 GB wall and OOM, while fused
  kernels stay linear.

## The sweep

Defined in `bench/config.py`. Two head configs — `std` (8 heads × 64) and
`tabpfn` (6 heads × 32, TabPFN-v2-like) — crossed with:

- **cross-row**: `n_cols=32`, `n_rows ∈ {1024, 4096, 8192, 16384}`
- **cross-col**: `n_rows=4096`, `n_cols ∈ {16, 32, 64, 128}`
- dtypes `{bf16, fp16}`

Edit `default_sweep()` to change shapes. Attention is full/bidirectional
(non-causal), matching how tabular FMs attend over rows/columns.

## Layout

```
modal_app.py        Modal image + GPU function + local entrypoint
bench/
  config.py         ShapeConfig + default_sweep()  (the shape matrix)
  backends.py       backend adapters behind one interface
  runner.py         CUDA-event timing + correctness (runs on the GPU)
  report.py         local CSV/JSON writer + summary table (stdlib only)
```

## Notes

- `sdpa-cudnn` needs `libnvrtc.so.12` on the loader path for its runtime-fusion
  engine; the image registers it via `ldconfig`. Without it, cuDNN attention
  reports "No execution plans support the graph".
- Backends that can't run a given shape/dtype (e.g. flash/cuDNN require
  fp16/bf16; eager/math OOM on long sequences) are recorded as
  `unsupported`/`oom`/`error` rather than crashing the run — see the
  availability and per-cell status in the output.
- Pinned compatible versions: torch 2.5.1 (cu124), flash-attn 2.7.4.post1
  (prebuilt `cu12torch2.5` wheel), xformers 0.0.28.post3.
```
