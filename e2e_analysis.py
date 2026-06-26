"""Analyze a PyTorch chrome trace into systems-level observability metrics.

Pure stdlib so it runs in-container AND can be unit-tested locally on a saved
trace. The chrome trace cleanly separates CPU ops, CUDA runtime calls, and GPU
kernels by `cat`, which lets us measure exactly the things a latency number
hides: copies, casts, syncs, launch overhead and stream concurrency.

Key outputs (all per single profiled `predict`):
  gpu_busy_union_ms  union of GPU kernel intervals  -> the compute floor
  gpu_busy_sum_ms    sum of kernel durations         (sum>union => overlap)
  overlap_ms         sum - union                     -> concurrency achieved
  window_ms          wall span of the profiled call
  busy_fraction      union / window                  -> low => sync/launch bound
  buckets            per-kind {ms, count}            -> where GPU time goes
  launches/syncs/...  CUDA runtime call counts        -> the overhead levers
  top_gaps           largest GPU idle bubbles        -> where to add concurrency
"""
from __future__ import annotations

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}


def _cat(ev):
    return (ev.get("cat") or ev.get("category") or "").lower()


def classify_kernel(name: str, cat: str) -> str:
    """Bucket a GPU kernel by what it is doing."""
    nl = name.lower()
    if cat == "gpu_memcpy" or "memcpy" in nl:
        return "memcpy"
    if cat == "gpu_memset" or "memset" in nl:
        return "memset"
    if any(k in nl for k in ("flash", "fmha", "sdpa", "scaled_dot",
                             "attention", "attn", "cudnn_generated")):
        return "attention"
    if any(k in nl for k in ("gemm", "cutlass", "wgmma", "cublas", "sgemm",
                             "hgemm", "matmul", "ampere_", "xmma", "s16816",
                             "_mm_", "dot_kernel", "addmm", "bmm", "::mm",
                             "baddbmm")):
        return "gemm"
    if "softmax" in nl:
        return "softmax"
    if "norm" in nl:
        return "norm"
    if any(k in nl for k in ("copy", "cast", "convert")):
        return "copy/cast"
    if any(k in nl for k in ("elementwise", "vectorized", "reduce", "fill",
                             "gelu", "activation", "index", "arange", "add",
                             "mul", "sub", "div", "clamp", "where", "cat_",
                             "stack", "transpose", "permute", "scatter",
                             "gather")):
        return "elementwise"
    return "other"


def _memcpy_dir(name: str) -> str:
    n = name.lower()
    if "dtoh" in n or "device->host" in n or "device -> host" in n:
        return "DtoH"
    if "htod" in n or "host->device" in n or "host -> device" in n:
        return "HtoD"
    if "dtod" in n or "device->device" in n:
        return "DtoD"
    return "other"


def _union_and_gaps(intervals):
    """intervals: list of (start_us, end_us). Returns (union_ms, gaps[(ms,at_ms)])."""
    if not intervals:
        return 0.0, []
    intervals = sorted(intervals)
    union = 0.0
    gaps = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            union += ce - cs
            gaps.append(((s - ce) / 1000.0, ce / 1000.0))  # (gap_ms, at_ms)
            cs, ce = s, e
        else:
            ce = max(ce, e)
    union += ce - cs
    return union / 1000.0, gaps


def analyze(trace: dict, wall_ms: float | None = None) -> dict:
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    gpu_events, runtime_events = [], []
    all_ts = []
    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev:
            continue
        cat = _cat(ev)
        ts, dur = float(ev["ts"]), float(ev["dur"])
        all_ts.append((ts, ts + dur))
        if cat in GPU_CATS:
            gpu_events.append((ev.get("name", ""), cat, ts, dur,
                               ev.get("tid")))
        elif cat in RUNTIME_CATS:
            runtime_events.append(ev.get("name", ""))

    # GPU buckets + union/overlap
    buckets = {}
    intervals = []
    memcpy_by_dir = {}
    streams = set()
    for name, cat, ts, dur, tid in gpu_events:
        b = classify_kernel(name, cat)
        rec = buckets.setdefault(b, {"ms": 0.0, "count": 0})
        rec["ms"] += dur / 1000.0
        rec["count"] += 1
        intervals.append((ts, ts + dur))
        streams.add(tid)
        if b == "memcpy":
            d = _memcpy_dir(name)
            mr = memcpy_by_dir.setdefault(d, {"ms": 0.0, "count": 0})
            mr["ms"] += dur / 1000.0
            mr["count"] += 1

    gpu_sum_ms = sum(r["ms"] for r in buckets.values())
    union_ms, gaps = _union_and_gaps(intervals)
    overlap_ms = max(0.0, gpu_sum_ms - union_ms)

    # trace wall window
    if all_ts:
        window_ms = (max(e for _, e in all_ts) - min(s for s, _ in all_ts)) / 1000.0
    else:
        window_ms = 0.0

    # runtime call counts (the overhead levers)
    def count(*subs):
        return sum(1 for n in runtime_events
                   if any(s.lower() in n.lower() for s in subs))

    launches = count("cudaLaunchKernel", "cuLaunchKernel")
    syncs = count("Synchronize")  # stream/device/event synchronize
    memcpy_calls = count("Memcpy")
    n_gpu = len(gpu_events)

    gaps.sort(reverse=True)
    rep = {
        "wall_ms": round(wall_ms, 3) if wall_ms else None,
        "window_ms": round(window_ms, 3),
        "gpu_busy_union_ms": round(union_ms, 3),
        "gpu_busy_sum_ms": round(gpu_sum_ms, 3),
        "overlap_ms": round(overlap_ms, 3),
        "busy_fraction": round(union_ms / window_ms, 3) if window_ms else None,
        "exposed_overhead_ms": (round(wall_ms - union_ms, 3)
                                if wall_ms else None),
        "n_gpu_kernels": n_gpu,
        "n_streams": len([s for s in streams if s is not None]),
        "launches": launches,
        "syncs": syncs,
        "memcpy_calls": memcpy_calls,
        "buckets": {k: {"ms": round(v["ms"], 3), "count": v["count"]}
                    for k, v in sorted(buckets.items(),
                                       key=lambda kv: -kv[1]["ms"])},
        "memcpy_by_dir": {k: {"ms": round(v["ms"], 3), "count": v["count"]}
                          for k, v in memcpy_by_dir.items()},
        "top_gaps_ms": [{"gap_ms": round(g, 3), "at_ms": round(a, 3)}
                        for g, a in gaps[:8]],
    }
    return rep


def _selftest():
    # synthetic trace: 2 kernels back-to-back with a gap, one memcpy, runtime
    tr = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "ampere_gemm", "ts": 0, "dur": 100, "tid": 7},
        {"ph": "X", "cat": "kernel", "name": "flash_fwd", "ts": 150, "dur": 50, "tid": 7},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH", "ts": 210, "dur": 20, "tid": 8},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 0, "dur": 1},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 230, "dur": 5},
    ]}
    r = analyze(tr, wall_ms=0.30)
    assert r["gpu_busy_sum_ms"] == 0.17, r
    assert r["gpu_busy_union_ms"] == 0.17, r          # no overlap
    assert r["buckets"]["gemm"]["count"] == 1
    assert r["buckets"]["attention"]["count"] == 1
    assert r["launches"] == 1 and r["syncs"] == 1
    assert r["memcpy_by_dir"]["DtoH"]["count"] == 1
    assert r["top_gaps_ms"][0]["gap_ms"] == 0.05      # the 50us gap
    print("e2e_analysis selftest OK:", r["buckets"])


if __name__ == "__main__":
    _selftest()
