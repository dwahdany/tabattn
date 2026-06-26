"""Timing + correctness harness. Runs on the GPU (inside the Modal container)."""
from __future__ import annotations

from .backends import build_backends, EagerBackend
from .config import ShapeConfig, dtype_of


def _classify_error(e: Exception) -> str:
    s = str(e).lower()
    if "out of memory" in s:
        return "oom"
    if ("no available kernel" in s or "not support" in s or "unsupported" in s
            or "no kernel" in s or "invalid" in s):
        return "unsupported"
    return "error"


def _make_qkv(cfg: ShapeConfig, device, dtype, seed: int = 0):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (cfg.batch, cfg.heads, cfg.seq, cfg.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, generator=g)
    k = torch.randn(shape, device=device, dtype=dtype, generator=g)
    v = torch.randn(shape, device=device, dtype=dtype, generator=g)
    return q, k, v


def _fwd_flops(cfg: ShapeConfig) -> int:
    # QK^T (2*S*S*D) + AV (2*S*S*D) per (batch, head)
    return 4 * cfg.batch * cfg.heads * cfg.seq * cfg.seq * cfg.head_dim


def _percentiles(times_ms: list[float]) -> dict:
    t = sorted(times_ms)
    n = len(t)
    return {
        "median_ms": t[n // 2],
        "p10_ms": t[max(0, int(0.1 * (n - 1)))],
        "p90_ms": t[int(0.9 * (n - 1))],
        "min_ms": t[0],
    }


def _time_loop(call, iters: int):
    """Time `call` (a zero-arg fn that runs one op) with CUDA events."""
    import torch
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        call()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def benchmark_one(backend, cfg: ShapeConfig, warmup: int, iters: int,
                  backward: bool) -> dict:
    import torch
    device = "cuda"
    dtype = dtype_of(cfg.dtype)
    base = {
        "backend": backend.name,
        "regime": cfg.regime,
        "config": cfg.name,
        "batch": cfg.batch,
        "seq": cfg.seq,
        "heads": cfg.heads,
        "head_dim": cfg.head_dim,
        "dtype": cfg.dtype,
    }
    flops = _fwd_flops(cfg)

    try:
        q, k, v = _make_qkv(cfg, device, dtype)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        return {**base, "status": _classify_error(e), "error": str(e)[:200]}

    out = None
    try:
        # ---- forward (inference) ----
        prepared = backend.prepare(q, k, v, requires_grad=False)
        with torch.no_grad():
            for _ in range(warmup):
                out = backend.run(prepared)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = _time_loop(lambda: backend.run(prepared), iters)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        pct = _percentiles(times)
        result = {
            **base,
            "status": "ok",
            "fwd_ms": pct["median_ms"],
            "fwd_min_ms": pct["min_ms"],
            "fwd_p10_ms": pct["p10_ms"],
            "fwd_p90_ms": pct["p90_ms"],
            "fwd_tflops": flops / (pct["median_ms"] / 1e3) / 1e12,
            "fwd_peak_mb": peak_mb,
        }
    except Exception as e:  # noqa: BLE001 - we want to record any failure
        torch.cuda.empty_cache()
        return {**base, "status": _classify_error(e), "error": str(e)[:200]}

    # ---- forward + backward (training step), optional ----
    if backward:
        try:
            pb = backend.prepare(q, k, v, requires_grad=True)

            def _step():
                for t in pb:
                    t.grad = None
                o = backend.run(pb)
                o.float().sum().backward()

            for _ in range(warmup):
                _step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times_b = _time_loop(_step, iters)
            pctb = _percentiles(times_b)
            # fwd + bwd ~ 3.5x forward FLOPs (bwd computes dq,dk,dv ~ 2.5x fwd)
            result.update({
                "fwdbwd_ms": pctb["median_ms"],
                "fwdbwd_min_ms": pctb["min_ms"],
                "fwdbwd_tflops": (3.5 * flops) / (pctb["median_ms"] / 1e3) / 1e12,
                "fwdbwd_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
            })
            del pb
        except Exception as e:  # noqa: BLE001
            result["fwdbwd_status"] = _classify_error(e)
            result["fwdbwd_error"] = str(e)[:200]

    del prepared, q, k, v, out
    torch.cuda.empty_cache()
    return result


def check_correctness(dtype_str: str = "bf16") -> list[dict]:
    """Compare each available backend against an fp32 eager reference on a small
    shape, reporting max abs error so we know we're comparing correct kernels."""
    import torch
    device = "cuda"
    cfg = ShapeConfig("chk", "cross_col", batch=8, seq=128, heads=4,
                      head_dim=64, dtype=dtype_str)
    qf, kf, vf = _make_qkv(cfg, device, torch.float32)

    ref_be = EagerBackend()
    with torch.no_grad():
        ref = ref_be.to_canonical(ref_be.run(ref_be.prepare(qf, kf, vf)))

    dt = dtype_of(dtype_str)
    qd, kd, vd = qf.to(dt), kf.to(dt), vf.to(dt)

    rows = []
    for be in build_backends():
        ok, why = be.available()
        if not ok:
            rows.append({"backend": be.name, "status": "unavailable", "detail": why[:120]})
            continue
        try:
            with torch.no_grad():
                out = be.to_canonical(be.run(be.prepare(qd, kd, vd)))
            err = (out.float() - ref).abs().max().item()
            rows.append({"backend": be.name, "status": "ok",
                         "max_abs_err": err, "dtype": dtype_str})
        except Exception as e:  # noqa: BLE001
            rows.append({"backend": be.name, "status": _classify_error(e),
                         "detail": str(e)[:120]})
    return rows


def run_all(configs: list[ShapeConfig], warmup: int, iters: int,
            backward: bool, do_correctness: bool, progress_cb=None) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    all_backends = build_backends()
    availability = {}
    for be in all_backends:
        ok, why = be.available()
        availability[be.name] = {"available": ok, "detail": why}
    active = [be for be in all_backends if be.available()[0]]

    meta = {
        "gpu": str(torch.cuda.get_device_name(0)),
        "torch": str(torch.__version__),  # TorchVersion is a torch subclass of str
        "cuda": str(torch.version.cuda),
        "n_configs": len(configs),
        "warmup": warmup,
        "iters": iters,
        "backward": backward,
        "active_backends": [b.name for b in active],
    }

    results = []
    for i, cfg in enumerate(configs):
        for be in active:
            results.append(benchmark_one(be, cfg, warmup, iters, backward))
        print(f"[{i + 1}/{len(configs)}] {cfg.name} done", flush=True)
        if progress_cb is not None:
            # checkpoint partial results so a client/network drop can't lose work
            progress_cb({
                "meta": {**meta, "completed_configs": i + 1, "status": "running"},
                "availability": availability,
                "correctness": [],
                "results": results,
            })

    correctness = check_correctness("bf16") if do_correctness else []
    payload = {
        "meta": {**meta, "completed_configs": len(configs), "status": "done"},
        "availability": availability,
        "correctness": correctness,
        "results": results,
    }
    if progress_cb is not None:
        progress_cb(payload)
    return payload
