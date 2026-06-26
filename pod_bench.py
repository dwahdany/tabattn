"""Standalone TabPFN-3 latency/observability harness for a dedicated GPU box
(no Modal). Reuses e2e_analysis. Subcommands: det | bench | hotspots.

    python pod_bench.py det
    python pod_bench.py bench [--points typical,sample]
    python pod_bench.py hotspots
"""
import argparse
import gzip
import json
import os
import time

os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/hf/hub")

import numpy as np
import torch
from sklearn.datasets import make_classification
from huggingface_hub import list_repo_files, hf_hub_download
from tabpfn import TabPFNClassifier

from e2e_analysis import analyze, classify_kernel

HF_REPO = "Prior-Labs/tabpfn_3"
OUT = "/workspace/out"
os.makedirs(OUT, exist_ok=True)

POINTS = {
    "tiny":    dict(n_train=1024,  n_test=512,  n_features=32),
    "typical": dict(n_train=4096,  n_test=2048, n_features=64),
    "sample":  dict(n_train=16384, n_test=2048, n_features=64),
    "feature": dict(n_train=8192,  n_test=2048, n_features=512),
}


def ckpt_path(filename=""):
    if not filename:
        files = list_repo_files(HF_REPO)
        cands = [f for f in files if f.endswith(".ckpt") and "classifier" in f.lower()]
        filename = ([f for f in cands if "multiclass" in f.lower()] or cands)[0]
    return filename, hf_hub_download(HF_REPO, filename)


def make_data(nt, nte, nf, seed=0):
    X, y = make_classification(n_samples=nt + nte, n_features=nf,
                               n_informative=max(2, nf // 2), n_classes=2,
                               random_state=seed)
    return X[:nt], y[:nt], X[nt:]


def mk(path, rs=0, n_est=1):
    return TabPFNClassifier(device="cuda", n_estimators=n_est, model_path=path,
                            ignore_pretraining_limits=True, random_state=rs)


def _dev_us(ev):
    for a in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(ev, a, None)
        if v:
            return v
    return 0.0


# --------------------------------------------------------------------------- #
def cmd_det(args):
    _, path = ckpt_path(args.filename)

    def diff(a, b):
        return {"max_abs_dproba": round(float(np.abs(a - b).max()), 6),
                "argmax_agree": round(float((a.argmax(1) == b.argmax(1)).mean()), 5)}

    out = []
    for nt, nte, nf in [(1024, 512, 32), (4096, 1024, 64), (16384, 1024, 64)]:
        Xtr, ytr, Xte = make_data(nt, nte, nf)
        c0 = mk(path, 0); c0.fit(Xtr, ytr)
        p1 = c0.predict_proba(Xte)
        p2 = c0.predict_proba(Xte)                              # A: same object
        cB = mk(path, 0); cB.fit(Xtr, ytr); pB = cB.predict_proba(Xte)  # B: fresh rs0
        cC = mk(path, 1); cC.fit(Xtr, ytr); pC = cC.predict_proba(Xte)  # C: rs1
        rec = {"size": f"{nt}x{nte}x{nf}", "A_same_object": diff(p1, p2),
               "B_fresh_rs0": diff(p1, pB), "C_diff_seed_rs1": diff(p1, pC)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
        torch.cuda.empty_cache()
    json.dump({"gpu": torch.cuda.get_device_name(0), "results": out},
              open(f"{OUT}/determinism.json", "w"), indent=2)


# --------------------------------------------------------------------------- #
def _component_hooks(model):
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
    ev, starts, handles = [], {}, []

    def pre(m, inp):
        e = torch.cuda.Event(enable_timing=True); e.record(); starts[id(m)] = e

    def mkpost(cat):
        def post(m, inp, out):
            s = starts.get(id(m))
            if s is None:
                return
            e = torch.cuda.Event(enable_timing=True); e.record()
            ev.append((cat, s, e))
        return post
    for m, cat in cat_of.items():
        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(mkpost(cat)))
    return ev, handles


def bench_point(cfg, path, repeats, warmup):
    nt, nte, nf = cfg["n_train"], cfg["n_test"], cfg["n_features"]
    key = f'{cfg["name"]}_{nt}x{nte}x{nf}'
    Xtr, ytr, Xte = make_data(nt, nte, nf)
    clf = mk(path, 0); clf.fit(Xtr, ytr)

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
    wall = walls[len(walls) // 2]
    peak = torch.cuda.max_memory_allocated() / 1e6

    # correctness vs stored baseline
    proba = clf.predict_proba(Xte)
    bpath = f"{OUT}/baseline_{key}.npy"
    if os.path.exists(bpath):
        base = np.load(bpath)
        corr = {"max_abs_dproba": float(np.abs(base - proba).max()),
                "argmax_agree": float((base.argmax(1) == proba.argmax(1)).mean())}
    else:
        np.save(bpath, proba); corr = {"baseline": "stored-now"}

    # component breakdown
    model = clf.models_[0]
    ev, handles = _component_hooks(model)
    ev.clear(); clf.predict(Xte); torch.cuda.synchronize()
    comp = {}
    for cat, s, e in ev:
        comp[cat] = comp.get(cat, 0.0) + s.elapsed_time(e)
    for h in handles:
        h.remove()
    comp = {k: round(v, 3) for k, v in sorted(comp.items(), key=lambda x: -x[1])}

    # chrome trace + systems analysis
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        clf.predict(Xte)
    torch.cuda.synchronize()
    tpath = f"{OUT}/{key}.chrome.json"
    prof.export_chrome_trace(tpath)
    analysis = analyze(json.load(open(tpath)), wall_ms=wall)
    with open(tpath, "rb") as fi, gzip.open(tpath + ".gz", "wb") as fo:
        fo.writelines(fi)
    os.remove(tpath)

    rep = {"point": cfg["name"], "key": key,
           "params": dict(n_train=nt, n_test=nte, n_features=nf),
           "wall_ms_median": round(wall, 3), "wall_ms_min": round(walls[0], 3),
           "peak_mem_mb": round(peak, 1), "correctness": corr,
           "component_breakdown_ms": comp, "systems": analysis}
    print(f'[{key}] wall={wall:.1f}ms gpu_busy={analysis["gpu_busy_union_ms"]:.1f} '
          f'busy={analysis["busy_fraction"]} launches={analysis["launches"]} '
          f'corr={corr}', flush=True)
    return rep


def cmd_bench(args):
    _, path = ckpt_path(args.filename)
    names = args.points.split(",") if args.points else list(POINTS)
    runs = []
    for name in names:
        runs.append(bench_point(dict(name=name, **POINTS[name]), path,
                                args.repeats, args.warmup))
        torch.cuda.empty_cache()
    payload = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "runs": runs}
    json.dump(payload, open(f"{OUT}/e2e_results.json", "w"), indent=2)
    print("WROTE", f"{OUT}/e2e_results.json", flush=True)


def cmd_hotspots(args):
    _, path = ckpt_path(args.filename)
    Xtr, ytr, Xte = make_data(2048, 256, 64)
    clf = mk(path, 0); clf.fit(Xtr, ytr)
    clf.predict(Xte); torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        clf.predict(Xte)
    torch.cuda.synchronize()
    ops = []
    for e in prof.key_averages():
        dt = _dev_us(e)
        if dt <= 0:
            continue
        ops.append({"op": e.key[:60],
                    "kind": "aten" if e.key.startswith("aten::") else "kernel",
                    "bucket": classify_kernel(e.key, ""),
                    "ms": round(dt / 1e3, 3), "count": e.count})
    ops.sort(key=lambda r: -r["ms"])
    res = {"by_aten_op": [o for o in ops if o["kind"] == "aten"][:20],
           "by_kernel": [o for o in ops if o["kind"] == "kernel"][:20]}
    json.dump(res, open(f"{OUT}/hotspots.json", "w"), indent=2)
    for o in res["by_aten_op"][:14]:
        print(f'[{o["bucket"]:>10}] {o["ms"]:>7.1f}ms x{o["count"]:<6} {o["op"]}',
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("det", "bench", "hotspots"):
        s = sub.add_parser(c)
        s.add_argument("--filename", default="")
        if c == "bench":
            s.add_argument("--points", default="")
            s.add_argument("--repeats", type=int, default=7)
            s.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    {"det": cmd_det, "bench": cmd_bench, "hotspots": cmd_hotspots}[a.cmd](a)
