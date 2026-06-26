"""Worker: times predict + captures outputs for whatever tabpfn code is
currently checked out. Run in a fresh subprocess by the referee so code edits
are picked up. Pins random_state + fp16 for comparability.

    python -B pod_worker.py --tag base --points typical,sample --repeats 15 --warmup 3
"""
import argparse
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

HF_REPO = "Prior-Labs/tabpfn_3"
OUT = "/workspace/out"
os.makedirs(OUT, exist_ok=True)

POINTS = {
    "tiny":    dict(n_train=1024,  n_test=512,  n_features=32),
    "typical": dict(n_train=4096,  n_test=2048, n_features=64),
    "sample":  dict(n_train=16384, n_test=2048, n_features=64),
    "feature": dict(n_train=8192,  n_test=2048, n_features=512),
}


def ckpt(filename=""):
    if not filename:
        files = list_repo_files(HF_REPO)
        c = [f for f in files if f.endswith(".ckpt") and "classifier" in f.lower()]
        filename = ([f for f in c if "multiclass" in f.lower()] or c)[0]
    return hf_hub_download(HF_REPO, filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--points", default="typical,sample")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    path = ckpt()
    res = {}
    for name in a.points.split(","):
        cfg = POINTS[name]
        nt, nte, nf = cfg["n_train"], cfg["n_test"], cfg["n_features"]
        X, y = make_classification(n_samples=nt + nte, n_features=nf,
                                   n_informative=max(2, nf // 2), n_classes=2,
                                   random_state=0)
        Xtr, ytr, Xte = X[:nt], y[:nt], X[nt:]
        clf = TabPFNClassifier(device="cuda", n_estimators=1, model_path=path,
                               ignore_pretraining_limits=True, random_state=0)
        clf.fit(Xtr, ytr)
        torch.cuda.synchronize()
        for _ in range(a.warmup):
            clf.predict(Xte)
        torch.cuda.synchronize()
        lats = []
        for _ in range(a.repeats):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            clf.predict(Xte)
            torch.cuda.synchronize(); lats.append((time.perf_counter() - t0) * 1e3)
        proba = clf.predict_proba(Xte).astype(np.float32)
        ppath = f"{OUT}/proba_{a.tag}_{name}.npy"
        np.save(ppath, proba)
        res[name] = {"lat_ms": lats, "proba_npy": ppath,
                     "params": dict(n_train=nt, n_test=nte, n_features=nf)}
        print(f"[{a.tag}/{name}] median={sorted(lats)[len(lats)//2]:.2f}ms", flush=True)
        torch.cuda.empty_cache()
    json.dump(res, open(f"{OUT}/worker_{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
