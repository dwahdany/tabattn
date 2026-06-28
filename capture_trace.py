"""Capture a torch chrome trace of one predict for whatever tabpfn is checked out.
Used to render µs-scale kernel-occupancy timelines per optimization step.

    python capture_trace.py --tag 0_stock_typical --point typical
"""
import argparse
import gzip
import os
import shutil

os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/hf/hub")

import torch
from sklearn.datasets import make_classification
from huggingface_hub import list_repo_files, hf_hub_download
from tabpfn import TabPFNClassifier
from torch.profiler import profile, ProfilerActivity

HF_REPO = "Prior-Labs/tabpfn_3"
PTS = {"typical": (4096, 2048, 64), "sample": (16384, 2048, 64)}


def ckpt():
    files = list_repo_files(HF_REPO)
    c = [f for f in files if f.endswith(".ckpt") and "classifier" in f.lower()]
    fn = ([f for f in c if "multiclass" in f.lower()] or c)[0]
    return hf_hub_download(HF_REPO, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--point", default="typical", choices=list(PTS))
    a = ap.parse_args()
    nt, nte, nf = PTS[a.point]
    path = ckpt()
    X, y = make_classification(n_samples=nt + nte, n_features=nf,
                               n_informative=max(2, nf // 2), n_classes=2,
                               random_state=0)
    clf = TabPFNClassifier(device="cuda", n_estimators=1, model_path=path,
                           ignore_pretraining_limits=True, random_state=0)
    clf.fit(X[:nt], y[:nt])
    Xte = X[nt:]
    for _ in range(3):           # warm up (incl. torch.compile / cuda-graph capture)
        clf.predict(Xte)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        clf.predict(Xte)
    torch.cuda.synchronize()
    out = f"/workspace/out/trace_{a.tag}.json"
    prof.export_chrome_trace(out)
    with open(out, "rb") as fi, gzip.open(out + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(out)
    print("wrote", out + ".gz")


if __name__ == "__main__":
    main()
