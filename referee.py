"""Referee for the autoresearch loop. Mechanical, untrusted-input: given a
candidate git ref in the editable tabpfn package, it measures speedup vs the
frozen trunk baseline and checks output-equivalence against a calibrated noise
floor, then appends a verdict to the ledger. Never trusts the proposer.

    python referee.py calibrate --points typical,sample
    python referee.py evaluate --ref <branch-or-commit> [--hypothesis "..."]

Anti-collapse properties:
  * baseline is re-measured from trunk every eval (no drift, no trust)
  * speedup gate = bootstrap CI lower-bound on median ratio > 1 + margin
  * correctness gate = candidate-vs-baseline within the model's own measured
    run-to-run noise floor (argmax-agreement + p99 |delta-proba|), per point
  * trunk is only advanced by an accepted, replicated candidate (promote)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

TPDIR = subprocess.check_output(
    [sys.executable, "-c", "import tabpfn,os;print(os.path.dirname(tabpfn.__file__))"]
).decode().strip()
OUT = "/workspace/out"
LEDGER = f"{OUT}/ledger.jsonl"
CAL = f"{OUT}/calibration.json"
TRUNK = "master"
WORKER = "/workspace/pod_worker.py"
MARGIN = 0.03          # require >3% speedup (CI lower bound)
AGREE_SLACK = 0.002    # candidate may flip <=0.2% more labels than the noise floor
DP_FACTOR = 1.5        # p99 |dproba| may be <=1.5x the noise floor


def git(*a):
    subprocess.run(["git", *a], cwd=TPDIR, check=True,
                   stdout=subprocess.DEVNULL)


def cur_ref():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=TPDIR).decode().strip()


def run_worker(tag, points, repeats, warmup):
    subprocess.run([sys.executable, "-B", WORKER, "--tag", tag, "--points",
                    points, "--repeats", str(repeats), "--warmup", str(warmup)],
                   check=True)
    return json.load(open(f"{OUT}/worker_{tag}.json"))


def med(x):
    return float(np.median(x))


def boot_ratio_ci(base, cand, n=2000, lo=2.5, hi=97.5, seed=0):
    """Bootstrap CI for speedup = median(base)/median(cand)."""
    rng = np.random.default_rng(seed)
    b, c = np.array(base), np.array(cand)
    rs = [np.median(rng.choice(b, b.size)) / np.median(rng.choice(c, c.size))
          for _ in range(n)]
    return float(np.percentile(rs, lo)), float(np.percentile(rs, hi))


def proba_diff(a_npy, b_npy):
    a, b = np.load(a_npy), np.load(b_npy)
    agree = float((a.argmax(1) == b.argmax(1)).mean())
    p99 = float(np.percentile(np.abs(a - b), 99))
    mx = float(np.abs(a - b).max())
    return {"argmax_agree": round(agree, 5), "p99_abs_dproba": round(p99, 5),
            "max_abs_dproba": round(mx, 5)}


def calibrate(points, repeats, warmup):
    git("checkout", "-q", TRUNK)
    r1 = run_worker("cal1", points, repeats, warmup)
    r2 = run_worker("cal2", points, repeats, warmup)
    cal = {"trunk": cur_ref(), "points": {}}
    for name in points.split(","):
        floor = proba_diff(r1[name]["proba_npy"], r2[name]["proba_npy"])
        cal["points"][name] = {
            "baseline_lat_ms": med(r1[name]["lat_ms"]),
            "noise_argmax_agree": floor["argmax_agree"],
            "noise_p99_abs_dproba": floor["p99_abs_dproba"],
            "noise_max_abs_dproba": floor["max_abs_dproba"],
        }
    json.dump(cal, open(CAL, "w"), indent=2)
    print(json.dumps(cal, indent=2))


def evaluate(ref, points, repeats, warmup, hypothesis):
    cal = json.load(open(CAL))
    git("checkout", "-q", TRUNK)
    base = run_worker("base", points, repeats, warmup)
    git("checkout", "-q", ref)
    cand_ref = cur_ref()
    try:
        cand = run_worker("cand", points, repeats, warmup)
    finally:
        git("checkout", "-q", TRUNK)

    per = {}
    speedups, ok_speed, ok_corr = [], [], []
    for name in points.split(","):
        bl, cl = base[name]["lat_ms"], cand[name]["lat_ms"]
        ci_lo, ci_hi = boot_ratio_ci(bl, cl)
        speedup = med(bl) / med(cl)
        d = proba_diff(base[name]["proba_npy"], cand[name]["proba_npy"])
        f = cal["points"][name]
        corr_ok = (d["argmax_agree"] >= f["noise_argmax_agree"] - AGREE_SLACK and
                   d["p99_abs_dproba"] <= f["noise_p99_abs_dproba"] * DP_FACTOR
                   + 1e-6)
        speed_ok = ci_lo > 1.0 + MARGIN
        per[name] = {"speedup": round(speedup, 4),
                     "ci": [round(ci_lo, 4), round(ci_hi, 4)],
                     "base_ms": round(med(bl), 3), "cand_ms": round(med(cl), 3),
                     "correctness": d, "noise_floor": f,
                     "speed_ok": speed_ok, "corr_ok": corr_ok}
        speedups.append(speedup); ok_speed.append(speed_ok); ok_corr.append(corr_ok)

    geomean = float(np.exp(np.mean(np.log(speedups))))
    verdict = ("accept" if all(ok_corr) and all(ok_speed) else
               "reject-correctness" if not all(ok_corr) else
               "no-speedup")
    rec = {"ref": cand_ref, "hypothesis": hypothesis, "verdict": verdict,
           "geomean_speedup": round(geomean, 4), "worst_correct": all(ok_corr),
           "points": per}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("calibrate", "evaluate"):
        s = sub.add_parser(c)
        s.add_argument("--points", default="typical,sample")
        s.add_argument("--repeats", type=int, default=15)
        s.add_argument("--warmup", type=int, default=3)
        if c == "evaluate":
            s.add_argument("--ref", default=TRUNK)
            s.add_argument("--hypothesis", default="")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.points, a.repeats, a.warmup)
    else:
        evaluate(a.ref, a.points, a.repeats, a.warmup, a.hypothesis)
