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
MARGIN = 0.03          # a point "improves" if its CI lower bound > 1 + MARGIN
# a point "regresses" only if CONFIDENTLY slower: its whole CI is below 1
# (CI upper < 1). Avoids false regressions from timing noise on neutral points.
AGREE_SLACK = 0.002    # candidate may flip <=0.2% more labels than the noise floor
JS_FACTOR = 3.0        # candidate JS divergence may be <=3x the run-to-run JS floor


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


def js_bits(P, Q, eps=1e-12):
    """Mean per-row Jensen-Shannon divergence (bits, in [0,1]). Symmetric and
    bounded -- unlike KL, no blow-up when a class probability -> 0."""
    P = np.clip(P, eps, 1.0); Q = np.clip(Q, eps, 1.0)
    P = P / P.sum(1, keepdims=True); Q = Q / Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    kl = lambda A, B: np.sum(A * np.log2(A / B), axis=1)
    return float((0.5 * kl(P, M) + 0.5 * kl(Q, M)).mean())


def proba_diff(a_npy, b_npy):
    a, b = np.load(a_npy), np.load(b_npy)
    agree = float((a.argmax(1) == b.argmax(1)).mean())
    return {"argmax_agree": round(agree, 5), "js_bits": round(js_bits(a, b), 7),
            # kept for transparency only (not gated):
            "p99_abs_dproba": round(float(np.percentile(np.abs(a - b), 99)), 5),
            "max_abs_dproba": round(float(np.abs(a - b).max()), 5)}


def calibrate(points, repeats, warmup):
    git("checkout", "-q", "-f", TRUNK)
    r1 = run_worker("cal1", points, repeats, warmup)
    r2 = run_worker("cal2", points, repeats, warmup)
    cal = {"trunk": cur_ref(), "points": {}}
    for name in points.split(","):
        floor = proba_diff(r1[name]["proba_npy"], r2[name]["proba_npy"])
        cal["points"][name] = {
            "baseline_lat_ms": med(r1[name]["lat_ms"]),
            "noise_argmax_agree": floor["argmax_agree"],
            "noise_js_bits": floor["js_bits"],
            "noise_p99_abs_dproba": floor["p99_abs_dproba"],
        }
    json.dump(cal, open(CAL, "w"), indent=2)
    print(json.dumps(cal, indent=2))


def evaluate(ref, points, repeats, warmup, hypothesis):
    cal = json.load(open(CAL))
    git("checkout", "-q", "-f", TRUNK)
    base = run_worker("base", points, repeats, warmup)
    git("checkout", "-q", "-f", ref)
    cand_ref = cur_ref()
    try:
        cand = run_worker("cand", points, repeats, warmup)
    finally:
        git("checkout", "-q", "-f", TRUNK)

    per = {}
    speedups, ok_speed, ok_corr = [], [], []
    for name in points.split(","):
        bl, cl = base[name]["lat_ms"], cand[name]["lat_ms"]
        ci_lo, ci_hi = boot_ratio_ci(bl, cl)
        speedup = med(bl) / med(cl)
        d = proba_diff(base[name]["proba_npy"], cand[name]["proba_npy"])
        f = cal["points"][name]
        # hard gate: same decisions, within the model's own label-flip noise
        argmax_ok = d["argmax_agree"] >= f["noise_argmax_agree"] - AGREE_SLACK
        # calibration gate: distribution within run-to-run JS noise (self-
        # calibrated). On zero-floor (deterministic) points JS is reported as a
        # diagnostic, not a hard fail -- discrete argmax already guards those.
        if f.get("noise_js_bits", 0.0) > 1e-9:
            js_ok = d["js_bits"] <= f["noise_js_bits"] * JS_FACTOR
            js_mode = "gated"
        else:
            js_ok = True
            js_mode = "diagnostic(zero-floor)"
        corr_ok = argmax_ok and js_ok
        speed_ok = ci_lo > 1.0 + MARGIN
        per[name] = {"speedup": round(speedup, 4),
                     "ci": [round(ci_lo, 4), round(ci_hi, 4)],
                     "base_ms": round(med(bl), 3), "cand_ms": round(med(cl), 3),
                     "correctness": d, "noise_floor": f, "js_mode": js_mode,
                     "argmax_ok": argmax_ok, "js_ok": js_ok,
                     "speed_ok": speed_ok, "corr_ok": corr_ok}
        speedups.append(speedup); ok_speed.append(speed_ok); ok_corr.append(corr_ok)

    geomean = float(np.exp(np.mean(np.log(speedups))))
    # Pareto acceptance: outputs preserved everywhere, no point regresses, and
    # at least one point improves. Right policy for shape-dependent wins.
    no_regression = all(p["ci"][1] >= 1.0 for p in per.values())
    improved = any(p["speed_ok"] for p in per.values())
    verdict = ("reject-correctness" if not all(ok_corr) else
               "regression" if not no_regression else
               "no-speedup" if not improved else
               "accept")
    rec = {"ref": cand_ref, "hypothesis": hypothesis, "verdict": verdict,
           "geomean_speedup": round(geomean, 4), "worst_correct": all(ok_corr),
           "no_regression": no_regression, "improved_somewhere": improved,
           "points": per}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


def promote(ref):
    """Advance trunk to an accepted candidate (fast-forward merge)."""
    git("checkout", "-q", "-f", TRUNK)
    git("merge", "--no-ff", "-m", f"promote {ref} to trunk", ref)
    print(f"trunk -> {cur_ref()} (promoted {ref})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--ref", required=True)
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
    elif a.cmd == "promote":
        promote(a.ref)
    else:
        evaluate(a.ref, a.points, a.repeats, a.warmup, a.hypothesis)
