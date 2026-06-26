"""Local-side reporting: write CSV/JSON and print a readable table.

Pure stdlib so the local entrypoint needs nothing beyond `modal`.
"""
from __future__ import annotations

import csv
import json


def write_outputs(payload: dict, stem: str) -> tuple[str, str]:
    json_path = f"{stem}.json"
    csv_path = f"{stem}.csv"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    rows = payload["results"]
    if rows:
        # union of keys, stable-ish order
        preferred = ["backend", "regime", "config", "dtype", "batch", "seq",
                     "heads", "head_dim", "status", "fwd_ms", "fwd_tflops",
                     "fwd_peak_mb", "fwdbwd_ms", "fwdbwd_tflops", "error"]
        keys = [k for k in preferred if any(k in r for r in rows)]
        extra = sorted({k for r in rows for k in r} - set(keys))
        keys += extra
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
    return json_path, csv_path


def _fmt(v, nd=3):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_summary(payload: dict) -> None:
    meta = payload["meta"]
    print("\n" + "=" * 78)
    print(f"GPU: {meta['gpu']}   torch {meta['torch']} / cuda {meta['cuda']}")
    print(f"warmup={meta['warmup']} iters={meta['iters']} backward={meta['backward']}")
    print("=" * 78)

    print("\nBackend availability:")
    for name, info in payload["availability"].items():
        mark = "ok " if info["available"] else "NO "
        detail = "" if info["available"] else f"  -- {info['detail']}"
        print(f"  [{mark}] {name}{detail}")

    if payload.get("correctness"):
        print("\nCorrectness vs fp32 eager (small shape, lower max_abs_err is better):")
        for r in payload["correctness"]:
            if r["status"] == "ok":
                print(f"  {r['backend']:14s} max_abs_err={_fmt(r['max_abs_err'], 5)}")
            else:
                print(f"  {r['backend']:14s} {r['status']}: {r.get('detail', '')}")

    results = payload["results"]
    backends = meta["active_backends"]

    # Group rows by config, one row per shape, columns = backends (fwd_ms).
    by_config: dict[str, dict] = {}
    order: list[str] = []
    for r in results:
        c = r["config"]
        if c not in by_config:
            by_config[c] = {}
            order.append(c)
        by_config[c][r["backend"]] = r

    def cell(r):
        if r is None:
            return "-"
        if r.get("status") != "ok":
            return r.get("status", "err")[:7]
        return _fmt(r["fwd_ms"], 3)

    colw = max(12, max((len(b) for b in backends), default=12))
    header = "config".ljust(46) + "".join(b.rjust(colw) for b in backends)

    for regime in ("cross_row", "cross_col"):
        print(f"\n--- {regime}: forward latency (ms, median; lower=faster) ---")
        print(header)
        for c in order:
            if not c.startswith(regime):
                continue
            row = by_config[c]
            line = c.ljust(46)
            best = None
            for b in backends:
                r = row.get(b)
                if r is not None and r.get("status") == "ok":
                    if best is None or r["fwd_ms"] < best[1]:
                        best = (b, r["fwd_ms"])
            for b in backends:
                txt = cell(row.get(b))
                if best and b == best[0]:
                    txt = "*" + txt  # mark fastest
                line += txt.rjust(colw)
            print(line)

    # Winner tally
    print("\n--- fastest-backend tally (forward) ---")
    wins: dict[str, int] = {}
    for c in order:
        row = by_config[c]
        best = None
        for b in backends:
            r = row.get(b)
            if r is not None and r.get("status") == "ok":
                if best is None or r["fwd_ms"] < best[1]:
                    best = (b, r["fwd_ms"])
        if best:
            wins[best[0]] = wins.get(best[0], 0) + 1
    for b, n in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {b:14s} {n} configs")
    print()
