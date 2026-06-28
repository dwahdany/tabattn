"""Render µs-scale GPU kernel-occupancy timelines per optimization step.

For each trunk state's chrome trace (trace_<step>_<point>.json.gz), draw a
horizontal strip: every GPU kernel as a colored bar (by type) on a microsecond
time axis, zoomed to a representative steady-state window. Vanilla shows the
launch storm (many tiny kernels + idle gaps); fused/compiled steps are denser.

    uv run python plot_traces.py --point typical
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import e2e_analysis as ea  # classify_kernel, GPU_CATS

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

STEPS = ["0_stock", "1_compile-large", "2_contiguous", "3_base_scales",
         "4_fused_qkv", "5_dist_embed", "6_isab_contig", "7_icl_compile",
         "8_fused_rmsnorm"]
LABELS = {s: s.split("_", 1)[1] for s in STEPS}
LABELS["0_stock"] = "stock (vanilla)"

BUCKET_COLOR = {
    "attention": "#c44e52", "gemm": "#4c72b0", "norm": "#3a923a",
    "softmax": "#55a868", "elementwise": "#dd8452", "copy/cast": "#8172b3",
    "memcpy": "#937860", "memset": "#b0b0b0", "other": "#cfcfcf",
}
WINDOW_US = 1500.0   # width of the zoom window
OFFSET_FRAC = 0.0    # align all strips to the predict start (same phase)


def gpu_events(trace):
    evs = trace.get("traceEvents", trace)
    out = []
    for ev in evs:
        if ev.get("ph") != "X" or "dur" not in ev:
            continue
        cat = (ev.get("cat") or "").lower()
        if cat in ea.GPU_CATS:
            out.append((float(ev["ts"]), float(ev["dur"]),
                        ea.classify_kernel(ev.get("name", ""), cat)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="typical")
    ap.add_argument("--window", type=float, default=WINDOW_US)
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    yticks, ylabels = [], []
    for i, step in enumerate(STEPS):
        p = Path("traces") / f"trace_{step}_{a.point}.json.gz"
        if not p.exists():
            continue
        with gzip.open(p) as f:
            evs = gpu_events(json.load(f))
        if not evs:
            continue
        t0 = min(e[0] for e in evs)
        t1 = max(e[0] + e[1] for e in evs)
        win_start = t0 + OFFSET_FRAC * (t1 - t0)
        win_end = win_start + a.window
        y = len(STEPS) - 1 - i           # vanilla on top
        # busy fraction within the window (union of kernel intervals)
        iv = []
        per_bucket = {}
        for ts, dur, b in evs:
            s, e = max(ts, win_start), min(ts + dur, win_end)
            if e <= s:
                continue
            per_bucket.setdefault(b, []).append((s - win_start, e - s))
            iv.append((s, e))
        for b, spans in per_bucket.items():
            ax.broken_barh(spans, (y + 0.12, 0.76),
                           facecolors=BUCKET_COLOR.get(b, "#ccc"), linewidth=0)
        iv.sort()
        busy = 0.0
        ce = -1
        for s, e in iv:
            if s > ce:
                busy += e - s; ce = e
            elif e > ce:
                busy += e - ce; ce = e
        # full-predict launch-storm metrics (not windowed) — the decisive numbers
        nk_full = len(evs)
        med_us = float(np.median([d for _, d, _ in evs]))
        yticks.append(y + 0.5)
        ylabels.append(f"{i}  {LABELS[step]}")
        ax.text(a.window * 0.995, y + 0.62,
                f"{busy / a.window * 100:.0f}% busy in window",
                ha="right", va="center", fontsize=7.5, color="#333")
        ax.text(a.window * 0.995, y + 0.34,
                f"predict: {nk_full} kernels · median {med_us:.1f}µs",
                ha="right", va="center", fontsize=7.5, color="#777")

    ax.set_xlim(0, a.window)
    ax.set_ylim(0, len(STEPS))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.set_xlabel("time (µs) — representative steady-state window")
    ax.set_title(f"GPU kernel occupancy per optimization step  ({a.point} predict)")
    handles = [Patch(facecolor=c, label=b) for b, c in BUCKET_COLOR.items()
               if b in {"attention", "gemm", "norm", "elementwise", "copy/cast", "memcpy"}]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=6, frameon=False, fontsize=9)
    fig.text(0.5, 0.94, "Vanilla = launch storm (many tiny kernels + idle gaps); "
             "fused/compiled steps pack more work per launch.",
             ha="center", fontsize=9.5, color="#6b6b6b")
    fig.subplots_adjust(top=0.90, bottom=0.12)
    out = Path("figures") / f"launch_storm_{a.point}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
