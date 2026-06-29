"""Plot the autoresearch trajectory: cumulative speedup vs optimization step.

Reads steps.json (median latency at each accepted trunk state, measured on the
pod) and ledger.jsonl (every attempt + verdict) -> figures/trajectory.png.

    uv run python plot_steps.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.edgecolor": "#444", "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
INK, SUBTLE = "#1a1a1a", "#6b6b6b"
GREEN, BLUE, RED = "#3a923a", "#4c72b0", "#c44e52"

# short labels for each accepted optimization (in order)
def labels_from_order(order):
    # order tokens look like "00_stock", "08_fused-rmsnorm" -> "stock", "fused-rmsnorm"
    out = []
    for i, o in enumerate(order):
        name = o.split("_", 1)[1] if "_" in o else o
        out.append(name if i == 0 else "+" + name)
    return out


def main():
    steps = json.load(open("steps.json"))
    order = steps["order"]
    STEP_LABELS = labels_from_order(order)
    st = steps["states"]
    pts = list(st[order[0]].keys())  # ['typical','sample']
    stock = st[order[0]]
    # cumulative speedup per state per point + geomean
    spd = {p: [stock[p] / st[o][p] for o in order] for p in pts}
    geo = [float(np.exp(np.mean([np.log(spd[p][k]) for p in pts])))
           for k in range(len(order))]

    ledger = [json.loads(l) for l in open("ledger.jsonl")]
    verdicts = [r["verdict"] for r in ledger]
    # staircase: cumulative geomean of trunk after each attempt
    acc = 0
    stair = []
    accept_pts = []  # (attempt_idx, geomean, step_index)
    for i, v in enumerate(verdicts):
        if v == "accept":
            acc += 1
            accept_pts.append((i + 1, geo[acc], acc))
        stair.append(geo[acc])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 5.4))

    # ---- left: trajectory over all attempts (staircase) ----
    xs = list(range(1, len(stair) + 1))
    axL.step([0] + xs, [geo[0]] + stair, where="post", color=INK, lw=2, zorder=3)
    # reject/no-speedup attempts as faint ticks on the line
    for i, v in enumerate(verdicts):
        if v != "accept":
            axL.plot(i + 1, stair[i], marker="x", ms=5, color="#bbb", zorder=2)
    # accepts as numbered green dots; names listed in the box (and on the right panel)
    for ai, (att, g, step) in enumerate(accept_pts):
        axL.plot(att, g, "o", ms=11, color=GREEN, zorder=5,
                 markeredgecolor="white", markeredgewidth=1)
        axL.annotate(str(step), xy=(att, g), ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white", zorder=6)
    axL.axhline(1.0, ls="--", color="#999", lw=1)
    legend_txt = "\n".join(f"{step}  {STEP_LABELS[step]}  ({g:.2f}×)"
                           for att, g, step in accept_pts)
    axL.text(0.97, 0.04, legend_txt, transform=axL.transAxes, ha="right",
             va="bottom", fontsize=8.4, color=GREEN, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREEN, lw=1, alpha=0.95))
    axL.set_xlabel("autoresearch attempt (proposer → critic → referee)")
    axL.set_ylabel("cumulative speedup of trunk vs stock (geomean)")
    axL.set_title("Trunk ratchets up only at verified accepts")
    axL.set_ylim(0.98, geo[-1] * 1.05)
    axL.grid(True, axis="y", color="#ececec", lw=0.7)
    axL.text(0.02, 0.97, "● accept (promoted)   ✕ rejected by gate",
             transform=axL.transAxes, va="top", fontsize=9, color=SUBTLE)

    # ---- right: per-optimization cumulative speedup, per point ----
    x = list(range(len(order)))
    axR.plot(x, spd["sample"], "-o", color=RED, lw=2, label="sample (16k×64)",
             markeredgecolor="white")
    axR.plot(x, spd["typical"], "-o", color=BLUE, lw=2, label="typical (4k×64)",
             markeredgecolor="white")
    axR.plot(x, geo, "--o", color=INK, lw=1.8, label="geomean", markeredgecolor="white")
    for k in (len(order) - 1,):
        axR.annotate(f"{spd['sample'][k]:.2f}×", (k, spd['sample'][k]),
                     xytext=(-4, 6), textcoords="offset points", ha="right",
                     fontsize=9.5, fontweight="bold", color=RED)
        axR.annotate(f"{spd['typical'][k]:.2f}×", (k, spd['typical'][k]),
                     xytext=(-4, -14), textcoords="offset points", ha="right",
                     fontsize=9.5, fontweight="bold", color=BLUE)
    axR.axhline(1.0, ls="--", color="#999", lw=1)
    axR.set_xticks(x)
    axR.set_xticklabels(STEP_LABELS, rotation=30, ha="right", fontsize=9)
    axR.set_ylabel("cumulative speedup vs stock")
    axR.set_xlabel("accepted optimization (stacked)")
    axR.set_title("Per-operating-point contribution")
    axR.grid(True, axis="y", color="#ececec", lw=0.7)
    axR.legend(loc="upper left", fontsize=9.5)

    fig.suptitle("Autonomous autoresearch on TabPFN-3: cumulative speedup over steps",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, 0.965, f"{len(verdicts)} attempts → {len(accept_pts)} verified "
             f"output-preserving accepts → {geo[-1]:.2f}× geomean "
             f"({spd['sample'][-1]:.2f}× sample, {spd['typical'][-1]:.2f}× typical)",
             ha="center", fontsize=10, color=SUBTLE)
    fig.subplots_adjust(top=0.86, wspace=0.24)

    out = Path("figures") / "trajectory.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}  (final geomean {geo[-1]:.3f}x)")


if __name__ == "__main__":
    main()
