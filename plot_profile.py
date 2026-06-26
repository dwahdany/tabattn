"""Stacked-area figure of TabPFN-3 inference cost composition vs context size.

Reads profile_sweep.json (produced from the Modal `sweep`) and writes
figures/profile_breakdown.png.

    uv run python plot_profile.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
INK, SUBTLE = "#1a1a1a", "#6b6b6b"

# bottom -> top; cross-row (sample) attention on top so its growth is the eye's
# last impression. Non-attention is grey; the three attention kinds are vivid.
COMPONENTS = [
    ("other",        "other (embed in/out · RMSNorm · copies/casts)", "#cccccc"),
    ("icl_mlp",      "ICL MLP",                                        "#dd8452"),
    ("isab",         "feature embedder (ISAB attention)",             "#8172b3"),
    ("colagg",       "cross-COLUMN attention (column aggregator)",    "#4c72b0"),
    ("iclattn",      "cross-ROW attention (ICL / sample)",            "#c44e52"),
]
K_ISAB = "1_feature_embedder(ISAB)"
K_COL = "2_column_aggregator(feature-attn)"
K_ATTN = "3a_icl_attention"
K_MLP = "3b_icl_mlp"


def components_ms(r):
    ms = r["ms"]
    fwd = r["fwd"]
    isab = ms.get(K_ISAB, 0.0)
    col = ms.get(K_COL, 0.0)
    attn = ms.get(K_ATTN, 0.0)
    mlp = ms.get(K_MLP, 0.0)
    other = max(0.0, fwd - isab - col - attn - mlp)
    return {"other": other, "icl_mlp": mlp, "isab": isab,
            "colagg": col, "iclattn": attn}, fwd


def stack(ax, xs, series, as_pct):
    base = np.zeros(len(xs))
    totals = np.array([sum(s[k] for k, _, _ in COMPONENTS) for s in series])
    for key, label, color in COMPONENTS:
        vals = np.array([s[key] for s in series], float)
        if as_pct:
            vals = 100 * vals / totals
        ax.fill_between(xs, base, base + vals, color=color, label=label,
                        lw=0.8, edgecolor="white", zorder=2)
        base = base + vals
    return base  # top edge


def main():
    data = json.load(open("profile_sweep.json"))
    runs = [r for r in data["runs"] if "fwd" in r]
    # n_train scaling at fixed 64 features
    rs = sorted((r for r in runs if r["n_features"] == 64),
                key=lambda r: r["n_train"])
    xs = [r["n_train"] for r in rs]
    series = [components_ms(r)[0] for r in rs]
    totals = [components_ms(r)[1] for r in rs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # left: 100% stacked (proportions)
    stack(axL, xs, series, as_pct=True)
    axL.set_ylim(0, 100)
    axL.set_ylabel("share of model forward (%)")
    axL.set_title("Composition shifts toward cross-row attention")
    # annotate cross-row share at the ends (nudged inward to avoid clipping)
    for i in (0, len(xs) - 1):
        pct = 100 * series[i]["iclattn"] / totals[i]
        dx, ha = (10, "left") if i == 0 else (-10, "right")
        axL.annotate(f"{pct:.0f}%", xy=(xs[i], 100 - pct / 2), xytext=(dx, 0),
                     textcoords="offset points", ha=ha, va="center",
                     fontsize=10, fontweight="bold", color="white")

    # right: absolute ms stacked (shows the total exploding)
    top = stack(axR, xs, series, as_pct=False)
    axR.set_ylabel("model-forward time (ms)")
    axR.set_ylim(0, max(totals) * 1.12)
    axR.set_title("...and the absolute cost is dominated by it at scale")
    for x, t in zip(xs, totals):
        axR.annotate(f"{t:.0f} ms", xy=(x, t), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=8.6,
                     color=INK)

    for ax in (axL, axR):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("training-set size in context  (n_train, 64 features)")
        ax.margins(x=0)
        ax.grid(True, axis="y", color="#e3e3e3", lw=0.7, zorder=0)

    handles, labels = axL.get_legend_handles_labels()
    # legend order top-of-stack first, for readability
    order = list(range(len(handles)))[::-1]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
               fontsize=9.6, columnspacing=1.6, handlelength=1.4)

    fig.suptitle("Where TabPFN-3 inference spends its time, vs context size",
                 fontsize=15.5, fontweight="bold", y=1.16)
    fig.text(0.5, 1.005,
             f"{data['gpu']} · {data['model']}.  As the in-context training set "
             "grows, cross-row (sample) attention goes from a sliver to ~half "
             "the forward pass —\nthe long-sequence regime the kernel benchmark "
             "targets. At moderate size, non-attention work (data movement, "
             "norms, embedders) dominates.",
             ha="center", fontsize=10, color=SUBTLE)
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.18)

    out = Path("figures") / "profile_breakdown.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
