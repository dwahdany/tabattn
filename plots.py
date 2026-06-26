"""Generate publication-quality figures from a results_*.json payload.

    uv run python plots.py [results_h100.json] [figures_dir]

Each figure makes one argument:
  1. scaling.png      -- how latency scales with sequence length (the regimes)
  2. ranking.png      -- who wins every config, at a glance (relative-to-best)
  3. utilization.png  -- how much of the H100 each backend actually uses
  4. memory.png       -- why the naive backends OOM (memory scales as S^2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

INK = "#1a1a1a"
SUBTLE = "#6b6b6b"

# Display order, colours, markers. Naive/materialising backends are greys;
# the fused/efficient kernels get vivid, distinct colours (cuDNN green and
# flash-attn red are the usual winners, so they pop).
BACKENDS = ["eager", "sdpa-math", "sdpa-flash", "sdpa-mem-eff",
            "sdpa-cudnn", "flash-attn", "xformers"]
COLOR = {
    "eager":        "#8f8f8f",
    "sdpa-math":    "#555555",
    "sdpa-flash":   "#4c72b0",
    "sdpa-mem-eff": "#dd8452",
    "sdpa-cudnn":   "#3a923a",
    "flash-attn":   "#c44e52",
    "xformers":     "#8172b3",
}
MARKER = {
    "eager": "o", "sdpa-math": "s", "sdpa-flash": "^", "sdpa-mem-eff": "D",
    "sdpa-cudnn": "P", "flash-attn": "X", "xformers": "v",
}
HEAD_LABEL = {"std": "std heads (8×64)", "tabpfn": "TabPFN heads (6×32)"}
REGIME_LABEL = {"cross_row": "cross-row  (attend over rows — long sequence)",
                "cross_col": "cross-column  (attend over features — short sequence)"}

H100_BF16_PEAK = 989.0   # TFLOP/s, dense bf16 tensor-core peak
H100_MEM_GB = 80.0


def load(path):
    return json.load(open(path))


def tag_of(cfg):  # "cross_row/std/bf16/rows=4096" -> "std"
    return cfg.split("/")[1]


def index(results):
    idx = {}
    for r in results:
        idx[(r["regime"], tag_of(r["config"]), r["dtype"], r["seq"], r["backend"])] = r
    return idx


def seqs_for(results, regime, tag, dtype):
    s = sorted({r["seq"] for r in results
                if r["regime"] == regime and tag_of(r["config"]) == tag
                and r["dtype"] == dtype})
    return s


def ms_fmt(v, _):
    if v >= 1:
        return f"{v:g}"
    return f"{v:g}"


def style_log_ax(ax):
    ax.grid(True, which="both", axis="both")
    ax.grid(True, which="minor", alpha=0.35)
    ax.tick_params(length=3)


# --------------------------------------------------------------------------- #
# Figure 1 - latency scaling (the hero)
# --------------------------------------------------------------------------- #
def fig_scaling(data, outdir, dtype="bf16"):
    results = data["results"]
    idx = index(results)
    regimes = ["cross_row", "cross_col"]
    tags = ["std", "tabpfn"]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    for i, tag in enumerate(tags):
        for j, regime in enumerate(regimes):
            ax = axes[i, j]
            seqs = seqs_for(results, regime, tag, dtype)
            # find per-seq winner for annotation
            winners = {}
            for s in seqs:
                cand = [(b, idx[(regime, tag, dtype, s, b)]["fwd_ms"])
                        for b in BACKENDS
                        if idx[(regime, tag, dtype, s, b)]["status"] == "ok"]
                if cand:
                    winners[s] = min(cand, key=lambda x: x[1])

            for b in BACKENDS:
                xs, ys = [], []
                oom_x = []
                for s in seqs:
                    r = idx[(regime, tag, dtype, s, b)]
                    if r["status"] == "ok":
                        xs.append(s); ys.append(r["fwd_ms"])
                    else:
                        oom_x.append(s)
                is_winner_anywhere = any(w[0] == b for w in winners.values())
                lw = 2.6 if is_winner_anywhere else 1.5
                z = 5 if is_winner_anywhere else 3
                if xs:
                    ax.plot(xs, ys, color=COLOR[b], marker=MARKER[b], lw=lw,
                            ms=6.5 if is_winner_anywhere else 5, zorder=z,
                            markeredgecolor="white", markeredgewidth=0.6)
                # mark the first OOM with an x at an S^2-extrapolated height
                if oom_x and xs:
                    s0 = oom_x[0]
                    yhat = ys[-1] * (s0 / xs[-1]) ** 2
                    ax.plot([xs[-1], s0], [ys[-1], yhat], color=COLOR[b],
                            ls=":", lw=1.2, zorder=2)
                    ax.plot([s0], [yhat], marker="x", color=COLOR[b],
                            ms=9, mew=2.2, zorder=4)

            # summarise the winner honestly: a clear winner across all sizes,
            # or "varies with size" when no backend dominates the regime
            if winners:
                from collections import Counter
                tally = Counter(w[0] for w in winners.values())
                top, n = tally.most_common(1)[0]
                if n == len(winners):
                    txt, tc, ec = f"fastest: {top}", COLOR[top], COLOR[top]
                else:
                    txt, tc, ec = "fastest varies with size", INK, "#999999"
                ax.text(0.97, 0.04, txt, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=10,
                        fontweight="bold", color=tc,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=ec, lw=1.1, alpha=0.95))

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(seqs)
            ax.set_xticklabels([f"{s:,}" for s in seqs])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
            style_log_ax(ax)
            xname = "rows in context (sequence length)" if regime == "cross_row" \
                else "columns / features (sequence length)"
            if i == 1:
                ax.set_xlabel(xname)
            if j == 0:
                ax.set_ylabel(f"{HEAD_LABEL[tag]}\nforward latency  (ms)",
                              fontsize=10.5)
            if i == 0:
                ax.set_title(REGIME_LABEL[regime], pad=10)

            # note OOM backends
            oom_bs = [b for b in BACKENDS
                      if any(idx[(regime, tag, dtype, s, b)]["status"] == "oom"
                             for s in seqs)]
            if oom_bs:
                ax.text(0.02, 0.97, "✕ = out of memory",
                        transform=ax.transAxes, fontsize=8.5, color=SUBTLE,
                        va="top", ha="left")

    handles = [Line2D([0], [0], color=COLOR[b], marker=MARKER[b], lw=2.2,
                      ms=6, markeredgecolor="white", label=b) for b in BACKENDS]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.015), fontsize=10, columnspacing=1.4,
               handletextpad=0.4)
    fig.suptitle("Attention-backend latency scaling on an H100",
                 fontsize=16, fontweight="bold", x=0.5, y=1.07)
    fig.text(0.5, 1.005,
             "A different kernel wins each regime: cuDNN / flash-attn dominate "
             "long cross-row sequences;\nfor short cross-column sequences the "
             "fused kernels lose to a plain materialised matmul.",
             ha="center", fontsize=10.5, color=SUBTLE)
    fig.subplots_adjust(top=0.93, bottom=0.085, hspace=0.18, wspace=0.13)
    p = Path(outdir) / "scaling.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figure 2 - relative-to-best ranking heatmap
# --------------------------------------------------------------------------- #
def fig_ranking(data, outdir, dtype="bf16"):
    results = data["results"]
    idx = index(results)
    # column order: regime x head x seq
    cols = []
    for regime in ["cross_row", "cross_col"]:
        for tag in ["std", "tabpfn"]:
            for s in seqs_for(results, regime, tag, dtype):
                cols.append((regime, tag, s))

    M = np.full((len(BACKENDS), len(cols)), np.nan)
    status = np.empty((len(BACKENDS), len(cols)), dtype=object)
    for cj, (regime, tag, s) in enumerate(cols):
        oks = [(bi, idx[(regime, tag, dtype, s, b)]["fwd_ms"])
               for bi, b in enumerate(BACKENDS)
               if idx[(regime, tag, dtype, s, b)]["status"] == "ok"]
        best = min(v for _, v in oks)
        for bi, b in enumerate(BACKENDS):
            r = idx[(regime, tag, dtype, s, b)]
            status[bi, cj] = r["status"]
            if r["status"] == "ok":
                M[bi, cj] = r["fwd_ms"] / best

    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    cmap = mpl.colormaps["RdYlGn_r"].copy()
    cmap.set_bad("#e8e8e8")
    norm = LogNorm(vmin=1.0, vmax=np.nanmax(M))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{s:,}" for _, _, s in cols], fontsize=9)
    ax.set_yticks(range(len(BACKENDS)))
    ax.set_yticklabels(BACKENDS, fontsize=10.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # cell annotations
    for bi in range(len(BACKENDS)):
        for cj in range(len(cols)):
            v = M[bi, cj]
            if np.isnan(v):
                lab = "OOM" if status[bi, cj] == "oom" else "n/a"
                ax.text(cj, bi, lab, ha="center", va="center",
                        fontsize=7.5, color="#9a9a9a")
                continue
            # white text on dark (slow) cells
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.5 else INK
            txt = "1.0×" if v < 1.05 else (f"{v:.0f}×" if v >= 9.5 else f"{v:.1f}×")
            ax.text(cj, bi, txt, ha="center", va="center", fontsize=7.6,
                    color=tc, fontweight="bold" if v < 1.05 else "normal")

    # group separators + top labels
    boundaries, labels, centers = [], [], []
    start = 0
    prev = None
    groups = [(regime, tag) for regime, tag, _ in cols]
    for k, g in enumerate(groups + [None]):
        if g != prev and prev is not None:
            boundaries.append(k - 0.5)
            labels.append(prev)
            centers.append((start + k - 1) / 2)
            start = k
        prev = g
    for bx in boundaries[:-0] if boundaries else []:
        pass
    for bx in boundaries:
        ax.axvline(bx, color="white", lw=3)
    for c, (regime, tag) in zip(centers, labels):
        nice = ("cross-row" if regime == "cross_row" else "cross-col") + \
               f"\n{tag}"
        ax.text(c, -0.75, nice, ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=INK)

    ax.set_xlabel("sequence length  (rows for cross-row, columns for cross-col)",
                  labelpad=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label("slowdown vs fastest backend  (1.0× = winner)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    for t in [1, 2, 5, 10, 50, 200]:
        if t <= norm.vmax:
            pass

    ax.set_title("Who wins every configuration  —  forward latency relative "
                 "to the fastest backend (H100, bf16)",
                 fontsize=13.5, pad=52, loc="left")
    fig.subplots_adjust(top=0.74, left=0.085, right=0.99, bottom=0.13)
    p = Path(outdir) / "ranking.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figure 3 - hardware utilisation (achieved TFLOP/s)
# --------------------------------------------------------------------------- #
def fig_utilization(data, outdir, dtype="bf16", tag="std"):
    results = data["results"]
    idx = index(results)
    regimes = ["cross_row", "cross_col"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)

    for j, regime in enumerate(regimes):
        ax = axes[j]
        seqs = seqs_for(results, regime, tag, dtype)
        for b in BACKENDS:
            xs, ys = [], []
            for s in seqs:
                r = idx[(regime, tag, dtype, s, b)]
                if r["status"] == "ok":
                    xs.append(s); ys.append(r["fwd_tflops"])
            if xs:
                ax.plot(xs, ys, color=COLOR[b], marker=MARKER[b], lw=1.9,
                        ms=5.5, markeredgecolor="white", markeredgewidth=0.6)
        ax.axhline(H100_BF16_PEAK, ls="--", color=INK, lw=1.1)
        ax.text(seqs[0], H100_BF16_PEAK * 0.95, "H100 bf16 peak (989 TFLOP/s)",
                fontsize=8.8, color=INK, va="top")
        ax.set_xscale("log", base=2)
        ax.set_xticks(seqs)
        ax.set_xticklabels([f"{s:,}" for s in seqs])
        ax.set_ylim(0, H100_BF16_PEAK * 1.08)
        style_log_ax(ax)
        ax.grid(True, which="both", axis="y")
        ax.set_title(REGIME_LABEL[regime], pad=8)
        ax.set_xlabel("rows in context" if regime == "cross_row"
                      else "columns / features")
        # peak utilisation callout for the best backend at max seq
        best = max(((b, idx[(regime, tag, dtype, seqs[-1], b)]["fwd_tflops"])
                    for b in BACKENDS
                    if idx[(regime, tag, dtype, seqs[-1], b)]["status"] == "ok"),
                   key=lambda x: x[1])
        ax.annotate(f"{best[1] / H100_BF16_PEAK * 100:.0f}% of peak",
                    xy=(seqs[-1], best[1]), xytext=(-4, 10),
                    textcoords="offset points", ha="right", fontsize=9.5,
                    fontweight="bold", color=COLOR[best[0]])
    axes[0].set_ylabel("achieved throughput  (TFLOP/s)")

    handles = [Line2D([0], [0], color=COLOR[b], marker=MARKER[b], lw=2,
                      ms=5.5, markeredgecolor="white", label=b) for b in BACKENDS]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.6, columnspacing=1.3)
    fig.suptitle("Long sequences are compute-bound; short ones are overhead-bound",
                 fontsize=15, fontweight="bold", y=1.05)
    fig.text(0.5, 0.96,
             f"Same kernels, {HEAD_LABEL[tag]}.  Cross-row climbs toward ~45% of "
             "the H100's peak; cross-column stays far below it — near-zero for "
             "few features, where launch overhead dominates.",
             ha="center", fontsize=10, color=SUBTLE)
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.06)
    p = Path(outdir) / "utilization.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Figure 4 - memory scaling explains the OOMs
# --------------------------------------------------------------------------- #
def fig_memory(data, outdir, dtype="bf16", tag="std"):
    results = data["results"]
    idx = index(results)
    regime = "cross_row"
    seqs = seqs_for(results, regime, tag, dtype)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    for b in BACKENDS:
        xs, ys = [], []
        oom_seqs = []
        for s in seqs:
            r = idx[(regime, tag, dtype, s, b)]
            if r["status"] == "ok":
                xs.append(s); ys.append(r["fwd_peak_mb"] / 1024.0)  # GB
            elif r["status"] == "oom":
                oom_seqs.append(s)
        if xs:
            ax.plot(xs, ys, color=COLOR[b], marker=MARKER[b], lw=2.0, ms=6,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4,
                    label=b)
        # quadratic extrapolation for the materialising backends -> shows the wall
        if oom_seqs and len(xs) >= 2:
            x = np.array(xs, float); y = np.array(ys, float)
            c2 = (y[-1] - y[0]) / (x[-1] ** 2 - x[0] ** 2)
            c0 = y[0] - c2 * x[0] ** 2
            xext = np.array([xs[-1]] + oom_seqs, float)
            yext = c0 + c2 * xext ** 2
            ax.plot(xext, yext, color=COLOR[b], ls=":", lw=1.5, zorder=3)
            ax.scatter(oom_seqs, (c0 + c2 * np.array(oom_seqs, float) ** 2),
                       marker="x", color=COLOR[b], s=90, linewidths=2.4, zorder=5)

    ax.axhline(H100_MEM_GB, ls="--", color=INK, lw=1.3)
    ax.text(seqs[0], H100_MEM_GB * 1.06, "H100 capacity = 80 GB  → OOM above this line",
            fontsize=9.5, color=INK, va="bottom", fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(seqs)
    ax.set_xticklabels([f"{s:,}" for s in seqs])
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g} GB" if v >= 1 else f"{v * 1024:g} MB"))
    style_log_ax(ax)
    ax.set_xlabel("rows in context (sequence length)")
    ax.set_ylabel("peak GPU memory")
    ax.legend(loc="lower right", ncol=2, fontsize=9.2,
              title="✕ = measured OOM (dotted = S² extrapolation)",
              title_fontsize=8.8, alignment="left")
    fig.suptitle("Why eager & math OOM: peak memory grows as the square of "
                 "sequence length", fontsize=13.5, fontweight="bold", y=1.04)
    fig.text(0.5, 0.965,
             f"cross-row, {HEAD_LABEL[tag]}.  Materialising the S×S score "
             "matrix is quadratic; fused kernels (flash / cuDNN / xformers) "
             "stay linear and flat.",
             ha="center", fontsize=10, color=SUBTLE)
    fig.subplots_adjust(top=0.88)
    p = Path(outdir) / "memory.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results_h100.json"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
    outdir.mkdir(exist_ok=True)
    data = load(src)
    print(f"Loaded {src}: {len(data['results'])} records on {data['meta']['gpu']}")
    for fn in (fig_scaling, fig_ranking, fig_utilization, fig_memory):
        p = fn(data, outdir)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
