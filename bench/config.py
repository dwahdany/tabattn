"""Shape configurations for the two attention regimes in tabular foundation models.

Tabular FMs (TabPFN, RPT-OSS, ...) run attention along two axes over a
(rows x columns) table that is embedded to (rows, cols, dim):

  * cross-row attention  -- for each column, tokens attend across all ROWS.
        sequence length = n_rows (LONG, up to ~10k samples in context)
        batch           = n_cols (number of features, modest)
  * cross-column attention -- for each row, tokens attend across all COLUMNS.
        sequence length = n_cols (SHORT, ~16-128 features)
        batch           = n_rows (LARGE, thousands of samples)

These are very different kernel regimes: cross-row is the classic long-sequence
case where FlashAttention shines; cross-column is many tiny attentions where
launch/overhead tends to dominate. That contrast is the whole point.

A ShapeConfig is expressed in the *canonical* BHSD layout
(batch, heads, seq, head_dim); backends transpose to their native layout.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ShapeConfig:
    name: str          # human label, e.g. "cross_row/std/rows=8192"
    regime: str        # "cross_row" | "cross_col"
    batch: int         # cross_row -> n_cols ; cross_col -> n_rows
    seq: int           # cross_row -> n_rows ; cross_col -> n_cols
    heads: int
    head_dim: int
    dtype: str         # "bf16" | "fp16" | "fp32"

    def asdict(self) -> dict:
        return asdict(self)


# Two head configurations: a "standard" modern transformer head and a
# TabPFN-v2-like small head (emsize 192 = 6 heads x 32).
HEAD_CONFIGS = [
    ("std", 8, 64),       # dim = 512
    ("tabpfn", 6, 32),    # dim = 192, TabPFN-v2-ish
]

# cross-row: vary number of rows (sequence length); cols (batch) held fixed.
CROSS_ROW_N_COLS = 32
CROSS_ROW_SEQS = [1024, 4096, 8192, 16384]

# cross-col: vary number of cols (sequence length); rows (batch) held fixed.
CROSS_COL_N_ROWS = 4096
CROSS_COL_SEQS = [16, 32, 64, 128]

DTYPES = ["bf16", "fp16"]


def default_sweep(quick: bool = False) -> list[ShapeConfig]:
    """Build the default benchmark sweep.

    quick=True returns a tiny smoke-test subset (one head config, two shapes,
    bf16 only) for fast iteration / CI.
    """
    head_configs = HEAD_CONFIGS[:1] if quick else HEAD_CONFIGS
    row_seqs = [4096] if quick else CROSS_ROW_SEQS
    col_seqs = [64] if quick else CROSS_COL_SEQS
    dtypes = ["bf16"] if quick else DTYPES

    cfgs: list[ShapeConfig] = []
    for dtype in dtypes:
        for tag, heads, head_dim in head_configs:
            for s in row_seqs:
                cfgs.append(ShapeConfig(
                    name=f"cross_row/{tag}/{dtype}/rows={s}",
                    regime="cross_row",
                    batch=CROSS_ROW_N_COLS,
                    seq=s,
                    heads=heads,
                    head_dim=head_dim,
                    dtype=dtype,
                ))
            for s in col_seqs:
                cfgs.append(ShapeConfig(
                    name=f"cross_col/{tag}/{dtype}/cols={s}",
                    regime="cross_col",
                    batch=CROSS_COL_N_ROWS,
                    seq=s,
                    heads=heads,
                    head_dim=head_dim,
                    dtype=dtype,
                ))
    return cfgs


def dtype_of(name: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
