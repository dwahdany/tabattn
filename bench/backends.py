"""Attention backend adapters behind a single interface.

Canonical layout is BHSD = (batch, heads, seq, head_dim). Each backend owns the
conversion to/from its native layout. The benchmark times only ``run`` on
tensors already in the native layout, so layout-conversion cost is excluded and
we measure the kernel itself.

All backends use the default softmax scale (1/sqrt(head_dim)) and full
(non-causal) bidirectional attention, which is the representative primitive for
tabular FMs (rows/cols attend to each other without a causal mask).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAS_SDPA_KERNEL = True
except Exception:  # very old torch
    _HAS_SDPA_KERNEL = False


class Backend:
    name = "base"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def prepare(self, q, k, v, requires_grad: bool = False):
        """Convert canonical BHSD q/k/v into this backend's native leaves."""
        raise NotImplementedError

    def run(self, prepared):
        """Run attention on prepared inputs; returns output in native layout."""
        raise NotImplementedError

    def to_canonical(self, out):
        """Convert a native-layout output back to BHSD (for correctness checks)."""
        return out


def _clone_leaves(tensors, requires_grad):
    out = []
    for x in tensors:
        t = x.clone()
        t.requires_grad_(requires_grad)
        out.append(t)
    return out


def _to_bshd_leaves(tensors, requires_grad):
    # BHSD (B,H,S,D) -> BSHD (B,S,H,D), as fresh leaves
    out = []
    for x in tensors:
        t = x.transpose(1, 2).contiguous()
        t.requires_grad_(requires_grad)
        out.append(t)
    return out


class EagerBackend(Backend):
    """Naive matmul + softmax + matmul in plain PyTorch. Materializes the SxS
    attention matrix, so it OOMs on long sequences -- the baseline that motivates
    fused kernels."""

    name = "eager"

    def prepare(self, q, k, v, requires_grad=False):
        return _clone_leaves((q, k, v), requires_grad)

    def run(self, prepared):
        q, k, v = prepared
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, v)


class SdpaBackend(Backend):
    """torch.nn.functional.scaled_dot_product_attention, forced to one kernel."""

    def __init__(self, sdp_backend, name):
        self._sdp_backend = sdp_backend  # SDPBackend enum
        self.name = name

    def available(self):
        if not torch.cuda.is_available():
            return False, "cuda unavailable"
        if not _HAS_SDPA_KERNEL:
            return False, "torch.nn.attention.sdpa_kernel unavailable"
        return True, ""

    def prepare(self, q, k, v, requires_grad=False):
        return _clone_leaves((q, k, v), requires_grad)  # native layout is BHSD

    def run(self, prepared):
        q, k, v = prepared
        with sdpa_kernel([self._sdp_backend]):
            return F.scaled_dot_product_attention(q, k, v)


class FlashAttnBackend(Backend):
    """Dao-AILab flash-attn (flash_attn_func). Native layout BSHD, fp16/bf16 only."""

    name = "flash-attn"

    def __init__(self):
        try:
            from flash_attn import flash_attn_func
            self._fn = flash_attn_func
            self._ok, self._err = True, ""
        except Exception as e:  # not installed / ABI mismatch
            self._fn, self._ok, self._err = None, False, repr(e)

    def available(self):
        if not self._ok:
            return False, self._err
        if not torch.cuda.is_available():
            return False, "cuda unavailable"
        return True, ""

    def prepare(self, q, k, v, requires_grad=False):
        return _to_bshd_leaves((q, k, v), requires_grad)

    def run(self, prepared):
        q, k, v = prepared
        return self._fn(q, k, v, dropout_p=0.0, causal=False)

    def to_canonical(self, out):
        return out.transpose(1, 2).contiguous()


class XformersBackend(Backend):
    """xformers.ops.memory_efficient_attention. Native layout BSHD."""

    name = "xformers"

    def __init__(self):
        try:
            import xformers.ops as xops
            self._xops = xops
            self._ok, self._err = True, ""
        except Exception as e:
            self._xops, self._ok, self._err = None, False, repr(e)

    def available(self):
        if not self._ok:
            return False, self._err
        if not torch.cuda.is_available():
            return False, "cuda unavailable"
        return True, ""

    def prepare(self, q, k, v, requires_grad=False):
        return _to_bshd_leaves((q, k, v), requires_grad)

    def run(self, prepared):
        q, k, v = prepared
        return self._xops.memory_efficient_attention(q, k, v)

    def to_canonical(self, out):
        return out.transpose(1, 2).contiguous()


def build_backends() -> list[Backend]:
    """All candidate backends (availability is checked separately per-backend)."""
    backends: list[Backend] = [EagerBackend()]
    if _HAS_SDPA_KERNEL:
        backends.append(SdpaBackend(SDPBackend.MATH, "sdpa-math"))
        backends.append(SdpaBackend(SDPBackend.FLASH_ATTENTION, "sdpa-flash"))
        backends.append(SdpaBackend(SDPBackend.EFFICIENT_ATTENTION, "sdpa-mem-eff"))
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.append(SdpaBackend(SDPBackend.CUDNN_ATTENTION, "sdpa-cudnn"))
    backends.append(FlashAttnBackend())
    backends.append(XformersBackend())
    return backends
