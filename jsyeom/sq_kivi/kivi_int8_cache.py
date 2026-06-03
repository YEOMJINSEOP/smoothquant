"""KiviINT8Cache — KIVI-style INT8 KV-cache quantization as a transformers Cache.

Orthogonal combination: the base model keeps its SmoothQuant W8A8 linears untouched;
this Cache injects KIVI's KV quantization at `update()` (the only interposition point in
the HF generation loop, validated in Phase 0).

KIVI scheme (asymmetric min-max, per group):
  - Key   : per-CHANNEL  → group along the TOKEN axis (chunks of group_size tokens)
  - Value : per-TOKEN    → group along the HEAD_DIM axis (chunks of group_size channels)
  - the most recent `residual_length` tokens are kept in fp16 (not quantized)

Since KIVI's CUDA/Triton kernels only support 2/4-bit, INT8 uses a FAKE-QUANT path
(quantize -> dequantize -> fp16), which is exact for accuracy evaluation. The math here
is identical to KIVI's `quant/new_pack.py` (verified by `verify_against_kivi()`); packing
to int32 is skipped for speed (only Task-2 saving needs the packed form).

Note: for accuracy eval we intentionally KEEP the full fp16 cache in the parent
DynamicCache and re-derive the quantized view each step. This measures the *accuracy
effect* of INT8 KV; it does not realize KIVI's memory savings (not the goal here).
"""
from typing import Any, Optional
import torch
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def _fake_quant_along(x: torch.Tensor, group_dim: int, group_size: int, n_bits: int) -> torch.Tensor:
    """Asymmetric min-max fake-quant of x, grouping `group_dim` into chunks of group_size.

    Returns a tensor of the same shape as x, dequantized (q*scale + mn).
    Requires x.shape[group_dim] % group_size == 0.
    """
    qmax = (1 << n_bits) - 1
    shape = x.shape
    L = shape[group_dim]
    assert L % group_size == 0, f"dim {group_dim} (={L}) must be divisible by group_size {group_size}"
    ng = L // group_size
    new_shape = shape[:group_dim] + (ng, group_size) + shape[group_dim + 1:]
    g = x.reshape(new_shape)
    red = group_dim + 1  # the group_size axis to reduce over
    mn = g.min(dim=red, keepdim=True)[0]
    mx = g.max(dim=red, keepdim=True)[0]
    scale = (mx - mn) / qmax
    scale = scale.clamp(min=1e-8)
    q = ((g - mn) / scale).round().clamp_(0, qmax)
    deq = q * scale + mn
    return deq.reshape(shape)


class KiviINT8Cache(DynamicCache):
    def __init__(self, k_bits: int = 8, v_bits: int = 8, group_size: int = 32,
                 residual_length: int = 128, **kwargs):
        super().__init__(**kwargs)
        assert residual_length % group_size == 0, "residual_length must be a multiple of group_size"
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length

    @torch.no_grad()
    def _quant_key(self, k: torch.Tensor) -> torch.Tensor:
        # k: [B, nkv, T, D]; quantize per-channel (group along T), keep last residual fp16
        T = k.shape[-2]
        if T <= self.residual_length:
            return k
        n_quant = ((T - self.residual_length) // self.group_size) * self.group_size
        if n_quant == 0:
            return k
        kq = _fake_quant_along(k[:, :, :n_quant, :], group_dim=2,
                               group_size=self.group_size, n_bits=self.k_bits)
        return torch.cat([kq, k[:, :, n_quant:, :]], dim=2)

    @torch.no_grad()
    def _quant_value(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, nkv, T, D]; quantize per-token (group along D), keep last residual fp16
        T = v.shape[-2]
        if T <= self.residual_length:
            return v
        n_quant = T - self.residual_length
        vq = _fake_quant_along(v[:, :, :n_quant, :], group_dim=3,
                               group_size=self.group_size, n_bits=self.v_bits)
        return torch.cat([vq, v[:, :, n_quant:, :]], dim=2)

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Accumulate full fp16 in parent, then return the KIVI-INT8 fake-quant view.
        k_full, v_full = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return self._quant_key(k_full), self._quant_value(v_full)


@torch.no_grad()
def verify_against_kivi(device="cuda"):
    """Confirm our direct fake-quant equals KIVI's pack->unpack at bits=8."""
    import sys
    sys.path.insert(0, "/SSD/JSY/KIVI")
    from quant.new_pack import (quant_and_pack_kcache, unpack_and_dequant_kcache,
                                quant_and_pack_vcache, unpack_and_dequant_vcache)
    B, nkv, T, D = 1, 8, 256, 128
    gs, bits = 32, 8
    k = torch.randn(B, nkv, T, D, dtype=torch.float16, device=device)
    v = torch.randn(B, nkv, T, D, dtype=torch.float16, device=device)

    # KIVI reference (pack then unpack) on a group_size-aligned slice
    kc, ks, kmn = quant_and_pack_kcache(k, gs, bits)
    k_kivi = unpack_and_dequant_kcache(kc, ks, kmn, gs, bits)
    vc, vs, vmn = quant_and_pack_vcache(v, gs, bits)
    v_kivi = unpack_and_dequant_vcache(vc, vs, vmn, gs, bits)

    # Ours (no residual carve-out here; quantize all, to match KIVI's full-tensor quant)
    k_ours = _fake_quant_along(k, group_dim=2, group_size=gs, n_bits=bits)
    v_ours = _fake_quant_along(v, group_dim=3, group_size=gs, n_bits=bits)

    dk = (k_kivi.float() - k_ours.float()).abs().max().item()
    dv = (v_kivi.float() - v_ours.float()).abs().max().item()
    print(f"key  max|KIVI - ours| = {dk:.3e}")
    print(f"value max|KIVI - ours| = {dv:.3e}")
    ok = dk < 1e-2 and dv < 1e-2
    print("MATCH" if ok else "MISMATCH", "(fp16 rounding tolerance)")
    return ok


if __name__ == "__main__":
    verify_against_kivi()
