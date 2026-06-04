"""FP8 (E4M3) per-tensor scaled fake-quant primitives + an FP8 KV-cache.

Project 3 — type-conversion baseline:  W = FP8(E4M3),  A = FP16,  KV = FP8(E4M3).

Unlike SmoothQuant (INT8) / AWQ (INT4), this is a *type conversion*: cast fp16 -> fp8
with a single per-tensor amax scale so the fp8 dynamic range (+-448 for e4m3fn) is used:

    scale = |x|.amax() / 448            # per-tensor (one scalar per weight / per K / per V)
    code  = (x / scale).to(float8)      # round-to-nearest fp8
    dequant: code.to(fp16) * scale

Compute path is FAKE-QUANT (fp8 -> fp16 dequant -> fp16 matmul), matching projects 1/2,
so activations stay FP16 and the comparison is apples-to-apples. Native fp8 GEMM is a
separate concern and would contradict "A-FP16".
"""
import torch
from transformers.cache_utils import DynamicCache

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0

# Same set of quantized Linears as projects 1/2 (embed / layernorms / lm_head kept fp16).
QPROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


@torch.no_grad()
def quant_per_tensor_fp8(x, dtype=FP8_DTYPE):
    """Per-tensor scaled cast of x (fp16/fp32) to fp8.

    Returns (code: fp8 tensor same shape as x, scale: fp32 scalar tensor shape [1]).
    Dequant = code.to(fp16) * scale.
    """
    fmax = torch.finfo(dtype).max
    amax = x.detach().abs().amax().clamp(min=1e-8).to(torch.float32)
    scale = (amax / fmax).reshape(1)
    code = (x.to(torch.float32) / scale).clamp(-fmax, fmax).to(dtype)
    return code, scale


@torch.no_grad()
def dequant_fp8(code, scale, out_dtype=torch.float16):
    return code.to(out_dtype) * scale.to(out_dtype)


@torch.no_grad()
def fake_quant_per_tensor_fp8(x, dtype=FP8_DTYPE):
    """quant -> dequant round-trip, returned in x's dtype (the inference weight/KV)."""
    code, scale = quant_per_tensor_fp8(x, dtype)
    return dequant_fp8(code, scale, x.dtype)


@torch.no_grad()
def apply_fp8_weight_fakequant(model, dtype=FP8_DTYPE):
    """In-place: replace each QPROJ Linear weight with its per-tensor fp8 fake-quant."""
    for i in range(model.config.num_hidden_layers):
        for proj in QPROJ:
            lin = model.get_submodule(f"model.layers.{i}.{proj}")
            lin.weight.data = fake_quant_per_tensor_fp8(lin.weight.data, dtype)
    return model


class Fp8KVCache(DynamicCache):
    """KV cache stored as per-tensor scaled FP8 (E4M3), dequantized to fp16 for attention.

    FAKE-QUANT: keep the full fp16 cache in the parent DynamicCache and return the fp8
    fake-quant view each step (one per-tensor scale for the layer's K, one for its V,
    recomputed from the current cache). This measures the accuracy effect of an FP8 KV
    cache; it does not realize the memory saving (not the goal here). No fp16 residual:
    the whole KV is cast to fp8 (standard fp8 KV-cache behavior, unlike KIVI).
    """
    def __init__(self, fp8_dtype=FP8_DTYPE, **kwargs):
        super().__init__(**kwargs)
        self.fp8_dtype = fp8_dtype

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_full, v_full = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return (fake_quant_per_tensor_fp8(k_full, self.fp8_dtype),
                fake_quant_per_tensor_fp8(v_full, self.fp8_dtype))
