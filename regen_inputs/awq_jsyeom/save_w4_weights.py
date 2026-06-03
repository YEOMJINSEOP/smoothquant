"""Project 2 Task 1 — save AWQ W4 (asymmetric, g128) weights of Llama-3.1-8B-Instruct.

Build: load fp16 -> apply_awq(our official search) -> pseudo_quantize W4 (asym, g128).
Extract per-group int4 codes + scale + zero (matching awq.quantizer.pseudo_quantize_tensor),
pack 2 nibbles/byte (true 4-bit), save per-layer safetensors.

Asymmetric AWQ (per group of `group_size` along in-features):
  scale = (max-min)/15 ;  zero = clamp(-round(min/scale), 0, 15) ;  code = clamp(round(w/scale)+zero, 0, 15)
  dequant: w ≈ (code - zero) * scale
"""
import os, sys, json, types, argparse
sys.modules.setdefault("awq_inference_engine", types.ModuleType("awq_inference_engine"))
sys.path.insert(0, "/SSD/JSY/llm-awq")
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
from awq.quantize.pre_quant import apply_awq, get_blocks, get_named_linears
from awq.quantize.quantizer import pseudo_quantize_tensor

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
AWQ = "/SSD/JSY/llm-awq/awq_cache/llama-3.1-8b-instruct-w4-g128.pt"
GS = 128


def extract_w4(w, group_size=GS):
    """Use awq pseudo_quantize_tensor's exact scale/zero/w_dq, recover int4 code from w_dq
    (code = w_dq/scale + zero, no re-rounding ambiguity → exact, matches the model)."""
    out, inf = w.shape; ng = inf // group_size
    w_dq, scale, zero = pseudo_quantize_tensor(
        w.clone(), n_bit=4, zero_point=True, q_group_size=group_size, get_scale_zp=True)
    # scale, zero: (out, ng) ; w_dq: (out, in) fp16
    sc = scale.reshape(out, ng, 1); zr = zero.reshape(out, ng, 1)
    code = (w_dq.reshape(out, ng, group_size) / sc + zr).round().clamp(0, 15).reshape(out, inf)
    return (code.to(torch.uint8), scale.reshape(out, ng).to(torch.float16),
            zero.reshape(out, ng).to(torch.uint8), w_dq)


def pack_nibbles(code):  # (out,in) uint8 [0,15] -> (out,in/2) uint8 (2 codes/byte)
    out, inf = code.shape
    c = code.reshape(out, inf // 2, 2)
    return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)


def dequant_from_packed(packed, scale, zero, group_size=GS):
    out = packed.shape[0]; inf = packed.shape[1] * 2
    lo = (packed & 0xF); hi = (packed >> 4) & 0xF
    code = torch.stack([lo, hi], -1).reshape(out, inf).float()
    ng = inf // group_size
    code = code.reshape(out, ng, group_size)
    return ((code - zero.float()[..., None]) * scale.float()[..., None]).reshape(out, inf)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/SSD/JSY/llm-awq/compressed_data/w4_awq_llama_31_8b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("load + apply_awq (extract W4 from post-awq ORIGINAL weight) ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cpu")
    apply_awq(model, torch.load(AWQ, map_location="cpu"))

    layers = get_blocks(model)
    n_layers = len(layers)
    KEEP = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    max_err = 0.0
    for i in range(n_layers):
        lin = get_named_linears(layers[i])  # name -> Linear (post-awq fp16, NOT yet quantized)
        tensors = {}
        for name, m in lin.items():
            w = m.weight.data  # original post-apply_awq fp16
            code, scale, zero, w_dq = extract_w4(w)
            packed = pack_nibbles(code)
            tensors[f"{name}.qweight"] = packed.contiguous()
            tensors[f"{name}.scale"] = scale.contiguous()
            tensors[f"{name}.zero"] = zero.contiguous()
            # verify in fp16 (the model's working dtype): dequant == exact W4 fake-quant weight
            deq = dequant_from_packed(packed, scale, zero).half()
            max_err = max(max_err, (deq - w_dq).abs().max().item())
        # layernorms (fp16)
        for ln in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            p = dict(layers[i].named_parameters())
            for k, v in p.items():
                if k.endswith(ln):
                    tensors[ln] = v.data.to(torch.float16).contiguous()
        save_file(tensors, os.path.join(args.out, f"layer_{i}.safetensors"))
    print(f"per-layer verify: max|dequant - W4fake| = {max_err:.3e} -> {'PASS' if max_err<1e-3 else 'FAIL'}")

    sd = dict(model.named_parameters())
    save_file({k: sd[k].data.to(torch.float16).contiguous() for k in KEEP if k in sd},
              os.path.join(args.out, "embeddings.safetensors"))
    json.dump({"base_model": MODEL, "method": "AWQ asymmetric W4 g128 (weight-only, W4A16)",
               "awq_search": os.path.basename(AWQ), "w_bit": 4, "q_group_size": GS, "zero_point": True,
               "format": "layer_<i>.safetensors: <proj>.qweight(uint8, 2 int4/byte; lo nibble=even col) + .scale(fp16,(out,in/128)) + .zero(uint8,(out,in/128)); dequant in fp16: ((code-zero)*scale).half() == model W4 weight (bit-exact)",
               "kv_quant": {"method": "KIVI-INT8 (fake-quant)", "k_bits": 8, "v_bits": 8, "group_size": 32,
                            "residual_length": 128, "key": "per-channel", "value": "per-token"},
               "num_layers": n_layers}, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print(f"saved → {args.out}/ (layer_0..{n_layers-1}.safetensors + embeddings.safetensors + config.json)")


if __name__ == "__main__":
    main()
