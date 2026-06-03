"""Task 1 — save the W8A8 (SmoothQuant) weights of Llama-3.1-8B-Instruct.

Pipeline: load fp16 -> smooth_lm(alpha) -> per-channel INT8 weight quantization,
matching SmoothQuant's `quantize_weight_per_channel_absmax` exactly:
    scale = clamp(|w|.amax(dim=-1), 1e-5) / 127      # per output channel
    w_int8 = round(w / scale).clamp(-127, 127)
    (dequant: w_int8 * scale)

Saved (safetensors):
  - per quantized Linear (q,k,v,o,gate,up,down x 32 layers): "<name>.weight" int8 + "<name>.scale" fp16
  - non-quantized params (embed_tokens, layernorms, model.norm, lm_head): fp16
  - sidecar config.json with all metadata

Also verifies that dequant(int8*scale) == SmoothQuant's fake-quant weight (exact).
"""
import os, sys, json, argparse
sys.path.insert(0, "/SSD/JSY/smoothquant")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCALES = "act_scales/llama-3.1-8b-instruct.pt"
QPROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def quant_per_channel_int8(w):
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--out", default="compressed_data/w8_of_w8a8_smoothquant_llama_31_8b")
    ap.add_argument("--kivi-group-size", type=int, default=32)
    ap.add_argument("--kivi-residual-length", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cpu")
    print(f"smoothing (alpha={args.alpha})...")
    smooth_lm(model, torch.load(SCALES), args.alpha)
    # Quantize FIRST, then extract int8+scale from the fake-quant weights so the saved
    # artifact is bit-exactly the weights used at inference (fp16 rounding included).
    print("quantize_model (W8A8) ...")
    model = quantize_model(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False)

    tensors = {}
    n_layers = model.config.num_hidden_layers
    max_err = 0.0
    for i in range(n_layers):
        for proj in QPROJ:
            name = f"model.layers.{i}.{proj}"
            w_fq = model.get_submodule(name).weight.data  # fp16, == round(w/s)*s
            scale = w_fq.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
            q = (w_fq / scale).round().clamp(-127, 127)
            max_err = max(max_err, (q * scale - w_fq).abs().max().item())
            tensors[f"{name}.weight"] = q.to(torch.int8).contiguous()
            tensors[f"{name}.scale"] = scale.to(torch.float16).contiguous()
    print(f"max |dequant(int8*scale) - fakequant| = {max_err:.3e}  ->", "PASS" if max_err < 1e-3 else "FAIL")

    # non-quantized params (fp16)
    keep = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(n_layers):
        keep += [f"model.layers.{i}.input_layernorm.weight",
                 f"model.layers.{i}.post_attention_layernorm.weight"]
    sd = dict(model.named_parameters())
    for k in keep:
        if k in sd:
            tensors[k] = sd[k].data.to(torch.float16).contiguous()

    save_file(tensors, os.path.join(args.out, "w8a8_weights.safetensors"))

    config = {
        "base_model": MODEL,
        "smoothing_alpha": args.alpha,
        "act_scales_file": os.path.basename(SCALES),
        "weight_quant": "per_channel_int8_symmetric (q_max=127)",
        "act_quant": "per_token_int8 (applied at inference, not stored)",
        "quantize_bmm_input": False,
        "quantized_linears": QPROJ,
        "num_layers": n_layers,
        "kv_quant": {"method": "KIVI-INT8 (fake-quant)", "k_bits": 8, "v_bits": 8,
                     "group_size": args.kivi_group_size, "residual_length": args.kivi_residual_length,
                     "key": "per-channel", "value": "per-token"},
        "note": "Reconstruct inference model by: load base -> smooth_lm(alpha) -> quantize_model(W8A8). "
                "These int8+scale tensors == the fake-quant weights of that model.",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # report sizes
    int8_bytes = sum(t.numel() for k, t in tensors.items() if t.dtype == torch.int8)
    fp16_bytes = sum(t.numel() * 2 for k, t in tensors.items() if t.dtype == torch.float16)
    print(f"saved to {args.out}/  | int8 params: {int8_bytes/1e9:.2f}GB, fp16(scale+kept): {fp16_bytes/1e9:.2f}GB")


if __name__ == "__main__":
    main()
