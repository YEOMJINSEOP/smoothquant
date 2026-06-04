"""Task 1 — save the W-FP8 (E4M3, per-tensor scaled) weights of Llama-3.1-8B-Instruct.

Pipeline: load fp16 -> per-tensor fp8 cast of each QPROJ Linear weight:
    scale = |w|.amax() / 448            # per output matrix (one scalar)
    code  = (w / scale).to(float8_e4m3fn)
    (dequant: code.to(fp16) * scale)

Written directly in the final per-layer upload layout (resumable, no separate split step):
  <out>/layer_<i>.safetensors  -> 7 quantized linears ("<proj>.weight" fp8 + "<proj>.scale" fp32[1])
                                   + input_layernorm.weight + post_attention_layernorm.weight (fp16)
  <out>/embeddings.safetensors -> model.embed_tokens.weight, model.norm.weight, lm_head.weight (fp16)
  <out>/config.json

Since code+scale ARE the quantizer output, dequant(code,scale) == the inference weight
exactly (verified: max_err == 0).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
from fp8_quant import FP8_DTYPE, FP8_MAX, QPROJ, quant_per_tensor_fp8, dequant_fp8

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compressed_data/w_of_wfp8a16kvfp8_llama_31_8b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cpu")
    n_layers = model.config.num_hidden_layers
    sd = dict(model.named_parameters())

    quant_err = 0.0  # informational: how lossy the fp8 cast is vs the original fp16 weight
    for i in range(n_layers):
        layer = {}
        for proj in QPROJ:
            w = model.get_submodule(f"model.layers.{i}.{proj}").weight.data  # fp16
            code, scale = quant_per_tensor_fp8(w, FP8_DTYPE)
            quant_err = max(quant_err, (dequant_fp8(code, scale).float() - w.float()).abs().max().item())
            layer[f"{proj}.weight"] = code.contiguous()       # the inference weight = code.to(fp16)*scale
            layer[f"{proj}.scale"] = scale.to(torch.float32).contiguous()
        for ln in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            layer[ln] = sd[f"model.layers.{i}.{ln}"].data.to(torch.float16).contiguous()
        save_file(layer, os.path.join(args.out, f"layer_{i}.safetensors"))
    print(f"saved per-layer; max fp8 quant-error vs original fp16 weight = {quant_err:.3e} (informational)")

    emb = {}
    for k in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        if k in sd:
            emb[k] = sd[k].data.to(torch.float16).contiguous()
    save_file(emb, os.path.join(args.out, "embeddings.safetensors"))

    config = {
        "base_model": MODEL,
        "weight_quant": "per_tensor FP8 E4M3 (float8_e4m3fn), scale=|w|.amax()/448",
        "fp8_format": "e4m3fn", "fp8_max": FP8_MAX,
        "act_quant": "none (A=FP16)",
        "quantized_linears": QPROJ,
        "num_layers": n_layers,
        "kv_quant": {"method": "per_tensor FP8 E4M3 (fake-quant)", "fp8_format": "e4m3fn",
                     "scale": "per-tensor |x|.amax()/448", "residual": "none (whole KV cast)"},
        "storage": {"layout": "per-layer",
                    "files": "layer_<i>.safetensors (i=0..%d) + embeddings.safetensors" % (n_layers - 1),
                    "layer_file_keys": [f"{p}.weight (fp8 e4m3)" for p in QPROJ]
                                       + [f"{p}.scale (fp32[1])" for p in QPROJ]
                                       + ["input_layernorm.weight", "post_attention_layernorm.weight"],
                    "embeddings_file_keys": ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]},
        "note": "Reconstruct inference model by: load base fp16 -> for each QPROJ Linear, "
                "weight = code.to(fp16) * scale. code(fp8)+scale(fp32[1]) reproduce it exactly.",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    files = [f for f in os.listdir(args.out) if f.endswith(".safetensors")]
    sz = sum(os.path.getsize(os.path.join(args.out, f)) for f in files)
    print(f"saved {len(files)} safetensors to {args.out}/  | total {sz/1e9:.2f} GB")


if __name__ == "__main__":
    main()
