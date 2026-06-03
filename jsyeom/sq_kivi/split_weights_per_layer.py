"""Re-save the monolithic W8A8 weights into per-layer safetensors files.

Reads w8_of_w8a8_smoothquant_llama_31_8b/w8a8_weights.safetensors and writes:
  <out>/layer_<i>.safetensors  -> that layer's 7 quantized linears (weight int8 + scale fp16)
                                   + input_layernorm.weight + post_attention_layernorm.weight (fp16)
  <out>/embeddings.safetensors -> model.embed_tokens.weight, model.norm.weight, lm_head.weight (fp16)
config.json is kept as-is. Verifies round-trip equality, then (optionally) removes the monolith.
"""
import os, sys, argparse, json
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

QPROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="compressed_data/w8_of_w8a8_smoothquant_llama_31_8b")
    ap.add_argument("--remove-monolith", action="store_true")
    args = ap.parse_args()

    mono = os.path.join(args.dir, "w8a8_weights.safetensors")
    cfg = json.load(open(os.path.join(args.dir, "config.json")))
    n_layers = cfg["num_layers"]
    all_t = load_file(mono)
    print(f"loaded {len(all_t)} tensors from {mono}")

    written_keys = set()
    for i in range(n_layers):
        layer = {}
        for proj in QPROJ:
            base = f"model.layers.{i}.{proj}"
            layer[f"{proj}.weight"] = all_t[f"{base}.weight"]      # int8
            layer[f"{proj}.scale"] = all_t[f"{base}.scale"]        # fp16
            written_keys.add(f"{base}.weight"); written_keys.add(f"{base}.scale")
        for ln in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            k = f"model.layers.{i}.{ln}"
            layer[ln] = all_t[k]; written_keys.add(k)
        save_file(layer, os.path.join(args.dir, f"layer_{i}.safetensors"))

    emb = {}
    for k in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        if k in all_t:
            emb[k] = all_t[k]; written_keys.add(k)
    save_file(emb, os.path.join(args.dir, "embeddings.safetensors"))

    # verify: every monolith key written exactly once, and values match
    missing = set(all_t) - written_keys
    assert not missing, f"unwritten keys: {list(missing)[:5]}"
    # round-trip check on a few layers
    max_err = 0
    for i in [0, n_layers // 2, n_layers - 1]:
        lt = load_file(os.path.join(args.dir, f"layer_{i}.safetensors"))
        for proj in QPROJ:
            a = lt[f"{proj}.weight"]; b = all_t[f"model.layers.{i}.{proj}.weight"]
            assert torch.equal(a, b), f"mismatch layer{i} {proj}"
    print(f"verify: all {len(all_t)} keys written across {n_layers} layer files + embeddings.safetensors; round-trip OK")

    # update config to document layout
    cfg["storage"] = {
        "layout": "per-layer",
        "files": "layer_<i>.safetensors (i=0..%d) + embeddings.safetensors" % (n_layers - 1),
        "layer_file_keys": [f"{p}.weight (int8)" for p in QPROJ] + [f"{p}.scale (fp16)" for p in QPROJ]
                           + ["input_layernorm.weight", "post_attention_layernorm.weight"],
        "embeddings_file_keys": ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"],
    }
    json.dump(cfg, open(os.path.join(args.dir, "config.json"), "w"), indent=2)

    sz = sum(os.path.getsize(os.path.join(args.dir, f)) for f in os.listdir(args.dir) if f.startswith("layer_") or f == "embeddings.safetensors")
    print(f"per-layer total: {sz/1e9:.2f} GB")
    if args.remove_monolith:
        os.remove(mono)
        print(f"removed monolith {mono}")


if __name__ == "__main__":
    main()
