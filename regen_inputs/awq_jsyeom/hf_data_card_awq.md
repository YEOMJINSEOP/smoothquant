---
tags:
  - quantization
  - awq
  - kivi
  - kv-cache
base_model: meta-llama/Llama-3.1-8B-Instruct
---

# AWQ W4A16 + KIVI-INT8 KV (Llama-3.1-8B-Instruct)

## `w4_awq_llama_31_8b/` — W4 weights

The **4-bit weights of the AWQ W4A16 model** (asymmetric / standard AWQ, group size 128).
Built with the official AWQ search (`apply_awq`) then per-group asymmetric quantization:
`scale = (max-min)/15`, `zero = clamp(-round(min/scale), 0, 15)`, `code = clamp(round(w/scale)+zero, 0, 15)`,
dequant `w ≈ (code - zero) × scale`. Stored per layer: `layer_0.safetensors … layer_31.safetensors` + `embeddings.safetensors`.

The 7 linears per layer are quantized (packed int4 codes + fp16 per-group scale + uint8 per-group zero); everything else stays fp16:

| key                                                         | dtype | shape                                   |
| ----------------------------------------------------------- | ----- | --------------------------------------- |
| `*.qweight` (2 int4 codes / byte; lo nibble = even column)  | uint8 | (out_features, in_features / 2)         |
| `*.scale` (per group)                                       | fp16  | (out_features, in_features / 128)       |
| `*.zero` (per group, asymmetric zero-point)                 | uint8 | (out_features, in_features / 128)       |
| `input_layernorm.weight`, `post_attention_layernorm.weight` | fp16  | (4096,)                                 |
| `model.embed_tokens.weight`, `lm_head.weight`               | fp16  | (128256, 4096)                          |
| `model.norm.weight`                                         | fp16  | (4096,)                                 |

Per-layer linears: `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` (k/v_proj out=1024 GQA).
Dequant in fp16 reproduces the model's W4 fake-quant weight bit-exactly.

## `kv_kivi8_of_w4a16_awq_llama_31_8b/` — KIVI-INT8 KV cache

The **KV cache of the same AWQ W4A16 model, quantized with KIVI INT8** (asymmetric; Key per-channel, Value per-token; `group_size=32`, `residual_length=128`, 8-bit). Identical KV scheme to the SmoothQuant variant — only the underlying model differs.

Layout: `<task>/sample_<n>/layer_<i>.safetensors`, each holding `{k_code, k_scale, k_min, v_code, v_scale, v_min}` (codes packed int32, scale/min fp16). Snapshot taken right after prefill; only the packed-INT8 portion is stored (recent 128-token fp16 residual excluded).

| key                | dtype | shape (T = sequence length)  |
| ------------------ | ----- | ---------------------------- |
| `k_code`           | int32 | (1, 8, n_quant_k/4, 128)     |
| `k_scale`, `k_min` | fp16  | (1, 8, n_quant_k/32, 1, 128) |
| `v_code`           | int32 | (1, 8, n_quant_v, 128/4)     |
| `v_scale`, `v_min` | fp16  | (1, 8, n_quant_v, 128/32, 1) |

`n_quant_k = floor((T-128)/32)*32`, `n_quant_v = T-128`.

Tasks (20 samples each): RULER@4K — `niah_multikey_1`, `ruler_vt`, `ruler_cwe`, `ruler_fwe`, `ruler_qa_squad`; plus `gsm8k_cot` and `longbench_hotpotqa`.
