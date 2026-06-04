# FP8 (E4M3) W-FP8 / A-FP16 / KV-FP8 — Llama-3.1-8B-Instruct

Per-tensor scaled **FP8 (E4M3)** weights and KV cache derived from
`meta-llama/Llama-3.1-8B-Instruct`. A type-conversion variant (weights and KV cast to FP8,
activations kept FP16), for comparison against INT quantization (SmoothQuant W8A8, AWQ W4A16).

## Quantization
- **Weights**: per-tensor scaled FP8 E4M3. `scale = |w|.amax() / 448`, `code = (w/scale).to(float8_e4m3fn)`, dequant `code.to(fp16) * scale`. Applied to q/k/v/o/gate/up/down (×32 layers); embeddings, layernorms, lm_head kept FP16.
- **KV cache**: per-tensor scaled FP8 E4M3 (one scale per layer's K, one per V). Whole KV cast — no FP16 residual.

## Contents
- `w_of_wfp8a16kvfp8_llama_31_8b/`
  - `layer_<i>.safetensors` — `<proj>.weight` (fp8 e4m3) + `<proj>.scale` (fp32[1]) for 7 projections, plus the two layernorms (fp16)
  - `embeddings.safetensors` — `embed_tokens` / `model.norm` / `lm_head` (fp16)
  - `config.json`
- `kv_fp8_of_wfp8a16kvfp8_llama_31_8b/`
  - `<task>/sample_<n>.safetensors` — `layer_<i>.{k_code,k_scale,v_code,v_scale}`; file metadata holds `T`
  - `index.json`

KV is a post-prefill snapshot over lm-eval prompts (7 tasks: niah_multikey_1, ruler_vt,
ruler_cwe, ruler_fwe, ruler_qa_squad, gsm8k_cot, longbench_hotpotqa; RULER at 4K, 20 samples/task).
