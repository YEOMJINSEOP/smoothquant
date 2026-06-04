# Project 3 — W-FP8 / A-FP16 / KV-FP8 (type conversion)

Llama-3.1-8B-Instruct cast to **FP8 (E4M3)** weights and KV cache, activations kept FP16.
A *type-conversion* baseline to compare against the integer-quantization pipelines
(Project 1: SmoothQuant W8A8+KV-INT8, Project 2: AWQ W4A16+KV-INT8).

## Scheme
- **Weight**: per-tensor scaled FP8 E4M3. `scale = |w|.amax() / 448`, `code = (w/scale).to(float8_e4m3fn)`, dequant `code*scale`. Same 7 Linears/layer as Projects 1/2 (q,k,v,o,gate,up,down ×32); embed / layernorms / lm_head kept FP16.
- **Activation**: FP16 (not quantized).
- **KV cache**: per-tensor scaled FP8 E4M3 (one scale per layer's K, one per V). Whole KV cast — no FP16 residual (unlike KIVI).
- **Compute**: fake-quant (fp8 → fp16 dequant → fp16 matmul), matching Projects 1/2 so the comparison is apples-to-apples. Native fp8 GEMM is out of scope (would contradict A-FP16).

## Files
| file | task |
|---|---|
| `code/fp8_quant.py` | shared FP8 primitives + `apply_fp8_weight_fakequant` + `Fp8KVCache` |
| `code/save_fp8_weights.py` | **Task 1** — save W-FP8 as `layer_<i>.safetensors` (`<proj>.weight` fp8 + `<proj>.scale` fp32[1] + layernorms) ×32 + `embeddings.safetensors` + `config.json` |
| `code/dump_kv_cache_fp8.py` | **Task 2** — dump FP8 KV as `<task>/sample_<n>.safetensors` (`layer_<i>.{k_code,k_scale,v_code,v_scale}`, meta has `T`) + `index.json` |
| `code/eval_lmeval_fp8.py` | **Task 3** — accuracy eval: `original` (fp16) vs `combined` (W-FP8+KV-FP8) |
| `code/etc/upload_to_hf.py` | resumable HF upload (`upload_large_folder`) |
| `regen_and_upload_fp8.sh` | self-healing regen+upload (save → dump → upload), overnight-safe |
| `HANDOFF_FP8.md` | runbook for regenerating + uploading on another server |
| `hf_data_card.md` | HF dataset card (copied to `compressed_data/README.md` at upload) |

HF dataset repo: **`jsyeom/fp8-wfp8a16kvfp8-gs14`** (private). Saved artifacts (`compressed_data/`)
are git-ignored — regenerated deterministically on the upload server, not committed.

## Fair-comparison guarantees (shared with Projects 1/2)
Same base model (Llama-3.1-8B-Instruct, fp16, flash-attn-2, same GPU), same task groups
(`ruler` / `gsm8k` / `hotpot`) and per-group config (limit/batch/maxlen), and **seed locked**
(`--seed 0` → random/numpy/torch/fewshot + `PYTHONHASHSEED=0` re-exec) so RULER prompts are
reproducible and identical across variants. The only difference vs `original` is the FP8 cast.

## Run
```bash
cd code
# Task 1: weights
python save_fp8_weights.py --out compressed_data/w_of_wfp8a16kvfp8_llama_31_8b
# Task 2: KV (4K dump, 20 samples/task)
python dump_kv_cache_fp8.py --out compressed_data/kv_fp8_of_wfp8a16kvfp8_llama_31_8b
# Task 3: eval (per variant×group)
CUDA_VISIBLE_DEVICES=0 python eval_lmeval_fp8.py --variant original --group gsm8k
CUDA_VISIBLE_DEVICES=1 python eval_lmeval_fp8.py --variant combined --group gsm8k
```

## Status
All three tasks smoke-tested PASS: weight save dequant-err 0.0 (6.98 GB fp8 + 2.10 GB fp16);
KV dump well-formed (fp8 codes + per-tensor fp32 scale); combined eval gsm8k exact_match 1.0
(no NaN/garbage, `Fp8KVCache` active in generation).
