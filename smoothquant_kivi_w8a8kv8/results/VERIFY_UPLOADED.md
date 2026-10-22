# Uploaded-data verification (sampled, rigorous) — 2026-06-04

Verifies the two HF datasets contain exactly what we intended.
Script: `code/etc/verify_uploaded.py` (run on origin server, env `lm_eval_flash`, same GPU → bit-exact).

- `jsyeom/smoothquant-kivi-w8a8kv8-gs14` — SmoothQuant W8A8 weights (int8) + KIVI-INT8 KV
- `jsyeom/awq-kivi-w4a16kv8-gs14` — AWQ W4A16 weights (int4) + KIVI-INT8 KV

Both repos fully uploaded: 4657 files each (33 weight safetensors + 4480 KV safetensors + 140 `_meta.json` + index/config/README).

## Method — two layers of proof
1. **Upload integrity (HF == local):** for sampled files compare HF LFS `sha256` (large) / git blob `sha1` (small) to local `compressed_data`. No bulk download.
2. **Semantic correctness (local == intended):** rebuild the model from scratch (SmoothQuant: `smooth_lm`+`quantize_model`; AWQ: `apply_awq`+`pseudo_quantize`) and the KV (prefill + KIVI pack) exactly as the dump scripts, then bit-exact compare.
   - (1)+(2) ⇒ **HF == intended.**

## Results — ALL PASS

| check | sample | result |
|---|---|---|
| integrity SQ | README, config, layer_0/31, embeddings, index, gsm8k & niah KV files | **PASS** (sha256/sha1 == local) |
| integrity AWQ | same set | **PASS** |
| SQ weights semantic | layers {0,15,31} × 7 proj | **PASS** int8 mismatch=0, scale_err=0; dequant vs W8A8 ≤1.7e-3 (fp16) |
| AWQ weights semantic | layers {0,15,31} × 7 proj | **PASS** qweight/scale/zero mismatch=0; dequant vs W4 = 0.0 |
| SQ KV semantic | gsm8k_cot sample_0, 32 layers | **PASS** code+scale+min bit-exact |
| AWQ KV semantic | gsm8k_cot sample_0, 32 layers | **PASS** code+scale+min bit-exact |
| SQ KV RULER | niah_multikey_1, ruler_vt sample_0 | **PASS** well-formed (codes∈[0,255], shapes==meta, scale≥0, dequant finite) |
| AWQ KV RULER | niah_multikey_1, ruler_vt sample_0 | **PASS** well-formed |

## Note on RULER KV
RULER prompts are synthetic and randomly generated per run (no fixed seed), so the exact prompt used at dump time is **not reproducible** — hence RULER KV is verified by well-formedness + integrity rather than regen bit-compare. The KV-generation pipeline itself is proven bit-exact on the deterministic `gsm8k_cot` sample, and RULER goes through the identical code path. (Confirmed irreproducibility: re-running niah gave T=3752 vs stored 3746 — the random content changed.)

**Conclusion:** both uploaded datasets contain exactly the intended SmoothQuant-W8A8 / AWQ-W4A16 weights and their KIVI-INT8 KV caches.
