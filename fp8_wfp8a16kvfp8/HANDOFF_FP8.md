# HANDOFF — regenerate & upload PROJECT 3 (W-FP8 / A-FP16 / KV-FP8) on a GPU server

You (the Claude Code instance on the other server) are taking over an upload job for
**Project 3**: Llama-3.1-8B-Instruct cast to **FP8 (E4M3, per-tensor)** weights + KV cache,
activations FP16. We **git-push the code**, then **regenerate `compressed_data` here and
upload to HuggingFace**. Regeneration is deterministic. Your job: set up, run the script,
watch the gates, and only declare done when the HF dataset repo holds the artifacts.

Deliverable (private HF **dataset** repo):
- `jsyeom/fp8-wfp8a16kvfp8-gs14` — W-FP8 weights + FP8 KV cache

> Self-contained: this project needs **ONLY** the smoothquant repo + the base model + an HF
> token. No KIVI, no AWQ, no `regen_inputs` (unlike projects 1/2). Nothing to re-search.

---

## 0. Secrets (operator provides as env vars — never commit these)
```bash
export HF_TOKEN=<hf write token, with gated meta-llama/Llama-3.1 read access>
export GH_TOKEN=<github PAT, only if the smoothquant repo is private>
```

## 1. Clone (replicate the same absolute path → zero path edits)
The scripts default to `/SSD/JSY/smoothquant`. Cloning there means **no edits**. Otherwise
clone anywhere and pass `SQ=/your/path/smoothquant` (and `ROOT=/your/path`) to the script.
```bash
mkdir -p /SSD/JSY && cd /SSD/JSY
git clone https://$GH_TOKEN@github.com/YEOMJINSEOP/smoothquant.git   # or without token if public
```

## 2. Environment (version pins = reproducibility)
```bash
conda create -n fp8 python=3.10 -y && conda activate fp8
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # fp8 needs a recent torch
pip install transformers==4.56.1 flash-attn==2.8.3 datasets==3.6.0 accelerate zstandard \
            wonderwords nltk jieba fuzzywuzzy rouge python-Levenshtein "huggingface_hub[hf_transfer]" \
            "safetensors>=0.4.3"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
pip install -e /SSD/JSY/-ASSIGN-mask_lm_eval/src/lm-eval   # lm_eval fork (RULER tasks) — clone if absent
```
Pins: torch 2.8.0+cu128 (must have `torch.float8_e4m3fn`), transformers 4.56.1, flash_attn 2.8.3,
lm_eval 0.4.9.1 (fork, for RULER prompts during the KV dump), safetensors ≥0.4.3 (fp8 support).

## 3. Run — SELF-HEALING (never stalls, auto-resumes from checkpoints)
```bash
cd /SSD/JSY/smoothquant/fp8_wfp8a16kvfp8
chmod +x regen_and_upload_fp8.sh

# (a) WHILE AWAKE — proves the whole pipeline works in ~1-2 min (model load, HF token,
#     gated-model access, lm_eval RULER prompt, fp8 weight+KV forward). Fix anything it flags.
./regen_and_upload_fp8.sh preflight

# (b) then launch detached so it survives logout/disconnect and runs overnight:
setsid nohup ./regen_and_upload_fp8.sh all >> regen_fp8.out 2>&1 &
tail -f regen_fp8.out
```
How it guarantees no morning restart:
- **checkpoints** (`$ROOT/.regen_fp8_ckpt/*.done`) → re-running resumes; finished stages skip.
- the **KV dump itself skips already-written samples**, so a mid-dump crash resumes per-sample.
- **retry+backoff** on every stage; **upload retries up to 100×** (resumable `upload_large_folder`).
- **outer loop** cycles until the repo verifies complete (or `MAX_CYCLES=300`).
- if it ever exits incomplete, just **re-run the same command** — it resumes.

What it produces (then uploads):
- `compressed_data/w_of_wfp8a16kvfp8_llama_31_8b/` — `layer_<i>.safetensors` (fp8 weight + fp32 scale + layernorms) ×32 + `embeddings.safetensors` + `config.json`
- `compressed_data/kv_fp8_of_wfp8a16kvfp8_llama_31_8b/` — `<task>/sample_<n>.safetensors` (per-sample, all layers) + `index.json`
- `compressed_data/README.md` (data card, copied from `hf_data_card.md`)

## 4. "Done" criteria — verify before declaring success
- `save_fp8_weights.py` printed the per-layer save line; `dump_kv_cache_fp8.py` wrote `index.json`.
- The repo lists the expected files:
```bash
python - <<'PY'
from huggingface_hub import HfApi; api=HfApi()
r="jsyeom/fp8-wfp8a16kvfp8-gs14"
f=api.list_repo_files(r,repo_type="dataset")
base=lambda x: x.rsplit("/",1)[-1]
print(r, len(f), "files",
      "| weight layers:", sum(base(x).startswith("layer_") and x.endswith(".safetensors") for x in f),
      "| embeddings:", any(x.endswith("embeddings.safetensors") for x in f),
      "| KV samples:", sum(base(x).startswith("sample_") and x.endswith(".safetensors") for x in f),
      "| index:", any(x.endswith("index.json") for x in f))
PY
```
  Expect 32 weight `layer_*.safetensors` + `embeddings.safetensors` + `config.json`,
  140 KV `…/sample_*.safetensors` (7 tasks × 20) + `index.json`. Total ≈ 45–60 GB.

## 5. Troubleshooting
| symptom | fix |
|---|---|
| `torch has no float8_e4m3fn` | torch too old → install torch ≥2.1 (we pin 2.8.0+cu128) |
| `CUDA not available` | check driver vs torch cu128; `nvidia-smi`; reinstall matching torch |
| gated model 401/403 | `HF_TOKEN` lacks Llama-3.1 access → accept license on HF, use that account's token |
| `lm_eval` import / no RULER | install the **fork** (`pip install -e .../src/lm-eval`), not pypi `lm-eval` |
| safetensors can't save fp8 | upgrade `safetensors>=0.4.3` |
| flash-attn build fails | match CUDA/torch; or set `attn_implementation="eager"` (slower; KV identical) |
| upload 502 / many-files | re-run script (resumable); keep `HF_HUB_ENABLE_HF_TRANSFER=1` |
| out of disk | need ≥100GB under ROOT (model 16G + artifacts ~50G) |

## 6. When finished
Report the repo URL, the `list_repo_files` counts, and the PASS lines.
**Tell the operator to rotate `HF_TOKEN` and `GH_TOKEN`** if they were shared in plaintext.
