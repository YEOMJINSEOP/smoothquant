# HANDOFF — regenerate & upload both projects on a fast GPU server (unattended)

You (the Claude Code instance on the fast server) are taking over an upload job.
The origin server has slow external network, so we **git-push code + transfer small
inputs**, then **regenerate `compressed_data` here and upload to HuggingFace**.
Regeneration is deterministic (same code+seed+versions → same artifacts; bit-exactness
already verified on origin). Your job: set up, run the script, watch the gates, fix any
surprises, and only declare done when both HF dataset repos hold the artifacts.

Two deliverables (private HF **dataset** repos):
- **P1** `jsyeom/smoothquant-kivi-w8a8kv8` — SmoothQuant W8A8 + KIVI-INT8 KV (Llama-3.1-8B-Instruct)
- **P2** `jsyeom/awq-kivi-w4a16kv8` — AWQ W4A16 (asym, g128) + KIVI-INT8 KV

> **Do NOT re-run the AWQ search.** It is a version-sensitive grid search; the official
> result is shipped as `awq_cache/llama-3.1-8b-instruct-w4-g128.pt` — reuse it verbatim.

---

## 0. Secrets (operator provides as env vars — never commit these)
```bash
export HF_TOKEN=<hf write token, with gated meta-llama/Llama-3.1 read access>
export GH_TOKEN=<github PAT for cloning the private repos>
```

## 1. Clone (replicate the same absolute paths → zero path edits)
The scripts hardcode `/SSD/JSY/{smoothquant,KIVI,llm-awq}`. Cloning there means **no edits**.
If `/SSD/JSY` is unavailable, clone anywhere and pass `ROOT=...` to the script, then
`grep -rl /SSD/JSY <repos>` and fix `sys.path`/defaults accordingly.
```bash
mkdir -p /SSD/JSY && cd /SSD/JSY
git clone https://$GH_TOKEN@github.com/YEOMJINSEOP/smoothquant.git
git clone https://$GH_TOKEN@github.com/YEOMJINSEOP/KIVI.git          # YOUR fork; new_pack.py == origin (verified)
git clone https://github.com/mit-han-lab/llm-awq.git                  # upstream; our jsyeom/ comes from NAS
```

## 2. Small inputs (shipped INSIDE the smoothquant repo — no NAS needed)
`smoothquant/regen_inputs/` (cloned in step 1) already contains:
- `llama-3.1-8b-instruct.pt` (act_scales, 5MB)
- `llama-3.1-8b-instruct-w4-g128.pt` (official AWQ search, 95MiB) — **reused, not regenerated**
- `awq_jsyeom/` (our AWQ scripts → script copies these into `llm-awq/jsyeom/`)

The script reads them via `INPUTS=$SQ/regen_inputs` (default). Only external dep to fetch
separately is the **lm_eval fork** (RULER tasks) at `$LMEVAL_DIR` (git clone).

## 3. Environment (version pins = reproducibility)
```bash
conda create -n sqkivi python=3.10 -y && conda activate sqkivi
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.1 flash-attn==2.8.3 datasets==3.6.0 accelerate zstandard \
            wonderwords nltk jieba fuzzywuzzy rouge python-Levenshtein "huggingface_hub[hf_transfer]"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
pip install -e $LMEVAL_DIR/src/lm-eval        # lm_eval 0.4.9.1 fork (RULER built in)
```
Pins: torch 2.8.0+cu128, **transformers 4.56.1**, flash_attn 2.8.3, lm_eval 0.4.9.1(fork), datasets 3.6.0.

## 4. Run — SELF-HEALING (never stalls, auto-resumes from checkpoints)
```bash
cd /SSD/JSY/smoothquant/smoothquant_kivi_w8a8kv8
chmod +x regen_and_upload.sh
export LMEVAL_DIR=/SSD/JSY/-ASSIGN-mask_lm_eval        # adjust to actual path

# (a) WHILE AWAKE — proves the whole pipeline works in ~1-2 min (model load, HF token,
#     gated-model access, lm_eval RULER prompt, KIVI forward). Fix anything it flags.
./regen_and_upload.sh preflight

# (b) then launch detached so it survives logout/disconnect and runs overnight:
setsid nohup ./regen_and_upload.sh all >> regen.out 2>&1 &
tail -f regen.out
```
How it guarantees no morning restart:
- **checkpoints** (`$ROOT/.regen_ckpt/*.done`) → re-running resumes; finished stages skip.
- **retry+backoff** on every stage; **uploads retry up to 100×** (resumable `upload_large_folder`).
- **outer loop** keeps cycling until both repos verify complete (or `MAX_CYCLES=300`).
- **verification is advisory** — fp16-LSB diffs on a different GPU are logged, never fatal
  (the artifact is still a valid W8A8+KIVI / W4A16+KIVI product).
- if it ever exits incomplete, just **re-run the same command** — it picks up where it left off.

## 5. "Done" criteria — verify before declaring success
- `verify_final.py` printed **PASS** (no `FAIL`) for P1; `save_w4_weights` printed **PASS** for P2.
- Both repos list the expected files:
```bash
python - <<'PY'
from huggingface_hub import HfApi; api=HfApi()
for r in ["jsyeom/smoothquant-kivi-w8a8kv8","jsyeom/awq-kivi-w4a16kv8"]:
    f=api.list_repo_files(r,repo_type="dataset")
    print(r, len(f), "files",
          "| weights:", sum("layer_" in x and x.endswith(".safetensors") for x in f),
          "| has index/config:", any(x.endswith(("index.json","config.json")) for x in f))
PY
```
  Expect each repo to have 32 weight `layer_*.safetensors` + `embeddings.safetensors` + `config.json`,
  and KV `…/sample_*/layer_*.safetensors` + `index.json`. P1 ≈ 51GB, P2 ≈ 45GB.

## 6. Troubleshooting
| symptom | fix |
|---|---|
| `CUDA not available` | check driver vs torch cu128; `nvidia-smi`; reinstall matching torch |
| gated model 401/403 | `HF_TOKEN` lacks Llama-3.1 access → accept license on HF, use token of that account |
| `lm_eval` import / no RULER | install the **fork** (`pip install -e $LMEVAL_DIR/src/lm-eval`), not pypi `lm-eval` |
| `import awq` → AutoAWQ | ensure `touch /SSD/JSY/llm-awq/awq/__init__.py` and `/SSD/JSY/llm-awq` first on PYTHONPATH |
| RoPE error on AWQ search | N/A — we do NOT run search; reuse `awq_cache/*.pt` |
| flash-attn build fails | match CUDA/torch; or set `attn_implementation="eager"` (slower; KV identical) |
| upload 502 / many-files | re-run script (resumable); keep `HF_HUB_ENABLE_HF_TRANSFER=1` |
| out of disk | need ≥150GB under ROOT (model 16G + P1 51G + P2 45G) |

## 7. When finished
Report: both repo URLs, the `list_repo_files` counts, and the PASS lines from the gates.
**Tell the operator to rotate `HF_TOKEN` and `GH_TOKEN`** (they were shared in plaintext).
