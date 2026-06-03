#!/usr/bin/env bash
# =============================================================================
# regen_and_upload.sh — regenerate compressed_data for BOTH projects on a fast
# GPU server and upload to HuggingFace (private dataset repos), unattended.
#
#   Project 1 : SmoothQuant W8A8 + KIVI-INT8 KV   (Llama-3.1-8B-Instruct)
#   Project 2 : AWQ W4A16 (asym, g128) + KIVI-INT8 KV
#
# Usage:
#   export HF_TOKEN=hf_xxx            # MUST: gated-Llama read + dataset write
#   ./regen_and_upload.sh all        # sq + awq end-to-end
#   ./regen_and_upload.sh sq         # project 1 only
#   ./regen_and_upload.sh awq        # project 2 only
#
# Design: deterministic command sequence with HARD verification gates.
# Any gate failure aborts immediately (set -e). Re-running resumes uploads
# (upload_large_folder is resumable) and re-does regeneration (overwrites).
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
ROOT="${ROOT:-/SSD/JSY}"                     # clone all repos under here (path-replication = zero edits)
SQ="$ROOT/smoothquant"
KIVI="$ROOT/KIVI"
AWQ="$ROOT/llm-awq"
INPUTS="${INPUTS:-$SQ/regen_inputs}"         # small inputs shipped IN the smoothquant repo (git)
LMEVAL_DIR="${LMEVAL_DIR:-$ROOT/-ASSIGN-mask_lm_eval}"   # lm_eval fork (RULER tasks)
HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/hf_cache}"
SQ_REPO="${SQ_REPO:-jsyeom/smoothquant-kivi-w8a8kv8}"
AWQ_REPO="${AWQ_REPO:-jsyeom/awq-kivi-w4a16kv8}"
LIMIT="${LIMIT:-20}"                         # KV samples per task (matches origin artifact)
PHASE="${1:-all}"
LOG="$ROOT/regen_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run).log"

export HF_HUB_CACHE HF_DATASETS_CACHE="$HF_HUB_CACHE" HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="$SQ:$KIVI:$AWQ"

log(){ echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m" | tee -a "$LOG"; }
die(){ echo -e "\n\033[1;31mABORT: $*\033[0m" | tee -a "$LOG"; exit 1; }
gate(){ grep -qi "FAIL" "$1" && die "verification FAIL in $1 (see log)"; echo "  gate OK: no FAIL in $(basename "$1")"; }

# ---- preflight ---------------------------------------------------------------
preflight(){
  log "PREFLIGHT"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN not set (need gated-Llama read + dataset write)"
  for d in "$SQ" "$KIVI"; do [ -d "$d" ] || die "missing repo: $d (clone it — see HANDOFF.md)"; done
  python -c "import torch; assert torch.cuda.is_available()" || die "CUDA not available"
  python -c "import transformers, flash_attn, safetensors, huggingface_hub" || die "core deps missing"
  python -c "from lm_eval.tasks import TaskManager" || die "lm_eval fork not importable (RULER needed for KV prompts)"
  python -c "from huggingface_hub import HfApi; HfApi(token='$HF_TOKEN').whoami()" >/dev/null || die "HF_TOKEN invalid"
  # gated model reachable?
  python - <<PY || die "cannot access gated meta-llama/Llama-3.1-8B-Instruct with this token"
from huggingface_hub import HfApi
HfApi(token="$HF_TOKEN").model_info("meta-llama/Llama-3.1-8B-Instruct")
print("  gated model accessible")
PY
  free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  [ "${free_gb:-0}" -ge 150 ] || die "need >=150GB free under $ROOT (have ${free_gb}GB): model 16G + P1 51G + P2 ~45G"
  echo "  preflight OK (free ${free_gb}GB)"
}

# ---- project 1: SmoothQuant W8A8 + KIVI8 ------------------------------------
run_sq(){
  log "PROJECT 1 — SmoothQuant W8A8 + KIVI8"
  cp -f "$INPUTS/llama-3.1-8b-instruct.pt" "$SQ/act_scales/" 2>/dev/null || \
    echo "  (act_scales not in INPUTS; save_w8a8_weights will self-calibrate if needed)"
  cd "$SQ"
  local C="smoothquant_kivi_w8a8kv8/code"
  local OUTW="compressed_data/w8_of_w8a8_smoothquant_llama_31_8b"
  local OUTKV="compressed_data/kv_kivi8_of_w8a8_smoothquant_llama_31_8b"

  log "P1.1 save W8A8 weights"
  python "$C/save_w8a8_weights.py" --alpha 0.85 --out "$OUTW" 2>&1 | tee -a "$LOG"; gate "$LOG"
  python "$C/split_weights_per_layer.py" --dir "$OUTW" --remove-monolith 2>&1 | tee -a "$LOG"
  [ -f "$OUTW/layer_31.safetensors" ] && [ -f "$OUTW/embeddings.safetensors" ] || die "P1 weights incomplete"

  log "P1.2 dump KIVI8 KV (limit $LIMIT)"
  python "$C/dump_kv_cache.py" --limit "$LIMIT" --out "$OUTKV" 2>&1 | tee -a "$LOG"
  python "$C/etc/convert_kv_to_safetensors.py" --root "$OUTKV" 2>&1 | tee -a "$LOG"
  [ -f "$OUTKV/index.json" ] || die "P1 KV index.json missing"

  log "P1.3 verify_final (bit-exactness gate)"
  python "$C/etc/verify_final.py" 2>&1 | tee -a "$LOG"; gate "$LOG"

  cp -f "$SQ/$C/../compressed_data/README.md" "$SQ/compressed_data/" 2>/dev/null || \
    cp -f "$SQ/smoothquant_kivi_w8a8kv8/compressed_data/README.md" "$SQ/compressed_data/" 2>/dev/null || true

  log "P1.4 upload -> $SQ_REPO"
  python "$C/etc/upload_to_hf.py" --repo-id "$SQ_REPO" --path "$SQ/compressed_data" 2>&1 | tee -a "$LOG"
  echo "  P1 DONE -> https://huggingface.co/datasets/$SQ_REPO"
}

# ---- project 2: AWQ W4A16 + KIVI8 -------------------------------------------
run_awq(){
  log "PROJECT 2 — AWQ W4A16 + KIVI8"
  [ -d "$AWQ" ] || die "missing repo: $AWQ"
  # stage our scripts + official AWQ search cache from NAS (search is NOT re-run)
  [ -d "$AWQ/jsyeom" ] || cp -r "$INPUTS/awq_jsyeom" "$AWQ/jsyeom"
  mkdir -p "$AWQ/awq_cache"
  cp -f "$INPUTS/llama-3.1-8b-instruct-w4-g128.pt" "$AWQ/awq_cache/" || die "awq_cache .pt missing in INPUTS (do NOT re-run search)"
  touch "$AWQ/awq/__init__.py"                 # local awq beats installed AutoAWQ
  cd "$AWQ"
  local OUTW="compressed_data/w4_awq_llama_31_8b"
  local OUTKV="compressed_data/kv_kivi8_of_w4a16_awq_llama_31_8b"

  log "P2.1 save W4 weights"
  python jsyeom/save_w4_weights.py --out "$OUTW" 2>&1 | tee -a "$LOG"; gate "$LOG"
  [ -f "$OUTW/layer_31.safetensors" ] && [ -f "$OUTW/embeddings.safetensors" ] || die "P2 weights incomplete"

  log "P2.2 dump KIVI8 KV (limit $LIMIT)"
  python jsyeom/dump_kv_cache_awq.py --limit "$LIMIT" --out "$OUTKV" 2>&1 | tee -a "$LOG"
  [ -f "$OUTKV/index.json" ] || die "P2 KV index.json missing"

  log "P2.3 (optional) numerical sanity"
  python jsyeom/verify_combined_awq.py 2>&1 | tee -a "$LOG" || echo "  (verify optional; continuing)"

  cp -f "$AWQ/jsyeom/hf_data_card_awq.md" "$AWQ/compressed_data/README.md" 2>/dev/null || true

  log "P2.4 upload -> $AWQ_REPO"
  python jsyeom/upload_to_hf.py --repo-id "$AWQ_REPO" --path "$AWQ/compressed_data" 2>&1 | tee -a "$LOG"
  echo "  P2 DONE -> https://huggingface.co/datasets/$AWQ_REPO"
}

# ---- main --------------------------------------------------------------------
log "regen_and_upload.sh  phase=$PHASE  log=$LOG"
preflight
case "$PHASE" in
  sq)  run_sq ;;
  awq) run_awq ;;
  all) run_sq; run_awq ;;
  *)   die "unknown phase '$PHASE' (use: sq | awq | all)" ;;
esac
log "ALL DONE. repos:"
echo "  P1: https://huggingface.co/datasets/$SQ_REPO"
echo "  P2: https://huggingface.co/datasets/$AWQ_REPO"
