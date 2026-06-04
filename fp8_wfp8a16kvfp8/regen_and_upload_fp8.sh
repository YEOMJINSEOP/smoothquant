#!/usr/bin/env bash
# =============================================================================
# regen_and_upload_fp8.sh — SELF-HEALING regen + HF upload for PROJECT 3.
#
#   Project 3 : W-FP8 (E4M3, per-tensor) + A-FP16 + KV-FP8 (E4M3, per-tensor)
#               on Llama-3.1-8B-Instruct.  TYPE CONVERSION baseline.
#
# Self-contained: needs ONLY the smoothquant repo clone + base model + HF token.
# No KIVI, no AWQ, no regen_inputs (unlike projects 1/2).
#
# DESIGN GOAL: an overnight run NEVER stalls needing a morning restart.
#   - per-stage .done checkpoints  -> re-run resumes, never restarts from zero
#   - retry+backoff on every stage -> transient/network failures auto-recover
#   - KV dump itself skips already-written samples (resume mid-dump)
#   - uploads retry up to 100x     -> resumable (upload_large_folder)
#   - outer loop until repo verified complete (or MAX_CYCLES)
#
# RECOMMENDED USE:
#   export HF_TOKEN=hf_xxx
#   ./regen_and_upload_fp8.sh preflight     # run WHILE AWAKE — proves it will work
#   setsid nohup ./regen_and_upload_fp8.sh all >> regen_fp8.out 2>&1 &
#   tail -f regen_fp8.out
# =============================================================================
set -uo pipefail     # deliberately NOT -e; we handle errors ourselves.

# ---- config (override via env) ----------------------------------------------
ROOT="${ROOT:-/SSD/JSY}"
SQ="${SQ:-$ROOT/smoothquant}"
PROJ="$SQ/fp8_wfp8a16kvfp8"
CODE="$PROJ/code"
PY="${PY:-python}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/hf_cache}"
REPO="${REPO:-jsyeom/fp8-wfp8a16kvfp8-gs14}"
OUT="$PROJ/compressed_data"
OUTW="$OUT/w_of_wfp8a16kvfp8_llama_31_8b"
OUTKV="$OUT/kv_fp8_of_wfp8a16kvfp8_llama_31_8b"
LIMIT="${LIMIT:-20}"
SEQLEN="${SEQLEN:-4096}"
MAX_CYCLES="${MAX_CYCLES:-300}"
PHASE="${1:-all}"
CKPT="$ROOT/.regen_fp8_ckpt"; mkdir -p "$CKPT"

export HF_HUB_CACHE HF_DATASETS_CACHE="$HF_HUB_CACHE" HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

ts(){ date +%H:%M:%S 2>/dev/null || echo "--:--:--"; }
log(){  echo -e "\n\033[1;36m[$(ts)] $*\033[0m"; }
warn(){ echo -e "\033[1;33m[$(ts)] WARN: $*\033[0m"; }
die(){  echo -e "\n\033[1;31m[$(ts)] FATAL: $*\033[0m"; exit 1; }   # only in preflight (awake)

# retry <max> <name> -- cmd...   (exp backoff capped at 300s)
retry(){ local max=$1 name=$2; shift 2; local n=0 d=20
  while true; do "$@" && return 0; n=$((n+1))
    [ "$n" -ge "$max" ] && { warn "$name: gave up after $max attempts"; return 1; }
    warn "$name: attempt $n failed -> retry in ${d}s"; sleep "$d"; d=$(( d<300 ? d*2 : 300 )); done; }

# stage <name> <max_retries> -- cmd...   (checkpointed + retried)
stage(){ local name=$1 max=$2; shift 2
  [ -f "$CKPT/$name.done" ] && { echo "  skip $name (done)"; return 0; }
  log "STAGE $name"
  if retry "$max" "$name" "$@"; then : > "$CKPT/$name.done"; return 0; else return 1; fi; }

# ============================ PREFLIGHT (run awake) ==========================
preflight(){
  log "PREFLIGHT + SMOKE (run this while awake; ~1-2 min)"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN not set"
  for d in "$SQ" "$CODE"; do [ -e "$d" ] || die "missing: $d"; done
  "$PY" - <<PY || die "deps / CUDA / HF token / lm_eval / fp8 smoke FAILED — fix before sleeping"
import os, sys
sys.path[:0]=["$CODE"]
import torch, transformers, flash_attn, safetensors, huggingface_hub
assert torch.cuda.is_available(), "CUDA unavailable"
assert hasattr(torch, "float8_e4m3fn"), "torch has no float8_e4m3fn"
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); print("HF user:", api.whoami()["name"])
api.model_info("meta-llama/Llama-3.1-8B-Instruct"); print("gated model OK")
from lm_eval.tasks import TaskManager, get_task_dict          # RULER prompt path (the fragile bit)
from fp8_quant import apply_fp8_weight_fakequant, Fp8KVCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F
tok=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
m=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",dtype=torch.float16,
    device_map="cuda:0",attn_implementation="flash_attention_2").eval()
ids=tok("The quick brown fox jumps over the lazy dog. "*20,return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    a=m(ids,use_cache=True,past_key_values=DynamicCache()).logits[:,-1].float()
apply_fp8_weight_fakequant(m)                                 # W-FP8
with torch.no_grad():
    b=m(ids,use_cache=True,past_key_values=Fp8KVCache()).logits[:,-1].float()   # +KV-FP8
cos=F.cosine_similarity(a,b,dim=-1).mean().item()
assert torch.isfinite(b).all() and cos>0.98, f"fp8 smoke bad cos={cos}"
print(f"fp8 smoke OK cos={cos:.5f}")
print("SMOKE PASS")
PY
  free_gb=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'); free_gb=${free_gb:-0}
  [ "$free_gb" -ge 100 ] || die "need >=100GB free under $ROOT (have ${free_gb}GB)"
  echo "  PREFLIGHT PASS (free ${free_gb}GB). Safe to launch 'all' detached."
}

# ============================ PROJECT 3 ======================================
p3(){
  cd "$PROJ" || return 1
  stage p3_weights 5 "$PY" "$CODE/save_fp8_weights.py" --out "$OUTW" || return 1
  [ -f "$OUTW/layer_31.safetensors" ] && [ -f "$OUTW/embeddings.safetensors" ] || {
    warn "P3 weights incomplete"; rm -f "$CKPT/p3_weights.done"; return 1; }
  stage p3_kv 5 "$PY" "$CODE/dump_kv_cache_fp8.py" --limit "$LIMIT" --seqlen "$SEQLEN" --out "$OUTKV" || return 1
  [ -f "$OUTKV/index.json" ] || { warn "P3 KV incomplete"; rm -f "$CKPT/p3_kv.done"; return 1; }
  cp -f "$PROJ/hf_data_card.md" "$OUT/README.md" 2>/dev/null || true
  stage p3_upload 100 "$PY" "$CODE/etc/upload_to_hf.py" --repo-id "$REPO" --path "$OUT" || return 1
  return 0
}

# repo completeness check (0 only if the repo looks complete)
repos_complete(){ "$PY" - "$REPO" <<'PYEOF'
import sys
from huggingface_hub import HfApi
api=HfApi(); r=sys.argv[1]
try: f=api.list_repo_files(r,repo_type="dataset")
except Exception as e: print(f"  {r}: not ready ({e})"); sys.exit(1)
emb=any(x.endswith("embeddings.safetensors") for x in f)
layers=sum(1 for x in f if x.endswith("layer_31.safetensors"))  # last weight layer present
kv=any("/sample_" in x and x.endswith(".safetensors") for x in f)
idx=any(x.endswith("index.json") for x in f)
good = emb and layers>=1 and kv and idx and len(f)>40
print(f"  {r}: files={len(f)} emb={emb} kv={kv} index={idx} -> {'OK' if good else 'INCOMPLETE'}")
sys.exit(0 if good else 1)
PYEOF
}

# ============================ MAIN ===========================================
log "regen_and_upload_fp8.sh  phase=$PHASE  root=$ROOT  repo=$REPO  ckpt=$CKPT"
case "$PHASE" in
  preflight) preflight; exit 0 ;;
  fp8|all) ;;
  *) die "unknown phase '$PHASE' (use: preflight | all)" ;;
esac

[ -f "$CKPT/preflight.done" ] || { preflight && : > "$CKPT/preflight.done"; }

cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  cycle=$((cycle+1)); log "===== CYCLE $cycle / $MAX_CYCLES ====="
  p3 || true
  if repos_complete; then : > "$CKPT/ALL_DONE"; break; fi
  warn "cycle $cycle incomplete; continuing in 60s"; sleep 60
done

if [ -f "$CKPT/ALL_DONE" ]; then
  log "ALL DONE ✅  https://huggingface.co/datasets/$REPO"
  echo "  (remember to ROTATE HF_TOKEN after use)"
else
  warn "reached MAX_CYCLES without full completion — re-run the same command to resume from checkpoints"; exit 1
fi
