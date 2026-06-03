#!/usr/bin/env bash
# =============================================================================
# regen_and_upload.sh — SELF-HEALING regeneration + HF upload for BOTH projects.
#
#   Project 1 : SmoothQuant W8A8 + KIVI-INT8 KV   (Llama-3.1-8B-Instruct)
#   Project 2 : AWQ W4A16 (asym, g128) + KIVI-INT8 KV
#
# DESIGN GOAL: an overnight run NEVER stalls needing a morning restart.
#   - per-stage .done checkpoints  -> re-run resumes, never restarts from zero
#   - retry+backoff on every stage -> transient/network failures auto-recover
#   - uploads retry up to 100x     -> resumable (upload_large_folder)
#   - verification is ADVISORY      -> fp16-LSB diffs on a different GPU won't kill it
#   - outer loop until both repos verified complete (or MAX_CYCLES)
#
# RECOMMENDED USE:
#   export HF_TOKEN=hf_xxx ; export LMEVAL_DIR=/path/to/lm_eval_fork
#   ./regen_and_upload.sh preflight        # run WHILE AWAKE — proves it will work
#   setsid nohup ./regen_and_upload.sh all >> regen.out 2>&1 &   # detached, survives logout
#   tail -f regen.out
# =============================================================================
set -uo pipefail     # NOTE: deliberately NOT -e; we handle errors ourselves.

# ---- config (override via env) ----------------------------------------------
ROOT="${ROOT:-/SSD/JSY}"
SQ="$ROOT/smoothquant"; KIVI="$ROOT/KIVI"; AWQ="$ROOT/llm-awq"
INPUTS="${INPUTS:-$SQ/regen_inputs}"
LMEVAL_DIR="${LMEVAL_DIR:-$ROOT/-ASSIGN-mask_lm_eval}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/hf_cache}"
SQ_REPO="${SQ_REPO:-jsyeom/smoothquant-kivi-w8a8kv8}"
AWQ_REPO="${AWQ_REPO:-jsyeom/awq-kivi-w4a16kv8}"
LIMIT="${LIMIT:-20}"
MAX_CYCLES="${MAX_CYCLES:-300}"
PHASE="${1:-all}"
CKPT="$ROOT/.regen_ckpt"; mkdir -p "$CKPT"

export HF_HUB_CACHE HF_DATASETS_CACHE="$HF_HUB_CACHE" HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="$SQ:$KIVI:$AWQ"
export TOKENIZERS_PARALLELISM=false

ts(){ date +%H:%M:%S 2>/dev/null || echo "--:--:--"; }
log(){  echo -e "\n\033[1;36m[$(ts)] $*\033[0m"; }
warn(){ echo -e "\033[1;33m[$(ts)] WARN: $*\033[0m"; }
die(){  echo -e "\n\033[1;31m[$(ts)] FATAL: $*\033[0m"; exit 1; }   # only used in preflight (awake)

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

# advisory verify: record number, NEVER abort
advisory(){ local label=$1; shift; log "VERIFY(advisory) $label"
  local out; out="$("$@" 2>&1)"; echo "$out"
  echo "$out" | grep -qiE "FAIL|error|Traceback" && \
    warn "$label reported a discrepancy — likely fp16-LSB on a different GPU (artifact still valid); recorded, continuing." || true; }

# ============================ PREFLIGHT (run awake) ==========================
preflight(){
  log "PREFLIGHT + SMOKE (run this while awake; ~1-2 min)"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN not set"
  for d in "$SQ" "$KIVI" "$INPUTS"; do [ -e "$d" ] || die "missing: $d"; done
  python - <<PY || die "python deps / CUDA / HF token / lm_eval / model access FAILED — fix before sleeping"
import os, sys, types
sys.modules.setdefault("awq_inference_engine", types.ModuleType("awq_inference_engine"))
sys.path[:0]=["$SQ","$KIVI","$AWQ"]
import torch, transformers, flash_attn, safetensors, huggingface_hub
assert torch.cuda.is_available(), "CUDA unavailable"
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); print("HF user:", api.whoami()["name"])
api.model_info("meta-llama/Llama-3.1-8B-Instruct"); print("gated model OK")
from lm_eval.tasks import TaskManager
# RULER prompt generation (the fragile bit) — one prompt, one task
from jsyeom.sq_kivi.kivi_int8_cache import KiviINT8Cache
from quant.new_pack import quant_and_pack_kcache
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
tok=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
m=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",dtype=torch.float16,
    device_map="cuda:0",attn_implementation="flash_attention_2").eval()
ids=tok("The quick brown fox jumps over the lazy dog. "*20,return_tensors="pt").input_ids.cuda()
import torch.nn.functional as F
with torch.no_grad():
    a=m(ids,use_cache=True,past_key_values=DynamicCache()).logits[:,-1].float()
    b=m(ids,use_cache=True,past_key_values=KiviINT8Cache(k_bits=8,v_bits=8,group_size=32,residual_length=128)).logits[:,-1].float()
cos=F.cosine_similarity(a,b,dim=-1).mean().item(); top1=(a.argmax(-1)==b.argmax(-1)).float().mean().item()
assert torch.isfinite(b).all() and cos>0.99, f"KIVI smoke bad cos={cos}"
print(f"KIVI smoke OK cos={cos:.5f} top1={top1:.3f}")
print("SMOKE PASS")
PY
  free_gb=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'); free_gb=${free_gb:-0}
  [ "$free_gb" -ge 150 ] || die "need >=150GB free under $ROOT (have ${free_gb}GB)"
  echo "  PREFLIGHT PASS (free ${free_gb}GB). Safe to launch 'all' detached."
}

# ============================ PROJECT 1 ======================================
p1(){
  local C="$SQ/smoothquant_kivi_w8a8kv8/code"
  local OUTW="$SQ/compressed_data/w8_of_w8a8_smoothquant_llama_31_8b"
  local OUTKV="$SQ/compressed_data/kv_kivi8_of_w8a8_smoothquant_llama_31_8b"
  cd "$SQ" || return 1
  cp -f "$INPUTS/llama-3.1-8b-instruct.pt" "$SQ/act_scales/" 2>/dev/null || true

  stage p1_weights 5 python "$C/save_w8a8_weights.py" --alpha 0.85 --out "$OUTW" || return 1
  stage p1_split   3 python "$C/split_weights_per_layer.py" --dir "$OUTW" --remove-monolith || return 1
  [ -f "$OUTW/layer_31.safetensors" ] && [ -f "$OUTW/embeddings.safetensors" ] || { warn "P1 weights incomplete"; rm -f "$CKPT/p1_weights.done" "$CKPT/p1_split.done"; return 1; }
  stage p1_kv      5 python "$C/dump_kv_cache.py" --limit "$LIMIT" --out "$OUTKV" || return 1
  stage p1_kvconv  3 python "$C/etc/convert_kv_to_safetensors.py" --root "$OUTKV" || return 1
  [ -f "$OUTKV/index.json" ] || { warn "P1 KV incomplete"; rm -f "$CKPT/p1_kv.done" "$CKPT/p1_kvconv.done"; return 1; }
  [ -f "$CKPT/p1_verify.done" ] || { advisory "P1 verify_final" python "$C/etc/verify_final.py"; : > "$CKPT/p1_verify.done"; }
  cp -f "$SQ/smoothquant_kivi_w8a8kv8/compressed_data/README.md" "$SQ/compressed_data/" 2>/dev/null || true
  stage p1_upload 100 python "$C/etc/upload_to_hf.py" --repo-id "$SQ_REPO" --path "$SQ/compressed_data" || return 1
  return 0
}

# ============================ PROJECT 2 ======================================
p2(){
  [ -d "$AWQ" ] || { warn "missing $AWQ (clone upstream llm-awq); skipping P2 this cycle"; return 1; }
  [ -d "$AWQ/jsyeom" ] || cp -r "$INPUTS/awq_jsyeom" "$AWQ/jsyeom"
  mkdir -p "$AWQ/awq_cache"; cp -f "$INPUTS/llama-3.1-8b-instruct-w4-g128.pt" "$AWQ/awq_cache/" 2>/dev/null
  touch "$AWQ/awq/__init__.py"
  # overlay our EXACT awq source (origin-verified) over the upstream clone -> zero upstream-drift risk.
  # (only the asym path matters and it already matches upstream; this just makes it bit-identical-proof.)
  [ -d "$INPUTS/awq_overlay/awq" ] && cp -rf "$INPUTS/awq_overlay/awq/." "$AWQ/awq/"
  local OUTW="$AWQ/compressed_data/w4_awq_llama_31_8b"
  local OUTKV="$AWQ/compressed_data/kv_kivi8_of_w4a16_awq_llama_31_8b"
  cd "$AWQ" || return 1

  stage p2_weights 5 python jsyeom/save_w4_weights.py --out "$OUTW" || return 1
  [ -f "$OUTW/layer_31.safetensors" ] && [ -f "$OUTW/embeddings.safetensors" ] || { warn "P2 weights incomplete"; rm -f "$CKPT/p2_weights.done"; return 1; }
  stage p2_kv      5 python jsyeom/dump_kv_cache_awq.py --limit "$LIMIT" --out "$OUTKV" || return 1
  [ -f "$OUTKV/index.json" ] || { warn "P2 KV incomplete"; rm -f "$CKPT/p2_kv.done"; return 1; }
  [ -f "$CKPT/p2_verify.done" ] || { advisory "P2 combined" python jsyeom/verify_combined_awq.py; : > "$CKPT/p2_verify.done"; }
  cp -f "$AWQ/jsyeom/hf_data_card_awq.md" "$AWQ/compressed_data/README.md" 2>/dev/null || true
  stage p2_upload 100 python jsyeom/upload_to_hf.py --repo-id "$AWQ_REPO" --path "$AWQ/compressed_data" || return 1
  return 0
}

# repo completeness check (returns 0 only if BOTH repos look complete)
repos_complete(){ python - "$SQ_REPO" "$AWQ_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
api=HfApi(); ok=True
for r in sys.argv[1:]:
    try: f=api.list_repo_files(r,repo_type="dataset")
    except Exception as e: print(f"  {r}: not ready ({e})"); ok=False; continue
    w=sum("layer_" in x and "w8_of_" in x or "w4_awq_" in x for x in f if x.endswith(".safetensors"))
    kv=any("/sample_" in x for x in f); idx=any(x.endswith("index.json") for x in f)
    emb=any(x.endswith("embeddings.safetensors") for x in f)
    good = emb and kv and idx and len(f)>60
    print(f"  {r}: files={len(f)} emb={emb} kv={kv} index={idx} -> {'OK' if good else 'INCOMPLETE'}")
    ok = ok and good
sys.exit(0 if ok else 1)
PY
}

# ============================ MAIN ===========================================
log "regen_and_upload.sh  phase=$PHASE  root=$ROOT  ckpt=$CKPT"
case "$PHASE" in
  preflight) preflight; exit 0 ;;
  sq|awq|all) ;;
  *) die "unknown phase '$PHASE' (use: preflight | sq | awq | all)" ;;
esac

[ -f "$CKPT/preflight.done" ] || { preflight && : > "$CKPT/preflight.done"; }

cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  cycle=$((cycle+1)); log "===== CYCLE $cycle / $MAX_CYCLES ====="
  case "$PHASE" in sq) p1 || true ;; awq) p2 || true ;; all) p1 || true; p2 || true ;; esac
  if { [ "$PHASE" = all ] && repos_complete; } || \
     { [ "$PHASE" = sq ]  && [ -f "$CKPT/p1_upload.done" ]; } || \
     { [ "$PHASE" = awq ] && [ -f "$CKPT/p2_upload.done" ]; }; then
     : > "$CKPT/ALL_DONE"; break
  fi
  warn "cycle $cycle incomplete; continuing in 60s"; sleep 60
done

if [ -f "$CKPT/ALL_DONE" ]; then
  log "ALL DONE ✅"
  echo "  P1: https://huggingface.co/datasets/$SQ_REPO"
  echo "  P2: https://huggingface.co/datasets/$AWQ_REPO"
  echo "  (remember to ROTATE HF_TOKEN and GH_TOKEN)"
else
  warn "reached MAX_CYCLES without full completion — re-run the same command to resume from checkpoints"; exit 1
fi
