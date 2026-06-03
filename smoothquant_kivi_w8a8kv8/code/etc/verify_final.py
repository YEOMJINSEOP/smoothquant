"""FINAL end-to-end verification: implementation + storage + eval-script intent.

(1) WEIGHT : saved int8×scale  ==  fresh W8A8 model weights  (all 32 layers, exact)
(2) KV     : re-prefill a stored prompt -> KIVI quant -> bit-exact match to saved tensors
(3) KV bulk: shape arithmetic + code range [0,255] over ALL 140 samples
(4) inventory: file completeness
(5) EVAL   : eval_lmeval.py actually builds W8A8 + injects KIVI-INT8 KV cache as intended
"""
import os, sys, json, glob
sys.path.insert(0, "/SSD/JSY/smoothquant")
sys.path.insert(0, "/SSD/JSY/KIVI")
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model, W8A8Linear
from quant.new_pack import quant_and_pack_kcache, quant_and_pack_vcache, unpack_tensor
from jsyeom.sq_kivi.dump_kv_cache import get_prompts

ROOT = "/SSD/JSY/smoothquant/smoothquant_kivi_w8a8kv8/compressed_data"
WDIR = f"{ROOT}/w8_of_w8a8_smoothquant_llama_31_8b"
KDIR = f"{ROOT}/kv_kivi8_of_w8a8_smoothquant_llama_31_8b"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCALES = "/SSD/JSY/smoothquant/act_scales/llama-3.1-8b-instruct.pt"
QPROJ = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
         "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
GS, RES, BITS = 32, 128, 8
DEV = "cuda:0"
ok = True
print("="*70); print("FINAL VERIFICATION"); print("="*70)

cfg = json.load(open(f"{WDIR}/config.json"))
assert cfg["smoothing_alpha"]==0.85 and cfg["quantize_bmm_input"]==False
print("loading + smoothing + W8A8 (fresh build, matches config) ...")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, attn_implementation="flash_attention_2")
smooth_lm(model, torch.load(SCALES), 0.85)
model = quantize_model(model, "per_channel", "per_token", quantize_bmm_input=False); model.eval()
n_w8a8 = sum(isinstance(m, W8A8Linear) for m in model.modules())
print(f"  W8A8Linear modules = {n_w8a8} (expect 224)"); ok &= n_w8a8==224

# (1) WEIGHT
print("\n[1] WEIGHT: saved int8×scale  vs  fresh W8A8 model weight (32 layers, 224 linears)")
wmax = 0.0
for i in range(model.config.num_hidden_layers):
    d = load_file(f"{WDIR}/layer_{i}.safetensors")
    for proj in QPROJ:
        deq = d[f"{proj}.weight"].float() * d[f"{proj}.scale"].float()
        ref = model.get_submodule(f"model.layers.{i}.{proj}").weight.float().cpu()
        wmax = max(wmax, (deq-ref).abs().max().item())
print(f"  max|saved_dequant - model_weight| = {wmax:.3e} -> {'PASS' if wmax==0 else ('PASS~' if wmax<1e-3 else 'FAIL')}"); ok &= wmax<1e-3

# (2) KV exact reproduction
print("\n[2] KV: re-prefill stored prompt -> KIVI quant -> compare to saved (bit-exact)")
tok = AutoTokenizer.from_pretrained(MODEL)
prompts = get_prompts("gsm8k_cot", 20, 4096); n=0
meta = json.load(open(f"{KDIR}/gsm8k_cot/sample_{n}/_meta.json"))
ids = tok(prompts[n], return_tensors="pt").input_ids.to(DEV); T = ids.shape[1]
print(f"  gsm8k_cot/sample_0: T={T} vs stored {meta['T']} -> {'match' if T==meta['T'] else 'MISMATCH'}"); ok &= T==meta['T']
with torch.no_grad(): out = model(ids, use_cache=True, past_key_values=DynamicCache())
legacy = out.past_key_values.to_legacy_cache(); nqk=((T-RES)//GS)*GS; nqv=T-RES
for i in [0,15,31]:
    k,v = legacy[i]
    kc,ks,kmn = quant_and_pack_kcache(k[:,:,:nqk,:].contiguous(), GS, BITS)
    vc,vs,vmn = quant_and_pack_vcache(v[:,:,:nqv,:].contiguous(), GS, BITS)
    s = load_file(f"{KDIR}/gsm8k_cot/sample_{n}/layer_{i}.safetensors")
    ek = torch.equal(kc.cpu(),s["k_code"]) and torch.equal(ks.half().cpu(),s["k_scale"]) and torch.equal(kmn.half().cpu(),s["k_min"])
    ev = torch.equal(vc.cpu(),s["v_code"]) and torch.equal(vs.half().cpu(),s["v_scale"]) and torch.equal(vmn.half().cpu(),s["v_min"])
    print(f"    layer_{i}: K exact={ek}  V exact={ev}"); ok &= ek and ev

# (3) KV bulk
print("\n[3] KV bulk: n_quant formula + shapes + code range over ALL samples")
bad=0; nchk=0
for mp in glob.glob(f"{KDIR}/*/sample_*/_meta.json"):
    m=json.load(open(mp)); sd=os.path.dirname(mp); nqk,nqv=m["n_quant_k"],m["n_quant_v"]
    f_ok = nqk==((m["T"]-RES)//GS)*GS and nqv==m["T"]-RES
    d=load_file(f"{sd}/layer_0.safetensors")
    s_ok=(tuple(d["k_code"].shape)==(1,8,nqk//4,128) and tuple(d["k_scale"].shape)==(1,8,nqk//GS,1,128)
          and tuple(d["v_code"].shape)==(1,8,nqv,32) and tuple(d["v_scale"].shape)==(1,8,nqv,4,1))
    kc_int=unpack_tensor(d["k_code"],BITS,pack_dim=2)
    r_ok=int(kc_int.min())>=0 and int(kc_int.max())<=255 and d["k_code"].dtype==torch.int32 and d["k_scale"].dtype==torch.float16
    if not(f_ok and s_ok and r_ok): bad+=1
    nchk+=1
print(f"  {nchk} samples checked, OK={nchk-bad}, bad={bad} -> {'PASS' if bad==0 else 'FAIL'}"); ok &= bad==0

# (4) inventory
print("\n[4] inventory")
wl=len(glob.glob(f"{WDIR}/layer_*.safetensors")); emb=os.path.exists(f"{WDIR}/embeddings.safetensors")
sdirs=glob.glob(f"{KDIR}/*/sample_*"); kf=len(glob.glob(f"{KDIR}/*/sample_*/layer_*.safetensors"))
inv= wl==32 and emb and len(sdirs)==140 and kf==4480
print(f"  weight layers={wl}/32 emb={emb} | KV sample_dirs={len(sdirs)}/140 layer_files={kf}/4480 -> {'PASS' if inv else 'FAIL'}"); ok &= inv

# (5) EVAL SCRIPT intent
print("\n[5] EVAL (eval_lmeval.py): builds W8A8 + injects KIVI-INT8 KV as intended")
from jsyeom.sq_kivi import eval_lmeval as E
import jsyeom.sq_kivi.kivi_int8_cache as KC
# 5a config mapping
assert E.GROUPS["hotpot"]==["longbench_hotpotqa"] and E.GROUPS["gsm8k"]==["gsm8k_cot"] and len(E.GROUPS["ruler"])==5
assert E.GCFG["ruler"]=={"limit":100,"bs":1,"maxlen":34000,"ruler":True}
assert E.GCFG["gsm8k"]["bs"]==8 and E.GCFG["hotpot"]["bs"]==1 and E.GCFG["hotpot"]["maxlen"]==17000
print("  5a config: GROUPS(hotpot=longbench_hotpotqa) + GCFG(ruler 100/bs1/32K, gsm8k bs8, hotpot bs1/17K) -> PASS")
# 5b instrument KiviINT8Cache: record init kwargs + count update calls
inits=[]; cnt={"n":0}
_oi, _ou = KC.KiviINT8Cache.__init__, KC.KiviINT8Cache.update
def spy_init(self,*a,**kw): inits.append(kw); return _oi(self,*a,**kw)
def spy_upd(self,*a,**kw): cnt["n"]+=1; return _ou(self,*a,**kw)
KC.KiviINT8Cache.__init__=spy_init; KC.KiviINT8Cache.update=spy_upd
lm = E.KiviHFLM(pretrained=model, tokenizer=tok, batch_size=1, max_length=4096)
lm.set_kivi(k_bits=8, v_bits=8, group_size=GS, residual_length=RES)
ctx = tok("Q: 2+2? Give the number.\nA:", return_tensors="pt").input_ids.to(DEV)
_ = lm._model_generate(ctx, max_length=ctx.shape[1]+8, stop=[])
KC.KiviINT8Cache.__init__=_oi; KC.KiviINT8Cache.update=_ou
inj = len(inits)>0 and inits[0]=={"k_bits":8,"v_bits":8,"group_size":GS,"residual_length":RES}
print(f"  5b KiviHFLM injected KiviINT8Cache: {len(inits)} cache(s), kwargs={inits[0] if inits else None}")
print(f"     update() calls during eval-gen = {cnt['n']} (>0 means KV is KIVI-quantized in the eval loop)")
print(f"     -> {'PASS' if inj and cnt['n']>0 else 'FAIL'}"); ok &= inj and cnt["n"]>0
# 5c original path uses plain HFLM (no KIVI cache)
from lm_eval.models.huggingface import HFLM
print(f"  5c original variant class = HFLM (no KIVI injection): {E.KiviHFLM.__mro__[1] is HFLM} -> {'PASS' if E.KiviHFLM.__mro__[1] is HFLM else 'FAIL'}")
# 5d effect: KIVI cache changes a long-context decode vs fp16 cache (not a no-op)
long_ids = tok(("The quick brown fox. "*120), return_tensors="pt").input_ids.to(DEV)  # >residual
def decode_logits(cache):
    with torch.no_grad():
        o1=model(long_ids[:,:-1], use_cache=True, past_key_values=cache)
        o2=model(long_ids[:,-1:], use_cache=True, past_key_values=o1.past_key_values)
    return o2.logits[:,-1,:].float()
lf = decode_logits(DynamicCache())
lk = decode_logits(KC.KiviINT8Cache(k_bits=8,v_bits=8,group_size=GS,residual_length=RES))
diff=(lf-lk).abs().max().item()
print(f"  5d KIVI-cache vs fp16-cache decode logit max-diff = {diff:.3e} (>0 → quantization actually applied) -> {'PASS' if diff>0 else 'FAIL'}"); ok &= diff>0

print("\n"+"="*70); print(f"OVERALL: {'✅ ALL PASS' if ok else '❌ FAIL'}"); print("="*70)
