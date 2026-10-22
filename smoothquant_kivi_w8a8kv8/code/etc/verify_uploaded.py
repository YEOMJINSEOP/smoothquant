"""Rigorous SAMPLED verification that the two uploaded HF datasets contain exactly
what we intended:
  - jsyeom/smoothquant-kivi-w8a8kv8-gs14 : SmoothQuant W8A8 weights (int8) + KIVI-INT8 KV
  - jsyeom/awq-kivi-w4a16kv8-gs14        : AWQ W4A16 weights (int4) + KIVI-INT8 KV

Two layers of proof (sampled):
  LAYER 1 — upload integrity (HF == local): for sampled files, compare HF LFS sha256
            (or git blob sha1 for small files) to the local compressed_data. No big download.
  LAYER 2 — semantic correctness (local == intended): REBUILD the model from scratch
            (SmoothQuant: smooth_lm+quantize ; AWQ: apply_awq+pseudo_quantize) and the KV
            (prefill + KIVI pack) exactly as the dump scripts did, then bit-exact compare
            to the stored tensors.
  (LAYER1 & LAYER2)  =>  HF == intended.

Run on the ORIGIN server (same GPU/versions) so regeneration is bit-exact.
  python code/etc/verify_uploaded.py --stage all
"""
import os, sys, json, hashlib, argparse, types
sys.path.insert(0, "/SSD/JSY/smoothquant")
sys.path.insert(0, "/SSD/JSY/KIVI")
sys.path.insert(0, "/SSD/JSY/llm-awq")
sys.modules.setdefault("awq_inference_engine", types.ModuleType("awq_inference_engine"))
import torch
from safetensors.torch import load_file
from safetensors import safe_open

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "cuda:0"
SQ_LOCAL = "/SSD/JSY/smoothquant/smoothquant_kivi_w8a8kv8/compressed_data"
AWQ_LOCAL = "/SSD/JSY/llm-awq/compressed_data"
SQ_REPO = "jsyeom/smoothquant-kivi-w8a8kv8-gs14"
AWQ_REPO = "jsyeom/awq-kivi-w4a16kv8-gs14"
ACT_SCALES = "/SSD/JSY/smoothquant/act_scales/llama-3.1-8b-instruct.pt"
AWQ_CACHE = "/SSD/JSY/llm-awq/awq_cache/llama-3.1-8b-instruct-w4-g128.pt"
GS, RES, KB, VB, SEQLEN = 32, 128, 8, 8, 4096
QPROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
LAYERS_SAMPLE = [0, 15, 31]
KV_SAMPLE = [("gsm8k_cot", 0)]                              # deterministic prompt -> regen bit-compare
KV_INTERNAL = [("niah_multikey_1", 0), ("ruler_vt", 0)]    # RULER synthetic (non-reproducible) -> internal-consistency
OK = lambda b: "\033[1;32mPASS\033[0m" if b else "\033[1;31mFAIL\033[0m"


# ----------------------------- LAYER 1: integrity ----------------------------
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def git_blob_sha1(p):
    h = hashlib.sha1(); n = os.path.getsize(p)
    h.update(f"blob {n}\0".encode())
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def integrity(repo, local, sample_rel):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    print(f"\n=== LAYER 1 integrity (HF==local): {repo} ===", flush=True)
    info = {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, expand=True)}
    allok = True
    for rel in sample_rel:
        lp = os.path.join(local, rel)
        if rel not in info or not os.path.exists(lp):
            print(f"  {OK(False)} {rel}  (HF:{rel in info} local:{os.path.exists(lp)})"); allok = False; continue
        rf = info[rel]
        lfs = getattr(rf, "lfs", None)
        if lfs and getattr(lfs, "sha256", None):
            match = (lfs.sha256 == sha256_file(lp)); how = "lfs-sha256"
        else:
            match = (rf.blob_id == git_blob_sha1(lp)); how = "git-sha1"
        sz_ok = (rf.size == os.path.getsize(lp))
        print(f"  {OK(match and sz_ok)} {rel:60s} {how} size={rf.size}")
        allok = allok and match and sz_ok
    print(f"  -> integrity {OK(allok)}")
    return allok


# ----------------------------- LAYER 2: weights ------------------------------
@torch.no_grad()
def verify_sq_weights():
    from transformers import AutoModelForCausalLM
    from smoothquant.smooth import smooth_lm
    from smoothquant.fake_quant import quantize_model
    print("\n=== LAYER 2 semantic — SmoothQuant W8A8 weights ===", flush=True)
    print("  rebuild: load fp16 -> smooth_lm(0.85) -> quantize_model(W8A8) ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cpu")
    smooth_lm(m, torch.load(ACT_SCALES), 0.85)
    m = quantize_model(m, "per_channel", "per_token", quantize_bmm_input=False)
    allok = True
    for i in LAYERS_SAMPLE:
        stored = load_file(os.path.join(SQ_LOCAL, "w8_of_w8a8_smoothquant_llama_31_8b", f"layer_{i}.safetensors"))
        for proj in QPROJ:
            w_fq = m.get_submodule(f"model.layers.{i}.{proj}").weight.data.float()
            scale = w_fq.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
            q = (w_fq / scale).round().clamp(-127, 127).to(torch.int8)
            sq_i8, sq_sc = stored[f"{proj}.weight"], stored[f"{proj}.scale"]
            di8 = (sq_i8 != q).sum().item()
            dsc = (sq_sc.float() - scale.half().float()).abs().max().item()
            # dequant must reproduce the fake-quant weight the model actually uses
            deq_err = (sq_i8.float() * sq_sc.float() - w_fq).abs().max().item()
            good = (di8 == 0 and dsc == 0 and deq_err < 1e-2)
            if not good or proj == QPROJ[0]:
                print(f"  {OK(good)} L{i:2d} {proj:18s} int8_mismatch={di8} scale_err={dsc:.0e} dequant_vs_Wfq={deq_err:.2e}")
            allok = allok and good
    del m; print(f"  -> SQ weights {OK(allok)}")
    return allok

@torch.no_grad()
def verify_awq_weights():
    from transformers import AutoModelForCausalLM
    from awq.quantize.pre_quant import apply_awq
    from jsyeom.save_w4_weights import extract_w4, pack_nibbles, dequant_from_packed
    print("\n=== LAYER 2 semantic — AWQ W4A16 weights ===", flush=True)
    print("  rebuild: load fp16 -> apply_awq(official search) ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cpu")
    apply_awq(m, torch.load(AWQ_CACHE, map_location="cpu"))
    allok = True
    for i in LAYERS_SAMPLE:
        stored = load_file(os.path.join(AWQ_LOCAL, "w4_awq_llama_31_8b", f"layer_{i}.safetensors"))
        for proj in QPROJ:
            w = m.get_submodule(f"model.layers.{i}.{proj}").weight.data
            code, scale, zero, w_dq = extract_w4(w)
            packed = pack_nibbles(code)
            sq, ss, sz = stored[f"{proj}.qweight"], stored[f"{proj}.scale"], stored[f"{proj}.zero"]
            dpk = (sq != packed).sum().item()
            dsc = (ss.float() - scale.float()).abs().max().item()
            dzr = (sz.int() != zero.int()).sum().item()
            deq_err = (dequant_from_packed(sq, ss, sz).half() - w_dq).abs().max().item()
            good = (dpk == 0 and dsc == 0 and dzr == 0 and deq_err < 1e-2)
            if not good or proj == QPROJ[0]:
                print(f"  {OK(good)} L{i:2d} {proj:18s} qweight_mismatch={dpk} scale_err={dsc:.0e} zero_mismatch={dzr} dequant_vs_Wdq={deq_err:.2e}")
            allok = allok and good
    del m; print(f"  -> AWQ weights {OK(allok)}")
    return allok


# ----------------------------- LAYER 2: KV cache -----------------------------
def get_prompts(task_name, limit, seqlen):
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager(metadata={"max_seq_lengths": [seqlen], "tokenizer": MODEL, "pretrained": MODEL})
    td = get_task_dict([task_name], tm)
    def flat(d):
        o = {}
        for k, v in d.items():
            o.update(flat(v)) if isinstance(v, dict) else o.update({k: v})
        return o
    t = list(flat(td).values())[0]
    t.build_all_requests(limit=limit, rank=0, world_size=1)
    seen, prompts = set(), []
    for inst in t.instances:
        if inst.doc_id in seen: continue
        seen.add(inst.doc_id); prompts.append(inst.args[0])
        if len(prompts) >= limit: break
    return prompts

@torch.no_grad()
def verify_kv(which, build_fn, local_root):
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from quant.new_pack import quant_and_pack_kcache, quant_and_pack_vcache
    print(f"\n=== LAYER 2 semantic — {which} KIVI-INT8 KV ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = build_fn(); model.eval()
    allok = True
    for task, n in KV_SAMPLE:
        prompts = get_prompts(task, n + 1, SEQLEN)
        ids = tok(prompts[n], return_tensors="pt").input_ids.to(DEV)
        out = model(ids, use_cache=True, past_key_values=DynamicCache())
        legacy = out.past_key_values.to_legacy_cache()
        T = ids.shape[1]; nqk = ((T - RES) // GS) * GS; nqv = max(T - RES, 0)
        sdir = os.path.join(local_root, task, f"sample_{n}")
        meta = json.load(open(os.path.join(sdir, "_meta.json")))
        tok_ok = (meta["T"] == T and meta["n_quant_k"] == nqk and meta["n_quant_v"] == nqv)
        print(f"  [{task} sample_{n}] T={T} (stored {meta['T']}) nqk={nqk} nqv={nqv} meta {OK(tok_ok)}", flush=True)
        bad = 0
        for i, (k, v) in enumerate(legacy):
            st = load_file(os.path.join(sdir, f"layer_{i}.safetensors"))
            kc, ks, kmn = quant_and_pack_kcache(k[:, :, :nqk, :].contiguous(), GS, KB)
            vc, vs, vmn = quant_and_pack_vcache(v[:, :, :nqv, :].contiguous(), GS, VB)
            checks = [
                (st["k_code"].to(DEV) != kc).sum().item(),
                (st["v_code"].to(DEV) != vc).sum().item(),
                (st["k_scale"].float().to(DEV) - ks.half().float()).abs().max().item(),
                (st["v_scale"].float().to(DEV) - vs.half().float()).abs().max().item(),
                (st["k_min"].float().to(DEV) - kmn.half().float()).abs().max().item(),
                (st["v_min"].float().to(DEV) - vmn.half().float()).abs().max().item(),
            ]
            if not (checks[0] == 0 and checks[1] == 0 and max(checks[2:]) == 0): bad += 1
        good = (tok_ok and bad == 0)
        print(f"    32 layers bit-exact (code+scale+min): {OK(good)}  (mismatched layers={bad})")
        allok = allok and good
    del model; torch.cuda.empty_cache()
    print(f"  -> {which} KV {OK(allok)}")
    return allok

@torch.no_grad()
def verify_kv_internal(which, local_root):
    """RULER prompts are synthetic+random (not reproducible run-to-run), so we cannot
    bit-compare against a regen. Instead prove the STORED files are a well-formed, valid
    KIVI-INT8 KV encoding consistent with their metadata (the pipeline that produced them
    is the SAME code already proven bit-exact on the deterministic gsm8k_cot sample):
      - codes in [0, 2^bits-1]
      - unpack shapes match _meta (k: nqk tokens / v: nqv tokens, head_dim D=128)
      - scale/min group shapes correct; scale>=0, all dequant values finite."""
    from quant.new_pack import unpack_and_dequant_kcache, unpack_and_dequant_vcache, unpack_tensor
    D, maxint = 128, 2 ** KB - 1
    print(f"\n=== LAYER 2 well-formedness — {which} RULER KV (stored, prompt irreproducible) ===", flush=True)
    allok = True
    for task, n in KV_INTERNAL:
        sdir = os.path.join(local_root, task, f"sample_{n}")
        meta = json.load(open(os.path.join(sdir, "_meta.json")))
        nqk, nqv = meta["n_quant_k"], meta["n_quant_v"]
        rng = shp = fin = sc = True
        for i in range(32):
            st = load_file(os.path.join(sdir, f"layer_{i}.safetensors"))
            kc, ks, kmn = st["k_code"].to(DEV), st["k_scale"].to(DEV), st["k_min"].to(DEV)
            vc, vs, vmn = st["v_code"].to(DEV), st["v_scale"].to(DEV), st["v_min"].to(DEV)
            ku = unpack_tensor(kc, KB, pack_dim=2); vu = unpack_tensor(vc, VB, pack_dim=3)
            if ku.min() < 0 or ku.max() > maxint or vu.min() < 0 or vu.max() > maxint: rng = False
            if ku.shape[2:] != (nqk, D) or vu.shape[2:] != (nqv, D): shp = False
            if ks.shape != (1, 8, nqk // GS, 1, D) or vs.shape != (1, 8, nqv, D // GS, 1): shp = False
            if (ks < 0).any() or not (torch.isfinite(ks).all() and torch.isfinite(kmn).all()
                                      and torch.isfinite(vs).all() and torch.isfinite(vmn).all()): sc = False
            Kh = unpack_and_dequant_kcache(kc, ks, kmn, GS, KB)
            Vh = unpack_and_dequant_vcache(vc, vs, vmn, GS, VB)
            if not (torch.isfinite(Kh).all() and torch.isfinite(Vh).all()): fin = False
        good = rng and shp and fin and sc
        print(f"  [{task} sample_{n}] T={meta['T']} nqk={nqk} nqv={nqv} | codes∈[0,{maxint}]={rng} "
              f"shapes==meta={shp} scale≥0&finite={sc} dequant_finite={fin} -> {OK(good)}")
        allok = allok and good
    print(f"  -> {which} RULER well-formedness {OK(allok)}")
    return allok


def build_sq():
    from transformers import AutoModelForCausalLM
    from smoothquant.smooth import smooth_lm
    from smoothquant.fake_quant import quantize_model
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV,
                                             attn_implementation="flash_attention_2")
    smooth_lm(m, torch.load(ACT_SCALES), 0.85)
    return quantize_model(m, "per_channel", "per_token", quantize_bmm_input=False)

def build_awq():
    from transformers import AutoModelForCausalLM
    from awq.quantize.pre_quant import apply_awq
    from awq.quantize.quantizer import pseudo_quantize_model_weight
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV,
                                             attn_implementation="flash_attention_2")
    apply_awq(m, torch.load(AWQ_CACHE, map_location="cpu"))
    pseudo_quantize_model_weight(m, w_bit=4, q_config={"zero_point": True, "q_group_size": 128})
    return m.to(DEV)


SQ_FILES = ["README.md", "w8_of_w8a8_smoothquant_llama_31_8b/config.json",
            "w8_of_w8a8_smoothquant_llama_31_8b/layer_0.safetensors",
            "w8_of_w8a8_smoothquant_llama_31_8b/layer_31.safetensors",
            "w8_of_w8a8_smoothquant_llama_31_8b/embeddings.safetensors",
            "kv_kivi8_of_w8a8_smoothquant_llama_31_8b/index.json",
            "kv_kivi8_of_w8a8_smoothquant_llama_31_8b/gsm8k_cot/sample_0/_meta.json",
            "kv_kivi8_of_w8a8_smoothquant_llama_31_8b/gsm8k_cot/sample_0/layer_0.safetensors",
            "kv_kivi8_of_w8a8_smoothquant_llama_31_8b/niah_multikey_1/sample_0/layer_15.safetensors"]
AWQ_FILES = ["README.md", "w4_awq_llama_31_8b/config.json",
             "w4_awq_llama_31_8b/layer_0.safetensors", "w4_awq_llama_31_8b/layer_31.safetensors",
             "w4_awq_llama_31_8b/embeddings.safetensors",
             "kv_kivi8_of_w4a16_awq_llama_31_8b/index.json",
             "kv_kivi8_of_w4a16_awq_llama_31_8b/gsm8k_cot/sample_0/_meta.json",
             "kv_kivi8_of_w4a16_awq_llama_31_8b/gsm8k_cot/sample_0/layer_0.safetensors",
             "kv_kivi8_of_w4a16_awq_llama_31_8b/niah_multikey_1/sample_0/layer_15.safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "integrity", "sq_w", "awq_w", "sq_kv", "awq_kv", "weights", "kv", "kv_internal"])
    args = ap.parse_args()
    R = {}
    s = args.stage
    if s in ("all", "integrity"):
        R["integrity_sq"] = integrity(SQ_REPO, SQ_LOCAL, SQ_FILES)
        R["integrity_awq"] = integrity(AWQ_REPO, AWQ_LOCAL, AWQ_FILES)
    if s in ("all", "weights", "sq_w"):  R["sq_weights"] = verify_sq_weights()
    if s in ("all", "weights", "awq_w"): R["awq_weights"] = verify_awq_weights()
    if s in ("all", "kv", "sq_kv"):  R["sq_kv"] = verify_kv("SmoothQuant W8A8", build_sq, os.path.join(SQ_LOCAL, "kv_kivi8_of_w8a8_smoothquant_llama_31_8b"))
    if s in ("all", "kv", "awq_kv"): R["awq_kv"] = verify_kv("AWQ W4A16", build_awq, os.path.join(AWQ_LOCAL, "kv_kivi8_of_w4a16_awq_llama_31_8b"))
    if s in ("all", "kv", "kv_internal"):
        R["sq_kv_ruler"] = verify_kv_internal("SmoothQuant W8A8", os.path.join(SQ_LOCAL, "kv_kivi8_of_w8a8_smoothquant_llama_31_8b"))
        R["awq_kv_ruler"] = verify_kv_internal("AWQ W4A16", os.path.join(AWQ_LOCAL, "kv_kivi8_of_w4a16_awq_llama_31_8b"))
    print("\n================= SUMMARY =================")
    for k, v in R.items(): print(f"  {OK(v)}  {k}")
    print("===========================================")
    print("ALL PASS" if all(R.values()) else "SOME FAILED")


if __name__ == "__main__":
    main()
