"""Project 2 Task 2 — dump KIVI-INT8 KV cache of the AWQ W4A16 model.

Same as project-1 dump_kv_cache.py, but the model is built with AWQ
(apply_awq + pseudo_quantize W4) instead of SmoothQuant W8A8. The KV quantization
(KIVI INT8, key per-channel / value per-token, group32 / residual128) is identical.

Layout (per-layer): <out>/<task>/sample_<n>/layer_<i>.safetensors
   {k_code,k_scale,k_min,v_code,v_scale,v_min}  + <sample>/_meta.json
Only the packed-INT8 portion (residual fp16 excluded); snapshot right after prefill.
"""
import os, sys, json, types, argparse, time
sys.modules.setdefault("awq_inference_engine", types.ModuleType("awq_inference_engine"))
sys.path.insert(0, "/SSD/JSY/llm-awq")
sys.path.insert(0, "/SSD/JSY/KIVI")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from safetensors.torch import save_file
from awq.quantize.pre_quant import apply_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight
from quant.new_pack import quant_and_pack_kcache, quant_and_pack_vcache

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
AWQ = "/SSD/JSY/llm-awq/awq_cache/llama-3.1-8b-instruct-w4-g128.pt"
RULER = ["niah_multikey_1", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad"]
OTHER = ["gsm8k_cot", "longbench_hotpotqa"]
DEV = "cuda:0"


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
        if inst.doc_id in seen:
            continue
        seen.add(inst.doc_id); prompts.append(inst.args[0])
        if len(prompts) >= limit:
            break
    return prompts


@torch.no_grad()
def dump_sample(model, tok, prompt, gs, res, kbits, vbits, sdir):
    os.makedirs(sdir, exist_ok=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model(ids, use_cache=True, past_key_values=DynamicCache())
    legacy = out.past_key_values.to_legacy_cache()
    T = ids.shape[1]
    nqk = ((T - res) // gs) * gs
    nqv = max(T - res, 0)
    nbytes = 0
    for i, (k, v) in enumerate(legacy):
        entry = {}
        if nqk > 0:
            kc, ks, kmn = quant_and_pack_kcache(k[:, :, :nqk, :].contiguous(), gs, kbits)
            entry.update(k_code=kc.cpu(), k_scale=ks.half().cpu(), k_min=kmn.half().cpu())
        if nqv > 0:
            vc, vs, vmn = quant_and_pack_vcache(v[:, :, :nqv, :].contiguous(), gs, vbits)
            entry.update(v_code=vc.cpu(), v_scale=vs.half().cpu(), v_min=vmn.half().cpu())
        lp = os.path.join(sdir, f"layer_{i}.safetensors")
        save_file(entry, lp); nbytes += os.path.getsize(lp)
    json.dump({"T": T, "n_quant_k": int(nqk), "n_quant_v": int(nqv), "group_size": gs,
               "residual_length": res, "k_bits": kbits, "v_bits": vbits,
               "note": "AWQ W4A16 model; packed INT8 only; fp16 residual excluded; post-prefill"},
              open(os.path.join(sdir, "_meta.json"), "w"), indent=2)
    del out, legacy
    return nbytes, T


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--residual-length", type=int, default=128)
    ap.add_argument("--k-bits", type=int, default=8)
    ap.add_argument("--v-bits", type=int, default=8)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--out", default="/SSD/JSY/llm-awq/compressed_data/kv_kivi8_of_w4a16_awq_llama_31_8b")
    ap.add_argument("--tasks", nargs="+", default=None)
    args = ap.parse_args()
    todo = args.tasks if args.tasks else (RULER + OTHER)

    tok = AutoTokenizer.from_pretrained(MODEL)
    print("load + apply_awq + pseudo_quantize W4 (W4A16 model) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV,
                                                 attn_implementation="flash_attention_2")
    apply_awq(model, torch.load(AWQ, map_location="cpu"))
    pseudo_quantize_model_weight(model, w_bit=4, q_config={"zero_point": True, "q_group_size": 128})
    model.to(DEV).eval()

    total = 0
    for task in todo:
        print(f"\n=== {task} (limit {args.limit}) ===", flush=True)
        prompts = get_prompts(task, args.limit, args.seqlen)
        for n, p in enumerate(prompts):
            t0 = time.time()
            sdir = os.path.join(args.out, task, f"sample_{n}")
            sz, T = dump_sample(model, tok, p, args.group_size, args.residual_length,
                                args.k_bits, args.v_bits, sdir)
            total += sz; torch.cuda.empty_cache()
            print(f"  sample_{n}: T={T} -> {sz/1e6:.1f}MB {time.time()-t0:.1f}s", flush=True)

    json.dump({"model": MODEL, "weight_method": "AWQ asym W4 g128 (W4A16)", "awq_search": os.path.basename(AWQ),
               "kv": {"k_bits": args.k_bits, "v_bits": args.v_bits, "group_size": args.group_size,
                      "residual_length": args.residual_length, "key": "per-channel", "value": "per-token"},
               "tasks": RULER + OTHER, "limit": args.limit, "ruler_seqlen": args.seqlen,
               "format": "<task>/sample_<n>/layer_<i>.safetensors -> {k_code,k_scale,k_min,v_code,v_scale,v_min} (+_meta.json)"},
              open(os.path.join(args.out, "index.json"), "w"), indent=2)
    print(f"\nDONE. total {total/1e9:.2f} GB -> {args.out}/")


if __name__ == "__main__":
    main()
