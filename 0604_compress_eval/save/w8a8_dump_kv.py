"""Task 2 — dump KIVI-INT8 KV cache (packed int8 + scale + min) of the W8A8 model.

For each sample: build the lm-eval prompt, prefill with the W8A8 model, take the fp16
KV (post-RoPE) from the cache, and quantize per KIVI at bits=8:
  - key   : per-channel  -> quant_and_pack_kcache (group along token axis)
  - value : per-token    -> quant_and_pack_vcache (group along head_dim axis)
The most recent `residual_length` tokens stay fp16 and are NOT saved (per spec):
only the packed-INT8 portion (codes + scale + min) is dumped, snapshot taken right
after prefill (before generation).

Save layout (per-layer): <out>/<task>/sample_<n>/layer_<i>.pt
    -> {k_code,k_scale,k_min,v_code,v_scale,v_min}   (+ <sample>/_meta.json)
"""
import os, sys, json, argparse, time
sys.path.insert(0, "/SSD/JSY/smoothquant")
sys.path.insert(0, "/SSD/JSY/KIVI")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from quant.new_pack import quant_and_pack_kcache, quant_and_pack_vcache

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCALES = "/SSD/JSY/smoothquant/regen_inputs/llama-3.1-8b-instruct.pt"
RULER = ["niah_multikey_1", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad"]
OTHER = ["gsm8k_cot", "longbench_hotpotqa"]
DEV = "cuda:0"


def get_prompts(task_name, limit, seqlen):
    from lm_eval.tasks import TaskManager, get_task_dict
    meta = {"max_seq_lengths": [seqlen], "tokenizer": MODEL, "pretrained": MODEL}
    tm = TaskManager(metadata=meta)
    td = get_task_dict([task_name], tm)
    # flatten potential group nesting
    def flat(d):
        o = {}
        for k, v in d.items():
            o.update(flat(v)) if isinstance(v, dict) else o.update({k: v})
        return o
    t = list(flat(td).values())[0]
    t.build_all_requests(limit=limit, rank=0, world_size=1)
    # one instance per doc for generate_until; dedup by doc_id, keep order
    seen, prompts = set(), []
    for inst in t.instances:
        did = inst.doc_id
        if did in seen:
            continue
        seen.add(did)
        prompts.append(inst.args[0])
        if len(prompts) >= limit:
            break
    return prompts


@torch.no_grad()
def dump_sample(model, tok, prompt, gs, res, kbits, vbits, sample_dir):
    # per-layer layout: <sample_dir>/layer_<i>.pt + <sample_dir>/_meta.json
    os.makedirs(sample_dir, exist_ok=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model(ids, use_cache=True, past_key_values=DynamicCache())
    legacy = out.past_key_values.to_legacy_cache()  # tuple of (k,v) per layer
    T = ids.shape[1]
    n_quant_k = ((T - res) // gs) * gs          # key grouped along tokens -> multiple of gs
    n_quant_v = max(T - res, 0)                  # value grouped along head_dim
    nbytes = 0
    for i, (k, v) in enumerate(legacy):
        entry = {}
        if n_quant_k > 0:
            kc, ks, kmn = quant_and_pack_kcache(k[:, :, :n_quant_k, :].contiguous(), gs, kbits)
            entry.update(k_code=kc.cpu(), k_scale=ks.half().cpu(), k_min=kmn.half().cpu())
        if n_quant_v > 0:
            vc, vs, vmn = quant_and_pack_vcache(v[:, :, :n_quant_v, :].contiguous(), gs, vbits)
            entry.update(v_code=vc.cpu(), v_scale=vs.half().cpu(), v_min=vmn.half().cpu())
        lp = os.path.join(sample_dir, f"layer_{i}.pt")
        torch.save(entry, lp)
        nbytes += os.path.getsize(lp)
    json.dump({"T": T, "n_quant_k": int(n_quant_k), "n_quant_v": int(n_quant_v),
               "group_size": gs, "residual_length": res, "k_bits": kbits, "v_bits": vbits,
               "note": "packed INT8 only; fp16 residual excluded; post-prefill snapshot"},
              open(os.path.join(sample_dir, "_meta.json"), "w"), indent=2)
    del out, legacy
    return nbytes, T


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--residual-length", type=int, default=128)
    ap.add_argument("--k-bits", type=int, default=8)
    ap.add_argument("--v-bits", type=int, default=8)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--out", default="compressed_data/kv_kivi8_of_w8a8_smoothquant_llama_31_8b")
    ap.add_argument("--tasks", nargs="+", default=None, help="subset of tasks to dump (default: all 7)")
    args = ap.parse_args()
    todo = args.tasks if args.tasks else (RULER + OTHER)

    tok = AutoTokenizer.from_pretrained(MODEL)
    print("loading + smoothing + W8A8 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, attn_implementation="flash_attention_2")
    smooth_lm(model, torch.load(SCALES), args.alpha)
    model = quantize_model(model, "per_channel", "per_token", quantize_bmm_input=False)
    model.eval()

    total_bytes = 0
    for task in todo:
        sl = args.seqlen if task in RULER else None
        print(f"\n=== {task} (limit {args.limit}, seqlen {sl}) ===", flush=True)
        prompts = get_prompts(task, args.limit, args.seqlen)
        odir = os.path.join(args.out, task)
        os.makedirs(odir, exist_ok=True)
        for n, p in enumerate(prompts):
            t0 = time.time()
            sdir = os.path.join(odir, f"sample_{n}")
            sz, T = dump_sample(model, tok, p, args.group_size, args.residual_length,
                                args.k_bits, args.v_bits, sdir)
            total_bytes += sz
            torch.cuda.empty_cache()
            print(f"  sample_{n}: T={T} -> {sdir}/ (32 layer files, {sz/1e6:.1f}MB, {time.time()-t0:.1f}s)", flush=True)

    # write index
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"model": MODEL, "alpha": args.alpha, "group_size": args.group_size,
                   "residual_length": args.residual_length, "k_bits": args.k_bits, "v_bits": args.v_bits,
                   "tasks": RULER + OTHER, "limit": args.limit, "ruler_seqlen": args.seqlen,
                   "format": "<task>/sample_<n>/layer_<i>.pt -> {k_code,k_scale,k_min,v_code,v_scale,v_min} (+_meta.json)",
                   "note": "packed INT8 only (fp16 residual excluded), post-prefill snapshot"}, f, indent=2)
    print(f"\nDONE. total dumped: {total_bytes/1e9:.2f} GB -> {args.out}/")


if __name__ == "__main__":
    main()
