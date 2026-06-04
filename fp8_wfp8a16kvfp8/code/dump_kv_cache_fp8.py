"""Task 2 — dump the FP8 (E4M3, per-tensor scaled) KV cache of the W-FP8 model.

For each sample: build the lm-eval prompt, prefill with the W-FP8 model, take the fp16
KV (post-RoPE) from the cache, and cast per-tensor to fp8:
    scale = |x|.amax() / 448   (one scalar per layer's K, one per V)
    code  = (x / scale).to(float8_e4m3fn)
The WHOLE KV is cast (no fp16 residual, unlike KIVI). Snapshot taken right after prefill.

Written directly as one safetensors per sample (resumable upload):
  <out>/<task>/sample_<n>.safetensors
      keys: layer_<i>.k_code (fp8), layer_<i>.k_scale (fp32[1]),
            layer_<i>.v_code (fp8), layer_<i>.v_scale (fp32[1])
      metadata: {"T": ..., "n_layers": ..., "fp8_format": "e4m3fn"}
  <out>/index.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/SSD/JSY/smoothquant")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from safetensors.torch import save_file
from fp8_quant import FP8_DTYPE, apply_fp8_weight_fakequant, quant_per_tensor_fp8

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RULER = ["niah_multikey_1", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad"]
OTHER = ["gsm8k_cot", "longbench_hotpotqa"]
DEV = "cuda:0"


def get_prompts(task_name, limit, seqlen):
    from lm_eval.tasks import TaskManager, get_task_dict
    meta = {"max_seq_lengths": [seqlen], "tokenizer": MODEL, "pretrained": MODEL}
    tm = TaskManager(metadata=meta)
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
        did = inst.doc_id
        if did in seen:
            continue
        seen.add(did)
        prompts.append(inst.args[0])
        if len(prompts) >= limit:
            break
    return prompts


@torch.no_grad()
def dump_sample(model, tok, prompt, out_path):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model(ids, use_cache=True, past_key_values=DynamicCache())
    legacy = out.past_key_values.to_legacy_cache()  # tuple of (k,v) per layer
    T = ids.shape[1]
    tensors = {}
    for i, (k, v) in enumerate(legacy):
        kc, ks = quant_per_tensor_fp8(k.contiguous(), FP8_DTYPE)
        vc, vs = quant_per_tensor_fp8(v.contiguous(), FP8_DTYPE)
        tensors[f"layer_{i}.k_code"] = kc.cpu()
        tensors[f"layer_{i}.k_scale"] = ks.float().cpu()
        tensors[f"layer_{i}.v_code"] = vc.cpu()
        tensors[f"layer_{i}.v_scale"] = vs.float().cpu()
    save_file(tensors, out_path, metadata={"T": str(T), "n_layers": str(len(legacy)),
                                           "fp8_format": "e4m3fn", "scale": "per-tensor amax/448",
                                           "residual": "none"})
    del out, legacy
    return os.path.getsize(out_path), T


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--out", default="compressed_data/kv_fp8_of_wfp8a16kvfp8_llama_31_8b")
    ap.add_argument("--tasks", nargs="+", default=None, help="subset of tasks (default: all 7)")
    args = ap.parse_args()
    todo = args.tasks if args.tasks else (RULER + OTHER)

    tok = AutoTokenizer.from_pretrained(MODEL)
    print("loading + W-FP8 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, attn_implementation="flash_attention_2")
    apply_fp8_weight_fakequant(model, FP8_DTYPE)
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
            fp = os.path.join(odir, f"sample_{n}.safetensors")
            if os.path.exists(fp):  # resume: skip already-dumped samples
                print(f"  sample_{n}: exists, skip", flush=True)
                continue
            sz, T = dump_sample(model, tok, p, fp)
            total_bytes += sz
            torch.cuda.empty_cache()
            print(f"  sample_{n}: T={T} -> {fp} ({sz/1e6:.1f}MB, {time.time()-t0:.1f}s)", flush=True)

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"model": MODEL, "weight_quant": "per_tensor FP8 E4M3",
                   "kv_quant": "per_tensor FP8 E4M3 (whole KV, no residual)",
                   "fp8_format": "e4m3fn", "tasks": RULER + OTHER, "limit": args.limit,
                   "ruler_seqlen": args.seqlen,
                   "format": "<task>/sample_<n>.safetensors -> layer_<i>.{k_code,k_scale,v_code,v_scale}; meta has T",
                   "note": "fp8 codes + per-tensor scale, post-prefill snapshot"}, f, indent=2)
    print(f"\nDONE. total dumped: {total_bytes/1e9:.2f} GB -> {args.out}/")


if __name__ == "__main__":
    main()
