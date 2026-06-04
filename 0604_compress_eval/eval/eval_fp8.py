"""Task 3 — accuracy eval: ORIGINAL vs (W-FP8 + A-FP16 + KV-FP8).

Two model variants:
  original : Llama-3.1-8B-Instruct fp16 (flash attn), plain DynamicCache
  combined : fp16 -> per-tensor FP8(E4M3) fake-quant of QPROJ weights (A stays FP16)
             KV cache cast to per-tensor FP8(E4M3) via Fp8KVCache injected into generate()

Mirrors project-1 eval_lmeval.py exactly (same task groups / configs / seed locking);
only build_model + the KV cache class differ (FP8 type-conversion instead of INT quant).

Run one (variant, group) per process, parallelizable across GPUs:
  CUDA_VISIBLE_DEVICES=0 python eval_lmeval_fp8.py --variant original --group ruler
  CUDA_VISIBLE_DEVICES=1 python eval_lmeval_fp8.py --variant combined --group ruler
"""
import os, sys, json, argparse, time
# Full determinism: PYTHONHASHSEED must be set before interpreter start, so re-exec once.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from fp8_quant import FP8_DTYPE, apply_fp8_weight_fakequant, Fp8KVCache

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RULER = ["niah_multikey_1", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad"]
GROUPS = {"ruler": RULER, "gsm8k": ["gsm8k_cot"], "hotpot": ["longbench_hotpotqa"]}
GCFG = {"ruler":  dict(limit=100,  bs=1, maxlen=34000, ruler=True),
        "gsm8k":  dict(limit=None, bs=8, maxlen=4096,  ruler=False),
        "hotpot": dict(limit=None, bs=1, maxlen=17000, ruler=False)}


class Fp8HFLM(HFLM):
    """HFLM that injects a fresh Fp8KVCache into every generation (KV FP8)."""
    def set_fp8(self, **kw):
        self._fp8_kwargs = kw
        return self

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["past_key_values"] = Fp8KVCache(**self._fp8_kwargs)
        return super()._model_generate(context, max_length, stop, **generation_kwargs)


def build_model(variant):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda:0", attn_implementation="flash_attention_2")
    model.eval()
    if variant == "combined":
        apply_fp8_weight_fakequant(model, FP8_DTYPE)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "combined"], required=True)
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    ap.add_argument("--limit", type=int, default=None, help="override per-task sample limit")
    ap.add_argument("--batch-size", default=None, help="int or 'auto'")
    ap.add_argument("--seed", type=int, default=0, help="locks all RNGs (random/numpy/torch/fewshot) for reproducible RULER prompts")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    tasks = GROUPS[args.group]
    cfg = GCFG[args.group]
    is_ruler = cfg["ruler"]
    limit = args.limit if args.limit is not None else cfg["limit"]
    batch_size = int(args.batch_size) if args.batch_size else cfg["bs"]
    max_length = cfg["maxlen"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[{args.variant}/{args.group}] loading model ...", flush=True)
    model = build_model(args.variant)

    hflm_cls = Fp8HFLM if args.variant == "combined" else HFLM
    lm = hflm_cls(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=max_length)
    if args.variant == "combined":
        lm.set_fp8(fp8_dtype=FP8_DTYPE)

    meta = {"max_seq_lengths": [32768], "tokenizer": MODEL, "pretrained": MODEL} if is_ruler else {}
    tm = TaskManager(metadata=meta) if meta else TaskManager()

    print(f"[{args.variant}/{args.group}] eval tasks={tasks} limit={limit} bs={batch_size} maxlen={max_length} seed={args.seed}", flush=True)
    t0 = time.time()
    res = evaluator.simple_evaluate(model=lm, tasks=tasks, limit=limit, batch_size=batch_size,
                                    task_manager=tm, bootstrap_iters=0,
                                    random_seed=args.seed, numpy_random_seed=args.seed,
                                    torch_random_seed=args.seed, fewshot_random_seed=args.seed)
    dt = time.time() - t0

    os.makedirs(args.out, exist_ok=True)
    save = {"variant": args.variant, "group": args.group, "tasks": tasks, "limit": limit,
            "batch_size": batch_size, "max_length": max_length, "ruler_seqlen": 32768 if is_ruler else None,
            "weight_method": "per_tensor FP8 E4M3 (W-FP8, A-FP16)" if args.variant == "combined" else "fp16",
            "kv_method": "per_tensor FP8 E4M3" if args.variant == "combined" else "fp16",
            "seed": args.seed,
            "config": {"fp8_format": "e4m3fn"},
            "elapsed_sec": dt, "results": res["results"]}
    fp = os.path.join(args.out, f"{args.variant}_{args.group}.json")
    json.dump(save, open(fp, "w"), indent=2, default=str)
    print(f"\n=== {args.variant}/{args.group} done in {dt/60:.1f} min -> {fp} ===")
    for tname, r in res["results"].items():
        print(f"  {tname}: " + ", ".join(f"{k}={v}" for k, v in r.items() if "stderr" not in k and isinstance(v, (int, float))))


if __name__ == "__main__":
    main()
