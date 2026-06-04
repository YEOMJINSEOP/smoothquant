"""Task 3 — accuracy evaluation: ORIGINAL vs (W8A8 + KIVI-INT8 KV).

Two model variants:
  original : Llama-3.1-8B-Instruct fp16 (flash attn), plain DynamicCache
  combined : fp16 -> smooth_lm(alpha) -> quantize_model(W8A8, bmm_input=False)  [option 1]
             KV cache quantized by KIVI-INT8 via KiviINT8Cache injected into generate()

Task groups:
  ruler  : niah_multikey_1, ruler_vt, ruler_cwe, ruler_fwe, ruler_qa_squad
           @ 32K (metadata max_seq_lengths=[32768]); limit 100; batch 1
  gsm8k  : gsm8k_cot           (full)
  hotpot : longbench_hotpotqa (LongBench, ~14K ctx, 200 samples)  (full, batch 1)

Run one (variant, group) per process so they can be parallelized across GPUs, e.g.:
  CUDA_VISIBLE_DEVICES=0 python eval_lmeval.py --variant original --group ruler
  CUDA_VISIBLE_DEVICES=1 python eval_lmeval.py --variant combined --group ruler
"""
import os, sys, json, argparse, time
# Full determinism: PYTHONHASHSEED must be set before interpreter start, so re-exec once.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
sys.path.insert(0, "/SSD/JSY/smoothquant")
sys.path.insert(0, "/SSD/JSY/KIVI")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from kivi_int8_cache import KiviINT8Cache

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCALES = "/SSD/JSY/smoothquant/regen_inputs/llama-3.1-8b-instruct.pt"
RULER = ["niah_multikey_1", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad"]
GROUPS = {"ruler": RULER, "gsm8k": ["gsm8k_cot"], "hotpot": ["longbench_hotpotqa"]}
# per-group eval config: (default limit, batch_size, max_length, needs RULER 32K metadata)
GCFG = {"ruler":  dict(limit=100,  bs=1, maxlen=34000, ruler=True),
        "gsm8k":  dict(limit=None, bs=8, maxlen=4096,  ruler=False),   # batch fixed 8
        "hotpot": dict(limit=None, bs=1, maxlen=17000, ruler=False)}   # longbench hotpot ~14K (max 16.4K); batch fixed 1


class KiviHFLM(HFLM):
    """HFLM that injects a fresh KiviINT8Cache into every generation (KV INT8)."""
    def set_kivi(self, **kw):
        self._kivi_kwargs = kw
        return self

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["past_key_values"] = KiviINT8Cache(**self._kivi_kwargs)
        return super()._model_generate(context, max_length, stop, **generation_kwargs)


def build_model(variant, alpha):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda:0", attn_implementation="flash_attention_2")
    model.eval()
    if variant == "combined":
        smooth_lm(model, torch.load(SCALES), alpha)
        model = quantize_model(model, weight_quant="per_channel", act_quant="per_token",
                               quantize_bmm_input=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["original", "combined"], required=True)
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    ap.add_argument("--limit", type=int, default=None, help="override per-task sample limit")
    ap.add_argument("--batch-size", default=None, help="int or 'auto'")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--residual-length", type=int, default=128)
    ap.add_argument("--k-bits", type=int, default=8)
    ap.add_argument("--v-bits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="locks all RNGs (random/numpy/torch/fewshot) for reproducible RULER prompts")
    ap.add_argument("--out", default="results/phase6")
    args = ap.parse_args()

    tasks = GROUPS[args.group]
    cfg = GCFG[args.group]
    is_ruler = cfg["ruler"]
    limit = args.limit if args.limit is not None else cfg["limit"]
    batch_size = int(args.batch_size) if args.batch_size else cfg["bs"]
    max_length = cfg["maxlen"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[{args.variant}/{args.group}] loading model ...", flush=True)
    model = build_model(args.variant, args.alpha)

    hflm_cls = KiviHFLM if args.variant == "combined" else HFLM
    lm = hflm_cls(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=max_length)
    if args.variant == "combined":
        lm.set_kivi(k_bits=args.k_bits, v_bits=args.v_bits,
                    group_size=args.group_size, residual_length=args.residual_length)

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
            "seed": args.seed,
            "config": {"alpha": args.alpha, "group_size": args.group_size,
                       "residual_length": args.residual_length, "k_bits": args.k_bits, "v_bits": args.v_bits,
                       "quantize_bmm_input": False},
            "elapsed_sec": dt, "results": res["results"]}
    fp = os.path.join(args.out, f"{args.variant}_{args.group}.json")
    json.dump(save, open(fp, "w"), indent=2, default=str)
    print(f"\n=== {args.variant}/{args.group} done in {dt/60:.1f} min -> {fp} ===")
    for tname, r in res["results"].items():
        print(f"  {tname}: " + ", ".join(f"{k}={v}" for k, v in r.items() if "stderr" not in k and isinstance(v, (int, float))))


if __name__ == "__main__":
    main()
