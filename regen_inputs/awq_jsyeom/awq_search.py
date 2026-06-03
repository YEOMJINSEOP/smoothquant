"""Phase 1 — AWQ search on Llama-3.1-8B-Instruct (standard ASYMMETRIC W4, g128).

Mirrors awq.entry's run_awq path but as a standalone script that stubs the
`awq_inference_engine` CUDA kernel (only needed for REAL quant / WQLinear; the
search + fake-quant paths don't use it). Saves AWQ results (per-channel scales +
clips) to awq_cache/.
"""
import sys, types, os, argparse
sys.modules.setdefault("awq_inference_engine", types.ModuleType("awq_inference_engine"))
sys.path.insert(0, "/SSD/JSY/llm-awq")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from awq.quantize.pre_quant import run_awq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--w-bit", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=128)
    ap.add_argument("--zero-point", action="store_true", default=True,
                    help="asymmetric (standard AWQ); default True")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--out", default="/SSD/JSY/llm-awq/awq_cache/llama-3.1-8b-instruct-w4-g128.pt")
    args = ap.parse_args()
    MODEL = args.model

    q_config = {"zero_point": args.zero_point, "q_group_size": args.q_group_size}
    print(f"AWQ search: {MODEL}  w_bit={args.w_bit} q_config={q_config} n_samples={args.n_samples}")

    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config.use_cache = False
    enc = AutoTokenizer.from_pretrained(MODEL, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    ).eval()

    awq_results = run_awq(model, enc, w_bit=args.w_bit, q_config=q_config,
                          n_samples=args.n_samples, seqlen=args.seqlen)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(awq_results, args.out)
    print(f"AWQ results saved → {args.out}")
    print("keys:", list(awq_results.keys()))


if __name__ == "__main__":
    main()
