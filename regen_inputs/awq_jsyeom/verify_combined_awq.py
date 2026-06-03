"""Project 2 Phase 2 — numerical verification of the combined model:
   original  vs  W4A16 (AWQ asym g128)  vs  W4A16 + KIVI-INT8 KV.

Build: Llama-3.1-8B-Instruct -> apply_awq(our official search) -> pseudo_quantize W4
-> KiviINT8Cache(k=v=8,g32,r128). Mirrors project-1 verify_combined.py.
Also confirms apply_awq + pseudo_quantize work under transformers 4.56 (weight-only ops).
"""
import sys, types, torch
sys.modules.setdefault("awq_inference_engine", types.ModuleType("awq_inference_engine"))
sys.path.insert(0, "/SSD/JSY/llm-awq")
sys.path.insert(0, "/SSD/JSY/smoothquant")
sys.path.insert(0, "/SSD/JSY/KIVI")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from awq.quantize.pre_quant import apply_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight
from jsyeom.sq_kivi.kivi_int8_cache import KiviINT8Cache

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
AWQ = "/SSD/JSY/llm-awq/awq_cache/llama-3.1-8b-instruct-w4-g128.pt"
DEV = "cuda:0"
GS, RES = 32, 128


@torch.no_grad()
def decode_logits(model, ids, ctx_len, cache):
    o1 = model(ids[:, :ctx_len], use_cache=True, past_key_values=cache)
    o2 = model(ids[:, ctx_len:ctx_len + 1], use_cache=True, past_key_values=o1.past_key_values)
    return o2.logits[:, -1, :].float()


@torch.no_grad()
def gen(model, ids, cache, n=40):
    return model.generate(ids, max_new_tokens=n, do_sample=False, past_key_values=cache, use_cache=True)[0, ids.shape[1]:]


def cmp(a, b):
    return (torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item(),
            (a.argmax(-1) == b.argmax(-1)).float().mean().item())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = ("The history of artificial intelligence began in antiquity. "
            "Modern AI research was founded at a Dartmouth workshop in 1956. ") * 12
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    ctx = min(400, ids.shape[1] - 1)
    print(f"seq={ids.shape[1]} ctx={ctx}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV,
                                                 attn_implementation="flash_attention_2").eval()
    ref = decode_logits(model, ids, ctx, DynamicCache())
    gref = gen(model, ids[:, :ctx], DynamicCache())

    print("applying AWQ (apply_awq + pseudo_quantize W4) ...", flush=True)
    apply_awq(model, torch.load(AWQ, map_location="cpu"))
    pseudo_quantize_model_weight(model, w_bit=4, q_config={"zero_point": True, "q_group_size": 128})
    model.to(DEV)
    # 4-bit grid check
    g0 = model.model.layers[0].self_attn.q_proj.weight.data[0, :128]
    print(f"W4 grid: layer0 q_proj group0 distinct={g0.unique().numel()} (<=16)")
    w4 = decode_logits(model, ids, ctx, DynamicCache())
    gw4 = gen(model, ids[:, :ctx], DynamicCache())

    mk = lambda: KiviINT8Cache(k_bits=8, v_bits=8, group_size=GS, residual_length=RES)
    comb = decode_logits(model, ids, ctx, mk())
    gcomb = gen(model, ids[:, :ctx], mk())

    print("\n=== next-token logits vs ORIGINAL (decode reading past KV) ===")
    for name, v in [("W4A16", w4), ("W4A16+KIVI8", comb)]:
        c, t = cmp(ref, v); print(f"  {name:14s}: cos={c:.5f} top1-match={t:.3f}")
    c, t = cmp(w4, comb); print(f"  {'KIVI vs W4A16':14s}: cos={c:.5f} top1-match={t:.3f}  (isolates KV-INT8)")

    print("\n=== greedy gen (40 tok) ===")
    print("  ORIGINAL    :", repr(tok.decode(gref, skip_special_tokens=True)))
    print("  W4A16       :", repr(tok.decode(gw4, skip_special_tokens=True)))
    print("  W4A16+KIVI8 :", repr(tok.decode(gcomb, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
