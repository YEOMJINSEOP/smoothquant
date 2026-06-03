"""Numerical verification of the combined model:
   original  vs  W8A8  vs  W8A8 + KIVI-INT8 KV.

The KV quantization only affects tokens that attend to PAST (cached) KV, so we
prefill a context then run one decode step that reads the quantized past, and
compare next-token logits. Also does a short greedy generation for a qualitative check.
"""
import sys, argparse
sys.path.insert(0, "/SSD/JSY/smoothquant")
sys.path.insert(0, "/SSD/JSY/KIVI")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from jsyeom.sq_kivi.kivi_int8_cache import KiviINT8Cache

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCALES = "act_scales/llama-3.1-8b-instruct.pt"
ALPHA, GS, RES = 0.85, 32, 128
DEV = "cuda:0"


@torch.no_grad()
def decode_logits(model, ids, ctx_len, cache):
    """Prefill ids[:ctx_len], then decode one more token reading the (possibly quantized) past."""
    ctx = ids[:, :ctx_len]
    nxt = ids[:, ctx_len:ctx_len + 1]
    o1 = model(ctx, use_cache=True, past_key_values=cache)
    cache = o1.past_key_values
    o2 = model(nxt, use_cache=True, past_key_values=cache)
    return o2.logits[:, -1, :].float()


@torch.no_grad()
def gen(model, ids, cache, n=40):
    out = model.generate(ids, max_new_tokens=n, do_sample=False,
                         past_key_values=cache, use_cache=True)
    return out[0, ids.shape[1]:]


def cmp(a, b, tok):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(b, -1), torch.softmax(a, -1), reduction="batchmean").item()
    return cos, top1, kl


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = ("The history of artificial intelligence began in antiquity, with myths and "
            "stories of artificial beings endowed with intelligence by master craftsmen. "
            "Modern AI research was founded at a workshop held on the campus of Dartmouth "
            "College in 1956. ") * 8
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    ctx_len = min(400, ids.shape[1] - 1)
    print(f"seq len={ids.shape[1]}, ctx_len={ctx_len} (residual={RES}, so ~{ctx_len-RES} tokens quantized)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, attn_implementation="flash_attention_2")
    model.eval()

    # (a) ORIGINAL
    ref = decode_logits(model, ids, ctx_len, DynamicCache())
    gen_ref = gen(model, ids[:, :ctx_len], DynamicCache())

    # apply SmoothQuant W8A8 (in place)
    act_scales = torch.load(SCALES)
    smooth_lm(model, act_scales, ALPHA)
    model = quantize_model(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False)

    # (b) W8A8 (fp16 cache)
    w8a8 = decode_logits(model, ids, ctx_len, DynamicCache())
    gen_w8a8 = gen(model, ids[:, :ctx_len], DynamicCache())

    # (c) W8A8 + KIVI-INT8
    def mkcache():
        return KiviINT8Cache(k_bits=8, v_bits=8, group_size=GS, residual_length=RES)
    comb = decode_logits(model, ids, ctx_len, mkcache())
    gen_comb = gen(model, ids[:, :ctx_len], mkcache())

    print("\n=== next-token logits vs ORIGINAL (decode step reading past KV) ===")
    for name, v in [("W8A8", w8a8), ("W8A8+KIVI-INT8", comb)]:
        cos, top1, kl = cmp(ref, v, tok)
        print(f"  {name:16s}: cos={cos:.5f}  top1-match={top1:.3f}  KL={kl:.4e}")
    cos, top1, kl = cmp(w8a8, comb, tok)
    print(f"  {'KIVI vs W8A8':16s}: cos={cos:.5f}  top1-match={top1:.3f}  KL={kl:.4e}  (isolates KV-INT8 effect)")

    print("\n=== greedy generation (40 tok) ===")
    print("  ORIGINAL      :", repr(tok.decode(gen_ref, skip_special_tokens=True)))
    print("  W8A8          :", repr(tok.decode(gen_w8a8, skip_special_tokens=True)))
    print("  W8A8+KIVI-INT8:", repr(tok.decode(gen_comb, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
