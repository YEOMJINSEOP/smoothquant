"""Eyeball that the combined (W8A8 + KIVI-INT8) model generates coherent tokens.

Generates greedily on a few real lm-eval prompts (gsm8k_cot, niah_multikey_1@4K,
ruler_vt@4K) with the KiviINT8Cache, and prints ORIGINAL vs COMBINED outputs so we
can confirm token generation is sane (not repetition/garbage) and the needle is found.
"""
import sys
sys.path.insert(0, "/SSD/JSY/smoothquant")
sys.path.insert(0, "/SSD/JSY/KIVI")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from lm_eval.tasks import TaskManager, get_task_dict
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from jsyeom.sq_kivi.kivi_int8_cache import KiviINT8Cache

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCALES = "act_scales/llama-3.1-8b-instruct.pt"
TASK_DIR = "/SSD/JSY/smoothquant/jsyeom/sq_kivi/lm_eval_tasks"
DEV = "cuda:0"


def get_prompt(task, seqlen=4096):
    meta = {"max_seq_lengths": [seqlen], "tokenizer": MODEL, "pretrained": MODEL}
    tm = TaskManager(include_path=TASK_DIR, metadata=meta)
    td = get_task_dict([task], tm)
    def flat(d):
        o = {}
        for k, v in d.items():
            o.update(flat(v)) if isinstance(v, dict) else o.update({k: v})
        return o
    t = list(flat(td).values())[0]
    t.build_all_requests(limit=1, rank=0, world_size=1)
    inst = t.instances[0]
    target = t.doc_to_target(t.dataset[t.config.test_split or "test"][inst.doc_id]) if False else None
    return inst.args[0]


@torch.no_grad()
def gen(model, tok, prompt, n, cache):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, past_key_values=cache, use_cache=True)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), ids.shape[1]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    cases = [("gsm8k_cot", None, 200), ("niah_multikey_1", 4096, 60), ("ruler_vt", 4096, 30)]
    prompts = {}
    for task, sl, _ in cases:
        prompts[task] = get_prompt(task, sl or 4096)
        print(f"[{task}] prompt tokens = {len(tok(prompts[task]).input_ids)}")

    print("\n##### ORIGINAL #####")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV,
                                                 attn_implementation="flash_attention_2").eval()
    for task, sl, n in cases:
        txt, plen = gen(model, tok, prompts[task], n, DynamicCache())
        print(f"\n--- [{task}] (plen={plen}) ORIGINAL ---\n{repr(txt)}")
    del model; torch.cuda.empty_cache()

    print("\n\n##### COMBINED (W8A8 + KIVI-INT8) #####")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV,
                                                 attn_implementation="flash_attention_2").eval()
    smooth_lm(model, torch.load(SCALES), 0.85)
    model = quantize_model(model, "per_channel", "per_token", quantize_bmm_input=False)
    for task, sl, n in cases:
        cache = KiviINT8Cache(k_bits=8, v_bits=8, group_size=32, residual_length=128)
        txt, plen = gen(model, tok, prompts[task], n, cache)
        print(f"\n--- [{task}] (plen={plen}) COMBINED ---\n{repr(txt)}")


if __name__ == "__main__":
    main()
