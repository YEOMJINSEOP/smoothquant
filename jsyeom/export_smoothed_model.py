import torch
import torch.nn as nn
import argparse
import json
import os
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply SmoothQuant smoothing to a model and save (without quantization)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-2-13b-hf",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--act-scales-path",
        type=str,
        default="act_scales/llama-2-13b.pt",
        help="path to pre-computed activation scales (.pt)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="smoothing alpha (0~1, higher = more migration to weights)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="smoothed_models/llama-2-13b-hf-smooth",
        help="local directory to save the smoothed model",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo id to upload (e.g. your-org/llama-2-13b-hf-smooth)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip PPL evaluation before saving",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_ppl(model, tokenizer, n_samples=40):
    """WikiText-2에서 perplexity를 측정한다."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids
    nlls = []
    for i in tqdm.tqdm(range(n_samples), desc="Evaluating PPL"):
        batch = encodings[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float() * 2048)
    return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


@torch.no_grad()
def main():
    args = parse_args()

    # 1) Load model & tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 2) Evaluate baseline PPL (before smoothing)
    if not args.skip_eval:
        print("=== Baseline (FP16) PPL ===")
        baseline_ppl = evaluate_ppl(model, tokenizer)
        print(f"Baseline PPL: {baseline_ppl:.3f}")

    # 3) Load activation scales (auto-download if not found locally)
    if not os.path.exists(args.act_scales_path):
        print(f"Scales not found at {args.act_scales_path}, downloading from HuggingFace...")
        filename = os.path.basename(args.act_scales_path)
        local_dir = os.path.dirname(args.act_scales_path) or "act_scales"
        hf_hub_download(
            repo_id="mit-han-lab/smoothquant-scales",
            filename=filename,
            local_dir=local_dir,
        )
        print(f"Downloaded to {args.act_scales_path}")

    print(f"Loading activation scales: {args.act_scales_path}")
    act_scales = torch.load(args.act_scales_path, map_location="cpu")

    # 4) Apply smoothing (no quantization)
    print(f"Applying smoothing with alpha={args.alpha}")
    smooth_lm(model, act_scales, args.alpha)

    # 5) Evaluate smoothed PPL & compare
    if not args.skip_eval:
        print("=== Smoothed PPL ===")
        smoothed_ppl = evaluate_ppl(model, tokenizer)
        print(f"Smoothed PPL: {smoothed_ppl:.3f}")
        diff = smoothed_ppl - baseline_ppl
        print(f"PPL diff: {diff:+.3f}")

    # 6) Save locally
    print(f"Saving smoothed model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7) Save smoothing metadata
    smoothquant_meta = {
        "base_model": args.model_name,
        "smoothing_alpha": args.alpha,
        "act_scales_source": "mit-han-lab/smoothquant-scales",
        "act_scales_file": os.path.basename(args.act_scales_path),
        "quantization_applied": False,
        "warning": "Do NOT apply smooth_lm() again. Smoothing is already applied.",
    }
    with open(os.path.join(args.output_dir, "smoothquant_config.json"), "w") as f:
        json.dump(smoothquant_meta, f, indent=2)

    # 8) Bundle act_scales alongside the model
    scales_dst = os.path.join(args.output_dir, os.path.basename(args.act_scales_path))
    if not os.path.exists(scales_dst):
        shutil.copy2(args.act_scales_path, scales_dst)
        print(f"Bundled activation scales: {scales_dst}")

    # 9) Generate Model Card (README.md)
    model_card = f"""---
language: en
tags:
  - smoothquant
  - llama2
base_model: {args.model_name}
---

# {os.path.basename(args.output_dir)}

This model has SmoothQuant smoothing applied. No quantization has been applied.

## Smoothing Configuration

| Parameter         | Value                           |
|-------------------|---------------------------------|
| Base model        | `{args.model_name}`             |
| Alpha             | `{args.alpha}`                  |
| Act scales source | `mit-han-lab/smoothquant-scales`|
| Quantization      | None (smoothing only)           |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.push_to_hub or args.output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{args.push_to_hub or args.output_dir}")
```

## Important Notes

- **Do NOT** call `smooth_lm()` again on this model. Smoothing is already applied.
- To apply quantization, simply call `quantize_model(model)`.
- To re-experiment with a different alpha, use the bundled `{os.path.basename(args.act_scales_path)}` on the original base model.
"""
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(model_card)

    print("Saved: smoothquant_config.json, act_scales, README.md")

    # 10) Optionally push to HuggingFace Hub
    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        api = HfApi()
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.push_to_hub,
            repo_type="model",
            create_pr=False,
        )
        print(f"Upload complete: https://huggingface.co/{args.push_to_hub}")

    print("Done.")


if __name__ == "__main__":
    main()



# 로컬 저장만:

# python jsyeom/export_smoothed_model.py \
#     --model-name meta-llama/Llama-2-13b-hf \
#     --act-scales-path act_scales/llama-2-13b.pt \
#     --alpha 0.85 \
#     --output-dir smoothed_models/llama-2-13b-hf-smooth

# HuggingFace Hub 업로드까지:

# python jsyeom/export_smoothed_model.py \
#     --model-name meta-llama/Llama-2-13b-hf \
#     --act-scales-path act_scales/llama-2-13b.pt \
#     --alpha 0.85 \
#     --output-dir smoothed_models/llama-2-13b-hf-smooth \
#     --push-to-hub jsyeom/llama-2-13b-hf-smooth