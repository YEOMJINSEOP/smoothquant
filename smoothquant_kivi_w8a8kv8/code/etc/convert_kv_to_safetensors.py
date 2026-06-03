"""Convert KV-cache dumps from per-layer .pt (pickle) to .safetensors.

Each `<task>/sample_<n>/layer_<i>.pt` is a flat dict of tensors
  {k_code(int32), k_scale(fp16), k_min(fp16), v_code(int32), v_scale(fp16), v_min(fp16)}
→ written as `layer_<i>.safetensors` (raw, safe, framework-agnostic). `_meta.json` kept as-is.

Per-file: write safetensors → verify round-trip equality → remove the .pt (avoids 2x disk).
Safe to re-run (skips already-converted).
"""
import os, sys, glob, argparse
import torch
from safetensors.torch import save_file, load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="compressed_data/kv_kivi8_of_w8a8_smoothquant_llama_31_8b")
    ap.add_argument("--keep-pt", action="store_true", help="keep .pt after converting (default: remove)")
    args = ap.parse_args()

    pts = sorted(glob.glob(os.path.join(args.root, "*", "sample_*", "layer_*.pt")))
    print(f"found {len(pts)} .pt layer files under {args.root}")
    converted = skipped = 0
    for i, pt in enumerate(pts):
        st = pt[:-3] + ".safetensors"
        if os.path.exists(st) and not os.path.exists(pt):
            skipped += 1
            continue
        d = torch.load(pt, map_location="cpu")
        tensors = {k: v.contiguous() for k, v in d.items()}
        save_file(tensors, st)
        # verify round-trip (exact equality; same dtypes)
        r = load_file(st)
        assert set(r) == set(tensors) and all(torch.equal(r[k], tensors[k]) for k in tensors), f"mismatch {st}"
        if not args.keep_pt:
            os.remove(pt)
        converted += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pts)} done", flush=True)
    print(f"DONE: converted={converted} skipped={skipped}. .safetensors files now: "
          f"{len(glob.glob(os.path.join(args.root, '*', 'sample_*', 'layer_*.safetensors')))}")


if __name__ == "__main__":
    main()
