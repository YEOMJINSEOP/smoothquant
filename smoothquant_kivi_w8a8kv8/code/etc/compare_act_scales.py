"""Compare our reproduced activation scales against mit-han-lab's published scales.

Validates that our calibration pipeline (Pile-val, 512 samples, seq_len 512) faithfully
reproduces the official SmoothQuant scales before we trust it on Llama-3.1-8B-Instruct.

Usage:
  python jsyeom/sq_kivi/compare_act_scales.py \
      --ours act_scales/llama-2-13b-ours.pt \
      --ref  act_scales/llama-2-13b.pt
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()

    ours = torch.load(args.ours, map_location="cpu")
    ref = torch.load(args.ref, map_location="cpu")

    ok = ours.keys() == ref.keys()
    print(f"keys: ours={len(ours)} ref={len(ref)} | identical_keyset={ok}")
    if not ok:
        only_ours = set(ours) - set(ref)
        only_ref = set(ref) - set(ours)
        if only_ours:
            print(f"  only in ours ({len(only_ours)}): {list(only_ours)[:5]}")
        if only_ref:
            print(f"  only in ref  ({len(only_ref)}): {list(only_ref)[:5]}")

    keys = [k for k in ref.keys() if k in ours]

    # Global flattened metrics
    o = torch.cat([ours[k].float().flatten() for k in keys])
    r = torch.cat([ref[k].float().flatten() for k in keys])

    abs_diff = (o - r).abs()
    rel_diff = abs_diff / (r.abs() + 1e-8)
    cos = torch.nn.functional.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
    # Pearson correlation
    om, rm = o - o.mean(), r - r.mean()
    pearson = (om @ rm / (om.norm() * rm.norm() + 1e-12)).item()

    print("\n=== GLOBAL (all channels, all layers) ===")
    print(f"  num values        : {o.numel()}")
    print(f"  cosine similarity : {cos:.6f}")
    print(f"  pearson corr      : {pearson:.6f}")
    print(f"  max  abs diff     : {abs_diff.max().item():.4e}")
    print(f"  mean abs diff     : {abs_diff.mean().item():.4e}")
    print(f"  median rel diff   : {rel_diff.median().item():.4e}")
    print(f"  mean   rel diff   : {rel_diff.mean().item():.4e}")
    print(f"  p99    rel diff   : {rel_diff.quantile(0.99).item():.4e}")
    print(f"  frac rel<1%       : {(rel_diff < 0.01).float().mean().item():.4f}")
    print(f"  frac rel<5%       : {(rel_diff < 0.05).float().mean().item():.4f}")

    # Per-key worst offenders
    per_key = []
    for k in keys:
        a, b = ours[k].float(), ref[k].float()
        rd = ((a - b).abs() / (b.abs() + 1e-8)).mean().item()
        per_key.append((rd, k))
    per_key.sort(reverse=True)
    print("\n=== worst 5 keys by mean rel diff ===")
    for rd, k in per_key[:5]:
        print(f"  {rd:.4e}  {k}")
    print("=== best 3 keys ===")
    for rd, k in per_key[-3:]:
        print(f"  {rd:.4e}  {k}")

    verdict = "PASS" if (pearson > 0.999 and rel_diff.median().item() < 0.02) else "REVIEW"
    print(f"\nVERDICT: {verdict}  (expect near-identical if pipeline is faithful)")


if __name__ == "__main__":
    main()
