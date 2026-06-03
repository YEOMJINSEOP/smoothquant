"""Verify our AWQ search reproduces mit-han-lab's published zoo result.

AWQ result = {"scale": [(prev_op, [layers], scale_tensor), ...],
              "clip":  [(name, clip_tensor), ...]}.
Compares ours vs reference by matching keys (prev_op / name) and reporting
cosine sim + relative diff on the scale/clip tensors.
"""
import argparse, torch


def load(p):
    d = torch.load(p, map_location="cpu")
    scale = {s[0]: s[2].float() for s in d["scale"]}   # prev_op -> scale
    clip = {c[0]: c[1].float() for c in d["clip"]}      # name -> clip
    return scale, clip


def cmp_dict(ours, ref, label):
    keys = [k for k in ref if k in ours]
    miss = [k for k in ref if k not in ours]
    o = torch.cat([ours[k].flatten() for k in keys])
    r = torch.cat([ref[k].flatten() for k in keys])
    cos = torch.nn.functional.cosine_similarity(o[None], r[None]).item()
    rel = ((o - r).abs() / (r.abs() + 1e-8))
    print(f"\n=== {label}: {len(keys)}/{len(ref)} keys matched (missing {len(miss)}) ===")
    print(f"  values            : {o.numel()}")
    print(f"  cosine similarity : {cos:.6f}")
    print(f"  max  abs diff     : {(o-r).abs().max().item():.4e}")
    print(f"  median rel diff   : {rel.median().item():.4e}")
    print(f"  mean   rel diff   : {rel.mean().item():.4e}")
    print(f"  frac rel<1%       : {(rel<0.01).float().mean().item():.4f}")
    print(f"  frac rel<5%       : {(rel<0.05).float().mean().item():.4f}")
    # worst keys
    per = sorted(((((ours[k]-ref[k]).abs()/(ref[k].abs()+1e-8)).mean().item()), k) for k in keys)
    print(f"  worst key: {per[-1][1]} (mean rel {per[-1][0]:.3e}) | best: {per[0][1]} ({per[0][0]:.3e})")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()
    os_, oc = load(args.ours)
    rs, rc = load(args.ref)
    c1 = cmp_dict(os_, rs, "SCALE")
    c2 = cmp_dict(oc, rc, "CLIP")
    print(f"\nVERDICT: {'PASS (reproduces official AWQ)' if c1>0.999 and c2>0.999 else 'REVIEW'}  scale_cos={c1:.5f} clip_cos={c2:.5f}")


if __name__ == "__main__":
    main()
