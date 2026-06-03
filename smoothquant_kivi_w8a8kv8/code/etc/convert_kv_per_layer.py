"""Migrate KV dumps from per-sample files to per-layer files.

Before: <root>/<task>/sample_<n>.pt = {"_meta":..., "layer_0":{...}, ..., "layer_31":{...}}
After:  <root>/<task>/sample_<n>/layer_<i>.pt = {k_code,k_scale,k_min,v_code,v_scale,v_min}
        <root>/<task>/sample_<n>/_meta.json
The original sample_<n>.pt is removed after a verified write.
"""
import os, sys, json, argparse, glob
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="compressed_data/kv_kivi8_of_w8a8_smoothquant_llama_31_8b")
    ap.add_argument("--tasks", nargs="+", default=None, help="subset (default: all task dirs)")
    args = ap.parse_args()

    tasks = args.tasks or sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    print(f"converting tasks: {tasks}")
    n_files = 0
    for task in tasks:
        tdir = os.path.join(args.root, task)
        samples = sorted(glob.glob(os.path.join(tdir, "sample_*.pt")))
        for sp in samples:
            name = os.path.splitext(os.path.basename(sp))[0]  # sample_N
            sdir = os.path.join(tdir, name)
            os.makedirs(sdir, exist_ok=True)
            d = torch.load(sp, map_location="cpu")
            layer_keys = [k for k in d if k.startswith("layer_")]
            for lk in layer_keys:
                torch.save(d[lk], os.path.join(sdir, f"{lk}.pt"))
            if "_meta" in d:
                json.dump(d["_meta"], open(os.path.join(sdir, "_meta.json"), "w"), indent=2)
            # verify count then remove the monolithic sample file
            written = len(glob.glob(os.path.join(sdir, "layer_*.pt")))
            assert written == len(layer_keys), f"{sdir}: wrote {written} != {len(layer_keys)}"
            os.remove(sp)
            n_files += written
        print(f"  {task}: {len(samples)} samples -> per-layer files")
    print(f"done. wrote {n_files} per-layer files across {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
