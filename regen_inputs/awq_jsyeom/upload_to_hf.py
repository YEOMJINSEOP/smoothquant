"""Upload compressed_data to a PRIVATE HuggingFace dataset repo (team sharing).

Prereq:
  pip install -U huggingface_hub
  huggingface-cli login           # or export HF_TOKEN=hf_xxx   (needs write access)

Usage:
  python upload_to_hf.py --repo-id your-org/smoothquant-kivi-w8a8kv8 \
      --path /SSD/JSY/smoothquant/smoothquant_kivi_w8a8kv8/compressed_data

Notes:
- repo is created PRIVATE by default. Add teammates via repo Settings → "Members"/collaborators,
  or `HfApi().add_collaborator(...)` (org repos: invite to the org/team).
- Data is DERIVED FROM meta-llama/Llama-3.1-8B-Instruct → Llama 3.1 Community License applies.
  Keep private for internal sharing; if ever made public, include the Llama license + "Built with Llama".
- Uses upload_large_folder (resumable, handles many files / large size). Safe to re-run to resume.
"""
import os, argparse
from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. your-org/smoothquant-kivi-w8a8kv8")
    ap.add_argument("--path", default="/SSD/JSY/smoothquant/smoothquant_kivi_w8a8kv8/compressed_data",
                    help="folder to upload (default: compressed_data)")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--public", action="store_true", help="make repo public (default: private)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--include", nargs="+", default=None,
                    help="allow_patterns (glob, relative to --path). e.g. 'w8_of_*' (weights) or 'kv_*' (KV)")
    ap.add_argument("--exclude", nargs="+", default=None, help="ignore_patterns (glob)")
    args = ap.parse_args()

    assert os.path.isdir(args.path), f"not a dir: {args.path}"
    api = HfApi(token=args.token)

    create_repo(args.repo_id, repo_type=args.repo_type, private=not args.public,
                exist_ok=True, token=args.token)
    print(f"repo ready: {args.repo_id} (type={args.repo_type}, private={not args.public})")
    if args.include or args.exclude:
        print(f"  include={args.include} exclude={args.exclude} (folder structure preserved)")

    # resumable large-folder upload (many files); patterns let you split weights vs KV
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=args.path,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
    )
    print(f"upload complete → https://huggingface.co/{args.repo_type}s/{args.repo_id}")


if __name__ == "__main__":
    main()
