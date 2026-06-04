"""Run the full accuracy-eval matrix across 4 methods x 3 task groups on N GPUs, -> one CSV.

Methods (vs the SAME fp16 baseline):
  original : Llama-3.1-8B-Instruct fp16
  sq       : SmoothQuant W8A8 + KIVI-INT8 KV
  awq      : AWQ W4A16 (asym, g128) + KIVI-INT8 KV
  fp8      : W-FP8(E4M3) + A-FP16 + KV-FP8(E4M3)
Task groups: ruler (5 RULER tasks @32K, limit 100), gsm8k (gsm8k_cot, full), hotpot (longbench_hotpotqa, full).

12 jobs (4 methods x 3 groups) run through a GPU worker pool (ruler-first so the heavy jobs
start immediately, one per GPU). Each job is a separate process pinned to one GPU. Seed is
locked (--seed, default 0) so RULER prompts are identical across methods (fair comparison).
Resumable: a job whose result JSON already exists is skipped. A failed job is logged and skipped;
the rest continue. After all jobs, every result JSON is merged into results_matrix/results.csv.

Setup (idempotent) wires the cross-repo deps for sq/awq eval from regen_inputs/ (act_scales,
AWQ scripts incl. eval, AWQ search cache, source overlay). Needs the smoothquant repo + an
upstream llm-awq clone; clones llm-awq if missing.

Run (after `conda activate <env>`):
  python run_eval_matrix.py --gpus 0 1 2 3
  # detached / overnight:
  setsid nohup python run_eval_matrix.py --gpus 0 1 2 3 >> eval_matrix.out 2>&1 &
  tail -f eval_matrix.out
  # quick smoke (tiny limit on every group):
  python run_eval_matrix.py --gpus 0 1 2 3 --limit 2
"""
import os, sys, json, csv, time, queue, threading, subprocess, shutil, argparse

ROOT = os.environ.get("ROOT", "/SSD/JSY")
SQ = os.environ.get("SQ", f"{ROOT}/smoothquant")
AWQ = os.environ.get("AWQ", f"{ROOT}/llm-awq")
INPUTS = os.environ.get("INPUTS", f"{SQ}/regen_inputs")
FP8C = f"{SQ}/fp8_wfp8a16kvfp8/code"
PYTHON = sys.executable

GROUPS = ["ruler", "gsm8k", "hotpot"]          # ruler first = heaviest first
# method -> (script, cwd, variant, extra PYTHONPATH)
METHODS = {
    "original": (f"{FP8C}/eval_lmeval_fp8.py",                       FP8C, "original", FP8C),
    "sq":       (f"{SQ}/smoothquant_kivi_w8a8kv8/code/eval_lmeval.py", SQ,  "combined", SQ),
    "awq":      (f"{AWQ}/jsyeom/eval_lmeval_awq.py",                 AWQ,  "combined", f"{AWQ}:{SQ}"),
    "fp8":      (f"{FP8C}/eval_lmeval_fp8.py",                       FP8C, "combined", FP8C),
}

LOCK = threading.Lock()
def log(msg):
    with LOCK:
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] {msg}", flush=True)


def setup():
    """Idempotent: wire cross-repo deps for sq/awq eval from regen_inputs/."""
    log("SETUP: wiring deps from regen_inputs/")
    assert os.path.isdir(INPUTS), f"missing {INPUTS} (pull the smoothquant repo)"
    # SmoothQuant act_scales
    os.makedirs(f"{SQ}/act_scales", exist_ok=True)
    src = f"{INPUTS}/llama-3.1-8b-instruct.pt"
    if os.path.exists(src):
        shutil.copy(src, f"{SQ}/act_scales/")
    # AWQ repo
    if not os.path.isdir(AWQ):
        log(f"  llm-awq missing -> cloning upstream into {AWQ}")
        subprocess.run(["git", "clone", "https://github.com/mit-han-lab/llm-awq.git", AWQ], check=True)
    if not os.path.isdir(f"{AWQ}/jsyeom"):
        shutil.copytree(f"{INPUTS}/awq_jsyeom", f"{AWQ}/jsyeom")
    else:  # always refresh the eval + upload scripts
        for f in ["eval_lmeval_awq.py", "save_w4_weights.py", "dump_kv_cache_awq.py", "upload_to_hf.py"]:
            s = f"{INPUTS}/awq_jsyeom/{f}"
            if os.path.exists(s):
                shutil.copy(s, f"{AWQ}/jsyeom/{f}")
    os.makedirs(f"{AWQ}/awq_cache", exist_ok=True)
    cache = f"{INPUTS}/llama-3.1-8b-instruct-w4-g128.pt"
    if os.path.exists(cache) and not os.path.exists(f"{AWQ}/awq_cache/llama-3.1-8b-instruct-w4-g128.pt"):
        shutil.copy(cache, f"{AWQ}/awq_cache/")
    open(f"{AWQ}/awq/__init__.py", "a").close()
    overlay = f"{INPUTS}/awq_overlay/awq"
    if os.path.isdir(overlay):
        subprocess.run(f"cp -rf {overlay}/. {AWQ}/awq/", shell=True)
    # sanity: required files present
    for p in [f"{SQ}/act_scales/llama-3.1-8b-instruct.pt",
              f"{AWQ}/awq_cache/llama-3.1-8b-instruct-w4-g128.pt",
              f"{AWQ}/jsyeom/eval_lmeval_awq.py"]:
        assert os.path.exists(p), f"setup incomplete, missing: {p}"
    log("  SETUP OK")


def job_outfile(results, method, group):
    variant = METHODS[method][2]
    return os.path.join(results, method, f"{variant}_{group}.json")


def run_job(gpu, method, group, results, seed, limit):
    script, cwd, variant, pypath = METHODS[method]
    out = os.path.join(results, method)
    os.makedirs(out, exist_ok=True)
    jf = job_outfile(results, method, group)
    if os.path.exists(jf):
        log(f"[GPU{gpu}] skip {method}/{group} (already done)")
        return True
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = pypath + ":" + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [PYTHON, "-u", script, "--variant", variant, "--group", group,
           "--out", out, "--seed", str(seed)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    log(f"[GPU{gpu}] START {method}/{group}  ({variant})")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env)
    dt = (time.time() - t0) / 60
    if r.returncode == 0 and os.path.exists(jf):
        log(f"[GPU{gpu}] DONE  {method}/{group}  in {dt:.1f} min")
        return True
    log(f"[GPU{gpu}] FAIL  {method}/{group}  (rc={r.returncode}, {dt:.1f} min)")
    return False


def collect_csv(results):
    rows = []
    for method in METHODS:
        for group in GROUPS:
            jf = job_outfile(results, method, group)
            if not os.path.exists(jf):
                continue
            d = json.load(open(jf))
            elapsed = round(d.get("elapsed_sec", 0) / 60, 1)
            for task, metrics in d.get("results", {}).items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and "stderr" not in k and k != "alias":
                        rows.append({"method": method, "group": group, "task": task,
                                     "metric": k, "value": v,
                                     "limit": d.get("limit"), "batch_size": d.get("batch_size"),
                                     "max_length": d.get("max_length"), "seed": d.get("seed"),
                                     "elapsed_min": elapsed})
    csv_path = os.path.join(results, "results.csv")
    cols = ["method", "group", "task", "metric", "value", "limit", "batch_size",
            "max_length", "seed", "elapsed_min"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["group"], r["task"], r["metric"], r["method"])))
    log(f"CSV: {len(rows)} rows -> {csv_path}")
    # compact summary (primary metric per task, methods side by side)
    print("\n=== SUMMARY (value by method) ===", flush=True)
    keys = sorted({(r["group"], r["task"], r["metric"]) for r in rows})
    methods = list(METHODS)
    print(f"{'group/task/metric':<48} " + "  ".join(f"{m:>9}" for m in methods))
    for g, t, mt in keys:
        cells = []
        for m in methods:
            vals = [r["value"] for r in rows if r["group"] == g and r["task"] == t and r["metric"] == mt and r["method"] == m]
            cells.append(f"{vals[0]:.4f}" if vals else "-")
        print(f"{(g+'/'+t+'/'+mt):<48} " + "  ".join(f"{c:>9}" for c in cells))
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="override per-task sample limit on EVERY group (smoke only)")
    ap.add_argument("--results", default=f"{SQ}/eval_all/results_matrix")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument("--collect-only", action="store_true", help="just (re)build the CSV from existing JSONs")
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)

    if args.collect_only:
        collect_csv(args.results); return
    if not args.skip_setup:
        setup()

    jobs = queue.Queue()
    for group in args.groups:           # ruler-first ordering preserved
        for method in args.methods:
            jobs.put((method, group))
    total = jobs.qsize()
    log(f"MATRIX: {total} jobs ({len(args.methods)} methods x {len(args.groups)} groups) on GPUs {args.gpus}, seed={args.seed}")

    failures = []
    def worker(gpu):
        while True:
            try:
                method, group = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                ok = run_job(gpu, method, group, args.results, args.seed, args.limit)
                if not ok:
                    failures.append(f"{method}/{group}")
            except Exception as e:
                failures.append(f"{method}/{group} ({e})")
                log(f"[GPU{gpu}] EXCEPTION {method}/{group}: {e}")
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in args.gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"ALL JOBS finished in {(time.time()-t0)/60:.1f} min; failures={failures or 'none'}")
    collect_csv(args.results)
    if failures:
        log("Re-run the SAME command to retry failed jobs (done ones are skipped).")
        sys.exit(1)


if __name__ == "__main__":
    main()
