# eval_all — 4-method accuracy matrix on N GPUs → one CSV

Runs the full eval comparing **original / sq (W8A8+KV8) / awq (W4A16+KV8) / fp8 (W-FP8+KV-FP8)**
across **ruler / gsm8k / hotpot**, parallelized over GPUs, merged into `results_matrix/results.csv`.

12 jobs (4 methods × 3 groups) flow through a GPU worker pool (ruler-first → heavy jobs start
immediately, one per GPU). Each job is a separate process pinned to one GPU. **Seed is locked**
(`--seed 0`) so RULER prompts are identical across methods → fair comparison. **Resumable**
(finished jobs skipped); a failed job is logged and skipped, the rest continue.

## Prerequisites (on the eval server)
- The conda env from projects 1/2 (torch 2.8.0+cu128, transformers 4.56.1, flash-attn 2.8.3,
  lm_eval fork, safetensors). No extra install for fp8 (torch already has float8).
- `smoothquant` repo (this) pulled. An upstream `llm-awq` clone (auto-cloned if missing).
- `HF_TOKEN` not required for eval (only model access — gated Llama-3.1 login already done in the env).

`run_eval_matrix.py` does idempotent **setup** itself: copies `act_scales`, AWQ scripts (incl. eval),
AWQ search cache and source overlay from `regen_inputs/` into the right places. Eval needs **no**
weight/KV regeneration and **no** KIVI clone (the INT8 KV cache class is bundled in the repo).

## Run
```bash
conda activate <env>
cd /SSD/JSY/smoothquant/eval_all

# full matrix on 4 GPUs (RULER 32K is slow — hours; detach it):
setsid nohup python run_eval_matrix.py --gpus 0 1 2 3 >> eval_matrix.out 2>&1 &
tail -f eval_matrix.out

# quick smoke (tiny limit on every group):
python run_eval_matrix.py --gpus 0 1 2 3 --limit 2

# subset / rebuild CSV only:
python run_eval_matrix.py --gpus 0 1 --methods original fp8 --groups gsm8k
python run_eval_matrix.py --collect-only
```

Output: `results_matrix/<method>/<variant>_<group>.json` (raw lm-eval) + `results_matrix/results.csv`
(long format: method, group, task, metric, value, limit, batch_size, max_length, seed, elapsed_min)
and a printed side-by-side summary.
