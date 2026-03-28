import os
os.environ["HF_HOME"] = "/SSD/.cache/"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# 기존 evaluate_smoothed_hf.py와 동일하되, quantize_model()만 추가된 형태.
# smoothed 모델(jsyeom/llama-2-13b-hf-smooth)을 Hub에서 로드한 뒤
# W8A8 fake quantization을 적용하고 lm_eval로 평가한다.
#
# 사용법:
# CUDA_VISIBLE_DEVICES=0 python jsyeom/smooth_and_quant/evaluate_smooth_and_quant.py --task leaderboard_mmlu_pro leaderboard_gpqa
# CUDA_VISIBLE_DEVICES=0 python jsyeom/smooth_and_quant/evaluate_smooth_and_quant.py --task leaderboard_bbh --output-dir ./results/smooth-and-quant

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.fake_quant import quantize_model
from datetime import datetime
import json
import torch
import wandb
import argparse

# wandb 설정
WANDB_PROJECT = "smooth-and-quant"
WANDB_ENTITY = None

def json_serializable(obj):
    """JSON 직렬화 불가능한 객체를 문자열로 변환"""
    try:
        return str(obj)
    except:
        return f"<non-serializable: {type(obj).__name__}>"


def save_results_to_json(save_data, save_path):
    """결과를 JSON 파일로 저장"""
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=json_serializable)
    print(f"  Saved to: {save_path}")


def log_to_wandb(save_data, model_name, task, timestamp, config):
    """결과를 wandb에 기록"""
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=model_name,
        name=f"{model_name}-{task}-{timestamp}",
        config=config
    )
    # Logs 섹션에 기록 (console 출력)
    print(json.dumps(save_data, indent=2, default=json_serializable))
    wandb.finish()


# 사용 가능한 task pool
AVAILABLE_TASKS = [
    "leaderboard_mmlu_pro",   # 1h
    "leaderboard_gpqa",       # 1h
    "leaderboard_ifeval",     # 9h
    "leaderboard_math_hard",  # 10h
    "leaderboard_musr",       # 10h
    "leaderboard_bbh",        # 29h
]

# 평가 대상 모델: Hub에 업로드된 smoothed 모델
MODEL_ID = "jsyeom/llama-2-13b-hf-smooth"

# argument parser 설정
parser = argparse.ArgumentParser(description="Evaluate smoothed + W8A8 quantized model with lm-eval")
parser.add_argument("--task", type=str, nargs="+", choices=AVAILABLE_TASKS, default=AVAILABLE_TASKS, help=f"Task(s) to evaluate. Available: {AVAILABLE_TASKS}")
parser.add_argument("--output-dir", type=str, default="./results/smooth-and-quant", help="Output directory for results")
args = parser.parse_args()

TASKS = args.task
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_name = MODEL_ID.split("/")[-1]  # llama-2-13b-hf-smooth

print(f"\n{'='*60}")
print(f"Evaluating: {MODEL_ID}")
print(f"Pipeline:   load smoothed model → quantize_model (W8A8)")
print(f"Tasks: {', '.join(TASKS)}")
print(f"{'='*60}\n")

model_dir = f"{OUTPUT_DIR}/{model_name}"
os.makedirs(model_dir, exist_ok=True)

try:
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    torch_dtype = torch.float16
    print(f"Loading model with dtype: {torch_dtype}...")

    # load smoothed model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # ============================================================
    # quantize_model 적용 ← evaluate_smoothed_hf.py에 없는 추가 단계
    # ============================================================
    print("Applying W8A8 quantization (weight=per_channel, act=per_token)...")
    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
    )
    print("Quantization applied.")
    
    """
    quantize_llama_like()가 Llama 모델의 모든 Linear 레이어를 W8A8Linear로 교체합니다 (fake_quant.py:196-230):
    LlamaMLP      →  gate_proj, up_proj, down_proj
    LlamaAttention →  q_proj, k_proj, v_proj, o_proj
    """
    # load model with lm_eval (pass pre-loaded model directly)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype=torch_dtype,
    )

    # task별 evaluation 및 save
    for task in TASKS:
        print(f"\n  Evaluating: {task}")

        task_results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task],
            batch_size=64,
            device="cuda"
        )

        # task별로 결과 저장 (필요한 정보만)
        task_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_data = {
            'summary': {
                'model': MODEL_ID,
                'pipeline': 'smoothed model + quantize_model (W8A8 fake quant)',
                'weight_quant': 'per_channel',
                'act_quant': 'per_token',
                'task': task,
                'timestamp': task_timestamp
            },
            'results': task_results.get('results', {}),
            'groups': task_results.get('groups', {}),
            'group_subtasks': task_results.get('group_subtasks', {}),
            'configs': task_results.get('configs', {}),
            'versions': task_results.get('versions', {}),
            'n-shot': task_results.get('n-shot', {}),
            'higher_is_better': task_results.get('higher_is_better', {}),
            'n-samples': task_results.get('n-samples', {}),
        }

        # 결과 저장 및 wandb 기록
        save_path = f"{model_dir}/eval_{task}_{task_timestamp}.json"
        save_results_to_json(save_data, save_path)
        log_to_wandb(save_data, model_name, task, task_timestamp, {
            "model": MODEL_ID,
            "pipeline": "smoothed + W8A8",
            "task": task,
            "dtype": str(torch_dtype),
            "weight_quant": "per_channel",
            "act_quant": "per_token",
        })

except Exception as e:
    print(f"Error evaluating {MODEL_ID}: {e}")
    import traceback
    traceback.print_exc()

finally:
    # GPU 메모리 정리
    if 'lm' in locals():
        del lm
    if 'model' in locals():
        del model
    if 'tokenizer' in locals():
        del tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("GPU memory cleared.")

print(f"\n{'='*60}")
print("\nEVALUATION COMPLETED")
print(f"\n{'='*60}")
