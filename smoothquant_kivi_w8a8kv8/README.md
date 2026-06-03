# SmoothQuant W8A8 + KIVI-INT8 KV — 종합 디렉토리

Llama-3.1-8B-Instruct에 **SmoothQuant W8A8 (weight·activation INT8) + KV cache INT8 (KIVI 방식)** 를
orthogonal하게 결합하고, (1) weight 저장, (2) KV cache 저장, (3) 원본 vs 결합 정확도 평가를 수행한 결과 모음.

작업일: 2026-06-03. 환경: 4× NVIDIA L40S, conda env `lm_eval_flash`.

---

## 디렉토리 구조

```
smoothquant_kivi_w8a8kv8/
├── README.md                  # (이 파일)
├── code/                      # 핵심 파이프라인 (3개 Task) + 진행로그
│   ├── PROGRESS.md            # 전 과정 결정·명령·결과·검증 로그 (감사용)
│   ├── kivi_int8_cache.py     # KiviINT8Cache(DynamicCache): KV INT8 fake-quant 주입
│   ├── save_w8a8_weights.py   # Task1: W8A8 weight 추출(int8+scale)
│   ├── split_weights_per_layer.py  # Task1: weight를 layer별 파일로 분리
│   ├── dump_kv_cache.py       # Task2: KIVI-INT8 KV 덤프(per-layer)
│   ├── eval_lmeval.py         # Task3: 원본 vs 결합 lm-eval 평가(KiviHFLM)
│   └── etc/                   # 비핵심: 검증·디버그·마이그레이션·공유 유틸
│       ├── compare_act_scales.py      # 캘리브 파이프라인 검증(공개본 대조)
│       ├── verify_combined.py         # 결합모델 logit 수치검증
│       ├── gen_sanity.py              # 생성 토큰 정상성 점검
│       ├── convert_kv_per_layer.py    # KV 구포맷→per-layer 마이그레이션(미사용)
│       ├── convert_kv_to_safetensors.py  # KV .pt→safetensors 변환(공유용)
│       └── upload_to_hf.py            # private HF dataset 업로드
├── act_scales/
│   └── llama-3.1-8b-instruct.pt     # SmoothQuant 캘리브 산출물(Pile-val 512×512)
├── compressed_data/           # ★ 산출 데이터 (51GB)
│   ├── docs.md                # 데이터 포맷 엄밀 명세(dtype/shape/수식/복원)
│   ├── w8_of_w8a8_smoothquant_llama_31_8b/      # Task1: W8 weight (8.5GB)
│   │   ├── layer_0..31.safetensors             # 레이어별 int8 weight + fp16 scale + LN
│   │   ├── embeddings.safetensors              # embed/lm_head/norm (fp16)
│   │   └── config.json
│   └── kv_kivi8_of_w8a8_smoothquant_llama_31_8b/  # Task2: KIVI-INT8 KV (43GB)
│       ├── <task>/sample_<n>/layer_<i>.pt      # k/v code+scale+min (per-layer)
│       │                     /_meta.json
│       └── index.json
└── results/
    └── phase6/                # Task3: 평가 결과 json + summary.json
```

> **포맷 상세는 `compressed_data/docs.md`**, **전 과정 추적은 `code/PROGRESS.md`** 참조.

---

## 결합 구성 (요약)

```
Llama-3.1-8B-Instruct (fp16)
  → smooth_lm(act_scales/llama-3.1-8b-instruct.pt, alpha=0.85)
  → quantize_model(weight=per_channel, act=per_token, quantize_bmm_input=False)   # W8A8 (옵션1: attention BMM은 fp16)
  → KiviINT8Cache(k_bits=8, v_bits=8, group_size=32, residual_length=128)         # KV INT8 (Key per-channel / Value per-token)
```
- W8A8: 가중치=per-channel 대칭 INT8, 활성=per-token INT8(추론 시 동적).
- KV: KIVI 방식 비대칭 INT8 fake-quant (KIVI 2/4bit 커널 대신 동일 수식 fake-quant).

---

## Task 결과

### Task 1 — Weight 저장  (`compressed_data/w8_of_...`)
per-channel 대칭 int8 + fp16 scale, layer별 safetensors. 역추출 오차 max 0.0(평가모델과 비트 일치).

### Task 2 — KV Cache 저장  (`compressed_data/kv_kivi8_of_...`)
7 task × 20 sample × 32 layer = 4480 .pt. prefill 직후 packed INT8(+scale+min)만 저장(fp16 residual 제외).
- 저장셋: RULER@4K {niah_multikey_1, ruler_vt, ruler_cwe, ruler_fwe, ruler_qa_squad} 각 20 + gsm8k_cot 20 + longbench_hotpotqa 20.

### Task 3 — Accuracy 평가  (`results/phase6/`)
원본(fp16) vs 결합(W8A8+KIVI-INT8):

| Task | original | combined | Δ |
| --- | --- | --- | --- |
| RULER@32K niah_multikey_1 | 1.000 | 1.000 | +0.000 |
| RULER@32K ruler_vt        | 0.998 | 1.000 | +0.002 |
| RULER@32K ruler_cwe       | 0.030 | 0.132 | +0.102 |
| RULER@32K ruler_fwe       | 0.940 | 0.927 | -0.013 |
| RULER@32K ruler_qa_squad  | 0.734 | 0.735 | +0.001 |
| gsm8k_cot (flexible)      | 0.768 | 0.754 | -0.014 |
| gsm8k_cot (strict)        | 0.754 | 0.740 | -0.014 |
| longbench_hotpotqa (F1)   | 0.240 | 0.205 | -0.035 |

→ long-context 검색 사실상 무손실, gsm8k −1.4pp. INT8 W8A8 + INT8 KV가 정확도를 작은 손실로 보존.

---

## 재현 (참고)

코드는 원본 repo에 의존(import): `PYTHONPATH=/SSD/JSY/smoothquant`, `sys.path += /SSD/JSY/KIVI`, `HF_HUB_CACHE=/SSD/JSY`, env `lm_eval_flash`.
(상대 출력경로 `compressed_data/...`, `act_scales/...`, `results/phase6`는 `/SSD/JSY/smoothquant`에서 실행 기준 — 본 디렉토리로 옮긴 사본이므로 재실행 시 경로 조정 필요.)

```bash
# 캘리브 (act_scales 생성)
python examples/generate_act_scales.py --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output-path act_scales/llama-3.1-8b-instruct.pt --dataset-path dataset/val.jsonl.zst --num-samples 512 --seq-len 512
# Task1
python code/save_w8a8_weights.py --alpha 0.85 && python code/split_weights_per_layer.py --remove-monolith
# Task2
python code/dump_kv_cache.py --limit 20
# Task3 (변형/그룹별, 4 GPU 병렬)
python code/eval_lmeval.py --variant {original|combined} --group {ruler|gsm8k|hotpot}
```

상세 명령·검증값은 `code/PROGRESS.md`.
