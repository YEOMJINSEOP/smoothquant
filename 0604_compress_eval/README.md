# 0604_compress_eval — 압축(weight+KV 저장) + 평가 로직 종합

Llama-3.1-8B-Instruct에 대한 **3가지 압축 방식**의 compressed-data 저장 로직과 accuracy
평가 로직을 한 곳에 모아 정리한 디렉토리. (각 방식의 canonical/배포 위치는 그대로 두고,
핵심 로직을 여기 복사·재배선해 **이 디렉토리에서 바로 실행 가능**하게 함.)

## 3가지 방식
| 방식 | Weight | Activation | KV cache |
|---|---|---|---|
| **W8A8KV8** (SmoothQuant + KIVI) | per-channel **symmetric INT8** (α=0.85) | per-token sym INT8 (동적) | **KIVI INT8** gs=32, res=128, Key per-channel/Value per-token, asym |
| **W4A16KV8** (AWQ + KIVI) | **asymmetric INT4**, group_size=128 (공식 search) | FP16 | KIVI INT8 (동일) |
| **WFP8A16KVFP8** | **per-tensor FP8 (E4M3)** scale=amax/448 | FP16 | **per-tensor FP8 (E4M3)**, residual 없음 |

압축 대상 linear: q·k·v·o·gate·up·down ×32 (embed/LayerNorm/lm_head는 fp16). 평가는
**fake-quant 정확도 측정**(quant→dequant→fp16 연산), activation은 FP16.

## 디렉토리 구조
```
0604_compress_eval/
  shared/   kivi_int8_cache.py   # KIVI INT8 KV 캐시 (W8A8·AWQ eval에서 사용)
            fp8_quant.py         # FP8 per-tensor primitives + Fp8KVCache
  save/     w8a8_save_weights.py / w8a8_dump_kv.py / w8a8_split_weights.py
            awq_save_weights.py  / awq_dump_kv.py
            fp8_save_weights.py  / fp8_dump_kv.py
  eval/     eval_w8a8.py / eval_awq.py / eval_fp8.py
            run_eval_matrix.py   # 4-method × 3-group 병렬 평가 → results.csv
```
> 정리용 **사본**임. 상위 라이브러리(`smoothquant`, `awq.quantize`, KIVI `quant.new_pack`)와
> AWQ search 캐시는 여전히 `/SSD/JSY/{smoothquant,llm-awq,KIVI}`에 의존(sys.path). 공유 모듈
> (kivi/fp8)은 `shared/`에서 import하도록 재배선됨. canonical 원본을 수정하면 여기 사본도 갱신 필요.

## 저장 포맷 (dtype)
| | Weight | KV codes | KV scale/zero |
|---|---|---|---|
| W8A8KV8 | int8 + fp16 scale | int32 (8bit×4 packing) | fp16 scale + fp16 min |
| W4A16KV8 | uint8 (int4×2/byte) + fp16 scale + uint8 zero | int32 (동일) | fp16 scale + fp16 min |
| WFP8A16KVFP8 | float8_e4m3fn + fp32 scale | float8_e4m3fn (packing 없음) | fp32 scale (zero 없음) |

## 출처 (압축 로직)
- SmoothQuant: 원본 `smoothquant`(`smooth_lm`+`quantize_model`) 직접 호출
- AWQ: 원본 `llm-awq`(`apply_awq`+`pseudo_quantize_model_weight`) 직접 호출 (search 재실행 안 함)
- KIVI KV: 원본 `quant_and_pack_kcache/vcache` 직접 호출
- FP8: 자체 구현(type conversion). save↔eval 일관성 + smoke 검증.

## 실행
```bash
conda activate lm_eval_flash      # torch 2.8.0+cu128, transformers 4.56.1, flash-attn 2.8.3, lm_eval fork
# --- 저장 (예: W8A8) ---
python save/w8a8_save_weights.py --out <dir>
python save/w8a8_dump_kv.py      --out <dir> --limit 20 --seqlen 4096
# (AWQ는 awq_*, FP8는 fp8_* 스크립트 사용)

# --- 평가 (4 method × 3 group 병렬, 1 CSV) ---
python eval/run_eval_matrix.py --gpus 0 1 2 3                       # ruler-first
python eval/run_eval_matrix.py --gpus 0 1 2 3 --groups gsm8k hotpot ruler   # 가벼운 것 먼저
python eval/run_eval_matrix.py --collect-only                      # 부분 결과 CSV 갱신
# 결과: eval/results_matrix/<method>/<variant>_<group>.json + results.csv
```
- `run_eval_matrix.py`의 `setup()`이 sq/awq 평가에 필요한 cross-repo 의존(act_scales, AWQ
  스크립트·search 캐시·overlay)을 `regen_inputs/`에서 자동 배선.
- seed 고정(random/numpy/torch/fewshot + `PYTHONHASHSEED=0` 재실행) → 네 방식 동일 프롬프트.

## 평가 결과 (Llama-3.1-8B-Instruct, lm-eval-harness fork)
| Method | GSM8K-CoT strict | GSM8K-CoT flexible | RULER (4-task AVG) | HotpotQA F1 |
|---|---|---|---|---|
| Original (fp16) | 0.754 | 0.768 | 0.918 | 0.263 |
| W8A8KV8 (SmoothQuant+KIVI) | 0.740 | 0.754 | 0.915 | 0.210 |
| W4A16KV8 (AWQ+KIVI) | 0.737 | 0.743 | 0.908 | 0.197 |
| WFP8A16KVFP8 | 0.751 | 0.764 | 0.909 | 0.228 |

- GSM8K-CoT: Exact Match (strict-match / flexible-extract filter), **8-shot**, 1319 samples
- RULER: niah_multikey_1·ruler_vt·ruler_fwe·ruler_qa_squad @**32K** 평균, task당 100 samples
  (`ruler_cwe`는 original 포함 전 방식 near-floor(0.03)라 평균서 제외 — 별도 확인 권장)
- HotpotQA: LongBench `qa_f1_score`, 200 samples
- 측정: completion-style (chat template 미적용), fake-quant accuracy. stderr 미산출(bootstrap_iters=0)
  → 미세차는 noise 가능. baseline은 fp16(모델 학습 dtype bf16과 다름).

## 캐노니컬 원본 위치 (수정 시 동기화 대상)
- W8A8: `smoothquant_kivi_w8a8kv8/code/` + `jsyeom/sq_kivi/kivi_int8_cache.py`
- AWQ: `/SSD/JSY/llm-awq/jsyeom/`
- FP8: `fp8_wfp8a16kvfp8/code/`
- 평가 오케스트레이터 원본: `eval_all/run_eval_matrix.py`
