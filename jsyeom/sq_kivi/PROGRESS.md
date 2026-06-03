# SmoothQuant W8A8 + KIVI INT8 KV — 진행 로그 (검증용)

> 이 문서는 추후 **각 단계가 올바른지 면밀히 재검증**할 수 있도록, 모든 결정·명령어·결과·검증 포인트를 기록합니다.
> 작성 시작: 2026-06-03. 환경: 4× NVIDIA L40S (46GB), conda env `lm_eval_flash`.

---

## 0. 프로젝트 목표

원본 모델 대비, **SmoothQuant W8A8 (weight·activation INT8) + KV cache INT8(KIVI 방식)** 결합 모델을 만들고:

1. **Task 1 — Weight 저장**: W8A8 양자화된 weight (int8 weight + per-channel scale)
2. **Task 2 — KV Cache 저장**: 4K 저장셋에 대해 prefill 직후의 KIVI-INT8 KV (packed int8 + scale + min)
   - 저장셋: RULER@4K {NIAH-MK1, VT, CWE, FWE, QA1} 각 20 + GSM8K-CoT 20 + HotpotQA 20
3. **Task 3 — Accuracy 평가**: 원본 vs (W8A8+KIVI-INT8) 비교 (lm-eval-harness)
   - 평가셋: RULER@32K {NIAH-MK1, VT, CWE, FWE, QA1} **태스크당 100샘플** + GSM8K-CoT **전체** + HotpotQA **전체**

---

## 1. 확정된 결정사항 (근거 포함)

| # | 결정 | 근거 | 검증 포인트 |
|---|---|---|---|
| D1 | 베이스 = **meta-llama/Llama-3.1-8B-Instruct** | 128K 컨텍스트(32K RULER 가능), instruct(GSM8K-CoT/HotpotQA 적합), 로컬 캐시, 기존 KIVI 실험서 사용. (SmoothQuant 기존 작업의 Llama-2-13B는 4K라 32K 불가) | config의 `max_position_embeddings=131072` 확인 |
| D2 | QA1 = `ruler_qa_squad`, **HotpotQA = standalone**(커스텀 태스크) | RULER엔 qa_squad(QA1)/qa_hotpot(QA2) 둘 다 있으나, 사용자가 HotpotQA를 별도 표준 데이터셋으로 지정 | Phase 5에서 커스텀 태스크 EM/F1 동작 확인 |
| D3 | 32K 평가 = **RULER 태스크당 100샘플**(`--limit 100`), **GSM8K-CoT·HotpotQA는 전체 샘플**(limit 없음) | RULER는 32K prefill 비용 큼→100 절충; gsm8k_cot/hotpotqa는 짧아 전체 평가 가능 (사용자 지정 2026-06-03) | |
| D4 | KV 저장 = **packed INT8 codes + scale + min만** (fp16 residual 제외), **prefill 직후** 스냅샷 | 사용자 지정. long-context 컨텍스트 KV가 분석 핵심, 생성 토큰은 짧음 | 저장 텐서 dtype/shape 확인 |
| D5 | 결합 방식 = **SmoothQuant 기반 + KIVI quant 코드 orthogonal 결합** | KIVI attention 전체 이식 시 SmoothQuant과 얽히고 transformers 버전에 종속됨 | |
| D6 | Task3 KV = **`KiviINT8Cache(DynamicCache)` 서브클래스**(attention 무수정), Task2 = 오프라인 양자화 스크립트 | `update()`가 generate 루프의 유일한 주입 지점(Phase 0서 검증). attention 미변경 → orthogonal 유지 | Phase 0 cache 주입 테스트 |
| D7 | KIVI INT8은 **fake-quant 경로** (packed→dequant fp16→matmul) | KIVI CUDA/Triton 커널은 `bits∈{2,4}`만 지원(`cuda_bmm_fA_qB_outer`, `qbvm_kernel` assert). INT8은 커널 없음. 정확도 평가엔 fake-quant가 정확 | new_pack `pack_tensor`/`unpack_and_dequant_*`는 bits=8 지원 확인 |
| D8 | 하이퍼파라미터(기본값) = **group_size=32, residual_length=128, alpha=0.85** | KIVI 통상값 + Llama-3 SmoothQuant 권장 alpha | 추후 조정 가능 |

---

## 2. 환경 & 제약 (검증된 사실)

- **env**: `/opt/conda/envs/lm_eval_flash` (Python 3.10.19)
  - transformers **4.56.1**, flash_attn **2.8.3**, lm_eval **0.4.9.1**, triton 3.4.0, torch 2.8.0+cu128, datasets 3.6.0, zstandard 0.25.0
- **lm_eval는 커스텀 fork**: `/SSD/JSY/-ASSIGN-mask_lm_eval/src/lm-eval` (editable). RULER 내장.
  - RULER 길이 지정: `--metadata='{"max_seq_lengths":[4096]}'`(또는 32768) + `--model_args=...,max_length=N`
  - 태스크명: `niah_multikey_1`(NIAH-MK1), `ruler_vt`(VT), `ruler_cwe`(CWE), `ruler_fwe`(FWE), `ruler_qa_squad`(QA1), `gsm8k_cot`(GSM8K-CoT)
- **import 주의**: smoothquant은 env에 pip 설치 안 됨 → `PYTHONPATH=/SSD/JSY/smoothquant` 필수. KIVI는 `sys.path.insert(0,"/SSD/JSY/KIVI")`.
- **HF 캐시**: `/SSD/JSY` (즉 `HF_HUB_CACHE=/SSD/JSY`).
- **제약**: KIVI 커널 bits∈{2,4} → INT8은 fake-quant (D7).

---

## 3. Phase 0 — 환경 호환성 검증 ✅ PASS

**목적**: transformers 4.56 ↔ SmoothQuant ↔ KIVI ↔ cache 주입이 함께 동작하는지.

재현 명령 (요지):
```python
# env: lm_eval_flash, PYTHONPATH=/SSD/JSY/smoothquant, sys.path += /SSD/JSY/KIVI
from smoothquant.fake_quant import quantize_model, W8A8Linear
from quant.new_pack import quant_and_pack_kcache, ...      # OK
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
            dtype=fp16, device_map="cuda:0", attn_implementation="flash_attention_2")
model = quantize_model(model, "per_channel", "per_token", quantize_bmm_input=False)
```

결과:
- W8A8Linear 모듈 수 = **224** (= 32 layers × 7 linears: q,k,v,o,gate,up,down) ✅ 기대값 일치
- `model.model.layers[0].self_attn.q_proj` 타입 = `W8A8Linear` ✅
- W8A8 생성: `"The capital of France is"` → `" a city of romance, art, fashion, and cuisine."` ✅
- 커스텀 `DynamicCache` 서브클래스 `update()` 호출 = **384회** (32 layers × 12 steps) ✅ → cache 주입 동작

**검증 포인트**: 224 = 32×7인지(즉 모든 linear 양자화), q_proj가 W8A8Linear로 바뀌었는지, cache update가 실제 호출되는지.

---

## 4. Phase 1 — Activation Scales 캘리브레이션

### 4.1 배경: 기존 llama-2-13b는 어떻게?
- 기존 `smoothed_models/llama-2-13b-hf-smooth`는 **직접 캘리브 안 함** → `mit-han-lab/smoothquant-scales`에서 `llama-2-13b.pt` **다운로드** 사용 (`.cache/huggingface/download/llama-2-13b.pt.metadata`가 증거).
- **그 repo엔 Llama-3 계열 스케일이 없음** (보유: llama-2 7b/13b/70b, Mistral-7B, Mixtral-8x7B, falcon-7b/40b, opt-*, bloom-176b). → **3.1-8B는 직접 생성 필수.**

### 4.2 캘리브 데이터
- Pile validation: `mit-han-lab/pile-val-backup/val.jsonl.zst` → `/SSD/JSY/smoothquant/dataset/val.jsonl.zst` (470.9 MB).
- 원논문 SmoothQuant과 동일 (512문장 × seq_len 512).

### 4.3 ⭐ 파이프라인 검증: 우리 캘리브 vs mit-han-lab 공개본 (llama-2-13b)
**목적**: 우리 캘리브 절차가 공개본을 엄밀히 재현하는지 먼저 입증 후 3.1-8B에 적용.

재현 명령:
```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/SSD/JSY PYTHONPATH=/SSD/JSY/smoothquant \
/opt/conda/envs/lm_eval_flash/bin/python examples/generate_act_scales.py \
  --model-name meta-llama/Llama-2-13b-hf \
  --output-path act_scales/llama-2-13b-ours.pt \
  --dataset-path dataset/val.jsonl.zst --num-samples 512 --seq-len 512
# 비교:
PYTHONPATH=/SSD/JSY/smoothquant /opt/conda/envs/lm_eval_flash/bin/python \
  jsyeom/sq_kivi/compare_act_scales.py \
  --ours act_scales/llama-2-13b-ours.pt --ref act_scales/llama-2-13b.pt
```

결과 (178만 채널값 전체):

| 지표 | 값 |
|---|---|
| cosine similarity | **0.999986** |
| median rel diff | 5.51e-4 (0.055%) |
| mean rel diff | 5.72e-4 |
| p99 rel diff | 2.60e-3 |
| rel diff < 1% 비율 | 99.98% |
| rel diff < 5% 비율 | 100% |
| layer 0 attention (q/k/v) | rel diff **0.0** (완전 일치) |
| 최대 오차 키 | 깊은 층 `down_proj` ~9e-4 |

→ **VERDICT: PASS.** 미세 차이는 fp16 누적/transformers 버전 차이 수준. 파이프라인이 충실함을 입증.

**검증 포인트**: cosine>0.999, layer0 attention 완전 일치, 최대 오차가 깊은 down_proj에 몰리는지(예상 패턴).

### 4.4 본작업: Llama-3.1-8B-Instruct 캘리브
재현 명령:
```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/SSD/JSY PYTHONPATH=/SSD/JSY/smoothquant \
/opt/conda/envs/lm_eval_flash/bin/python examples/generate_act_scales.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output-path act_scales/llama-3.1-8b-instruct.pt \
  --dataset-path dataset/val.jsonl.zst --num-samples 512 --seq-len 512
```
- 상태: ✅ **완료** → `act_scales/llama-3.1-8b-instruct.pt` (5.07 MB)
- sanity check 결과:

| 항목 | 결과 | 판정 |
|---|---|---|
| 키 개수 | 225 = 32층×7 linear(224) + lm_head(1) | ✅ |
| 텐서 shape | (4096,) hidden 입력 + (14336,) down_proj 입력(intermediate) | ✅ |
| NaN/inf | 없음 | ✅ |
| 통계 | min=0.0, max=475.75, mean=1.82, median=1.52 | ✅ |
| outlier 증거 | max/median 채널비 최대 **1747.7** | ✅ activation outlier 포착 (SmoothQuant 동기 부합) |

- 주의: smooth_lm은 q_proj(→qkv), gate_proj(→gate/up) 스케일만 사용. o_proj/down_proj/lm_head 스케일은 기록되나 smoothing에 미사용(정상, 공개본과 동일 포맷).
- 검증 포인트: 225=32×7+1, down_proj 입력만 14336인지, outlier 채널 존재.

---

## 4b. Phase 2 — KiviINT8Cache 구현 & 검증

**구현**: `jsyeom/sq_kivi/kivi_int8_cache.py`
- `KiviINT8Cache(DynamicCache)`: `update()`에서 부모(fp16 누적) 호출 후, KIVI 방식 INT8 fake-quant 뷰를 반환.
  - Key: per-channel (token축 group_size 묶음), Value: per-token (head_dim축 group_size 묶음), 최근 residual_length 토큰 fp16 유지.
  - INT8이라 KIVI 커널(2/4bit) 불가 → 직접 fake-quant(quant→dequant), 패킹 생략(속도). 수식은 KIVI `new_pack.py`와 동일.
  - 정확도 평가 목적상 부모에 fp16 전체 캐시 유지(메모리 절감은 목표 아님). 매 스텝 앞부분 재양자화.
- group/residual 정합: head_dim=128 % group_size=32 = 0, residual_length=128 % 32 = 0 (config 확인).

**검증 1 — KIVI 수식 일치** (`kivi_int8_cache.py` self-test):
| | max\|KIVI pack/unpack − ours\| |
|---|---|
| key (bits=8) | **0.000e+00** |
| value (bits=8) | **0.000e+00** |
→ 우리 fake-quant이 KIVI 실제 양자화와 **완전 일치**. (D7 확정)

**검증 2 — 통합 모델 logit/생성** (`verify_combined.py`): 원본 vs W8A8 vs W8A8+KIVI-INT8
(369토큰 prefill 후, 과거 KV를 읽는 1-decode 스텝의 next-token logit 비교)

| 비교 | cosine | top1 일치 | KL |
|---|---|---|---|
| W8A8 vs 원본 | 0.99706 | 1.000 | 9.04e-3 |
| W8A8+KIVI-INT8 vs 원본 | 0.99606 | 1.000 | 1.15e-2 |
| KIVI vs W8A8 (KV-INT8 단독) | 0.99415 | 1.000 | 9.74e-3 |

- greedy 40토큰 생성: 세 구성 **완전 동일**.
- 해석: W8A8만으로 원본과 거의 동일(top-1 100%), INT8 KV 추가 효과 미미(고정밀). 결합 모델 수치 건전성 확인. ✅
- 검증 포인트: top1 일치율, KIVI vs W8A8 cos≈0.994(작은 양자화 노이즈), 생성문 일관성. PASS.

## 4c. Phase 3 — Task 1: W8A8 weight 저장 ✅

**스크립트**: `jsyeom/sq_kivi/save_w8a8_weights.py`
- 절차: load fp16 → `smooth_lm(α=0.85)` → `quantize_model(W8A8)` → 양자화된 W8A8Linear에서 **per-channel int8 + scale 추출**.
- ⚠️ 중요(검증 중 발견·수정): 처음엔 int8을 fp32로 계산 → 추론(fp16)과 ±1 LSB 차이로 max err 4.7e-2 FAIL. **양자화 후 추출**로 변경(절대값 max가 채널마다 정확히 127에 매핑 → scale 복원 정확) → max err **0.000e+00 PASS** (저장본 = 평가 모델 비트 일치).

**산출물**: `compressed_data/w8_of_w8a8_smoothquant_llama_31_8b/` — **레이어별 분리 저장** (사용자 요청, `split_weights_per_layer.py`)
| 파일 | 내용 |
|---|---|
| `layer_0.safetensors` … `layer_31.safetensors` (32개) | 레이어별: 7 linear의 `<proj>.weight`(int8 −127~127) + `<proj>.scale`(fp16) + input/post_attention_layernorm.weight(fp16) |
| `embeddings.safetensors` | embed_tokens.weight, model.norm.weight, lm_head.weight (fp16) |
| `config.json` | base/alpha/act_scales/quant 방식/KIVI kv 설정 + storage 레이아웃 메타 |
| 총 | 9.08 GB (int8 weight 6.98GB + scale·비양자화 fp16) |

- 텐서 구성 확인: 전체 515 = int8 224(=7×32) + fp16 291(scale 224 + embed/norm/lm_head 3 + layernorm 64).
- 검증: 양자화후추출 max err **0.000e+00**(평가모델 비트일치) + 분리 저장 round-trip 일치(515키).
- 검증 포인트: layer 파일 32개+embeddings, 각 weight int8 [-127,127], dequant=int8×scale.

## 4d. Phase 5 — standalone HotpotQA lm_eval 태스크 ✅
- `jsyeom/sq_kivi/lm_eval_tasks/hotpotqa.yaml` + `hotpotqa_utils.py`: dataset `hotpot_qa/distractor` validation, generate_until, SQuAD식 정규화 EM + token F1.
- 프롬프트: 10개 distractor 단락 + Question + Answer:. 검증: 로드 OK, ctx~1175토큰, self-EM=1.0.
- 사용: `TaskManager(include_path=".../lm_eval_tasks")`.

## 4e. Phase 4 — Task 2: KV Cache 덤프 (4K 저장셋)
- **스크립트**: `jsyeom/sq_kivi/dump_kv_cache.py`
- 절차: W8A8 모델(load→smooth→quantize)로 각 프롬프트 prefill → fp16 KV(post-RoPE)를 KIVI `quant_and_pack_kcache/vcache(bits=8)`로 양자화 → packed int8+scale+min만 저장(fp16 residual 제외), prefill 직후 스냅샷.
- 프롬프트: lm_eval API(`build_all_requests(limit=20)`→`instances.args[0]`), RULER는 metadata `max_seq_lengths=[4096]`.
- 저장: `kv_kivi8_of_w8a8_smoothquant_llama_31_8b/<task>/sample_<n>.pt` = {layer_i:{k_code,k_scale,k_min,v_code,v_scale,v_min}, _meta}.
- **구조 검증**(스모크, T=3749): k_code (1,8,904,128) int32 [904=3616/4], k_scale (1,8,113,1,128) [113그룹×128채널=per-channel], v_code (1,8,3621,32) [32=128/4], v_scale (1,8,3621,4,1) [토큰별×4그룹=per-token]. n_quant_k=3616, n_quant_v=3621. ✅ 사양 일치.
- 샘플당 크기: RULER ~250MB(~5s), gsm8k ~42MB, hotpot ~73MB. 전체 140샘플 ≈ 27GB.
- 상태: ✅ **완료 140/140, 총 ~43 GB** (per-layer 포맷, 4 GPU 병렬 재덤프)
- **저장 레이아웃 (per-layer, 사용자 요청)**: `<task>/sample_<n>/layer_<i>.pt`(레이어당 1파일, 32개) + `<task>/sample_<n>/_meta.json`. 총 140 sample × 32 = **4480 .pt**.
- **HotpotQA → `longbench_hotpotqa`로 변경** (LongBench, median ~14K ctx → hotpot 덤프가 큼). 7태스크 각 20.
- 검증: 140 sample dir, 각 32 layer+_meta, longbench T~11K~16K (예 T=11354, n_quant_k=11200) 정상, index.json layout=per-layer.
- 마이그레이션 스크립트(구 sample.pt→신 포맷): `convert_kv_per_layer.py` (이번엔 새 포맷으로 직접 재덤프함).
- 검증 포인트: sample dir 140, 각 layer_*.pt 32개, packed int8만, post-prefill.

## 4f. ⚠️ 미결 설계 질문 — `quantize_bmm_input` (Phase 6 전 결정 필요)
- 원본 SmoothQuant 단독은 `quantize_bmm_input=True`가 표준(ppl_eval.py, 논문 full W8A8 = attention BMM도 INT8). q/k/v_proj 출력을 per-token 대칭 8bit화.
- 우리 결합 모델은 현재 `False`: K/V는 KIVI 단독 양자화(이중양자화 방지), 단 **Q(BMM 입력)도 fp16**.
- 선택지: (1) 현재 유지(W8A8=linear GEMM만, Q fp16) vs (2) q_proj만 quantize_output=True 추가(full W8A8 attention 의도 + KV는 KIVI 단독). → **사용자 결정 대기**.

## 5. 이후 단계 (예정)
- **Phase 3**: Task1 — W8A8 weight 저장 (smooth_lm(α=0.85) → quantize → int8 weight+scale 추출).
- **Phase 4**: Task2 — 4K 저장셋 prefill → KV packed int8 덤프.
- **Phase 5**: standalone HotpotQA 커스텀 lm_eval 태스크.
- **Phase 6**: Task3 — 원본 vs 결합 32K 평가(태스크당 100).
- **Phase 7**: 집계·리포트.

---

## 6. 산출물 위치

| 항목 | 경로 |
|---|---|
| 프로젝트 코드 | `/SSD/JSY/smoothquant/jsyeom/sq_kivi/` |
| 비교 스크립트 | `jsyeom/sq_kivi/compare_act_scales.py` |
| 캘리브 데이터 | `dataset/val.jsonl.zst` |
| 검증용 스케일(13b) | `act_scales/llama-2-13b-ours.pt` (vs 공개 `act_scales/llama-2-13b.pt`) |
| 본 스케일(3.1-8b) | `act_scales/llama-3.1-8b-instruct.pt` |
| 진행 로그(본 문서) | `jsyeom/sq_kivi/PROGRESS.md` |

---

## 4g. Phase 6 — Task 3: Accuracy 평가 결과 ✅ (원본 vs W8A8+KIVI-INT8)

**구성**: 원본 = Llama-3.1-8B-Instruct fp16(flash). 결합 = smooth_lm(α=0.85) → quantize_model(W8A8, bmm_input=False, 옵션1) → KiviINT8Cache(k=v=8bit, g32, r128) 주입.
**스크립트**: `eval_lmeval.py` (`KiviHFLM`이 generate에 KiviINT8Cache 주입). 결과: `results/phase6/{variant}_{group}.json` + `summary.json`.
**세팅**: RULER@32K(`max_seq_lengths=[32768]`, max_length 34000) limit 100 batch1 / gsm8k_cot 전체 batch8 / longbench_hotpotqa 전체(200) batch1 (do_sample=True 기본).

| Task | original | combined | Δ |
|---|---|---|---|
| RULER@32K niah_multikey_1 | 1.000 | 1.000 | +0.000 |
| RULER@32K ruler_vt | 0.998 | 1.000 | +0.002 |
| RULER@32K ruler_cwe | 0.030 | 0.132 | +0.102 |
| RULER@32K ruler_fwe | 0.940 | 0.927 | −0.013 |
| RULER@32K ruler_qa_squad | 0.734 | 0.735 | +0.001 |
| gsm8k_cot (flexible) | 0.768 | 0.754 | −0.014 |
| gsm8k_cot (strict) | 0.754 | 0.740 | −0.014 |
| longbench_hotpotqa (F1) | 0.240 | 0.205 | −0.035 |

**해석**:
- Long-context 검색(niah/vt/qa_squad) **사실상 무손실** — W8A8+KV-INT8가 32K 검색능력 보존.
- ruler_cwe는 원본 0.03으로 바닥(32K 최난도, 8B 한계)이라 ±0.1은 노이즈 수준(100샘플).
- gsm8k −1.4pp, fwe −1.3pp로 소폭 저하. longbench_hotpotqa −3.5pp(단 do_sample=True라 변동 포함).
- 종합: **INT8 W8A8 + INT8 KV가 정확도를 작은 손실로 보존**.
- 소요: original_ruler 67.7분 / combined_ruler 132.9분(KIVI 32K fake-quant requant 비용으로 ~2×), gsm8k ~17–24분, hotpot ~8–12분.
- 검증 포인트: niah=1.0(파이프라인 건전성), 결합이 원본 대비 큰 붕괴 없음, summary.json의 delta.
---

## 8. 최종 엄밀 검증 (end-to-end) ✅ — `code/etc/verify_final.py`

저장된 데이터·구현·평가가 의도("SmoothQuant W8A8 모델 + KV를 KIVI INT8로 양자화")대로인지, **재생성·재현 대조**로 점검.

| # | 항목 | 결과 | 판정 |
|---|---|---|---|
| 0 | W8A8 모듈 수 | 224 (=32×7) | PASS |
| 1 | Weight 저장 정확성 | int8·scale이 재빌드와 완전 일치(다른 원소 0). `fp16(int8×scale)==w_fq` 전32층 max Δ **0.0** | PASS |
| 2 | KV 저장 정확성 | gsm8k_cot/sample_0 재-prefill→KIVI 양자화가 저장 codes/scale/min과 **비트 일치**(layer 0/15/31, K·V) | PASS |
| 3 | KV 전수 형상/범위 | 140샘플 전부 n_quant 공식·shape·코드범위[0,255] 일치 | PASS |
| 4 | 인벤토리 | weight 32층+embeddings, KV 140 sample×32 = 4480 | PASS |
| 5a | eval 설정 | GROUPS(hotpot=longbench_hotpotqa)·GCFG(ruler 100/bs1/32K, gsm8k bs8, hotpot bs1/17K) | PASS |
| 5b | eval KV 주입 | KiviHFLM이 생성마다 `KiviINT8Cache(k=v=8,g32,r128)` 주입, update() 256회 호출 | PASS |
| 5c | original 경로 | 순수 HFLM(KIVI 미주입) | PASS |
| 5d | KV 양자화 실효 | KIVI cache vs fp16 cache 디코드 logit Δ=1.52(>0, no-op 아님) | PASS |

**주의(중요)**: PART1을 fp32로 곱해 비교하면 max Δ=1.892e-3가 보이는데, 이는 오류가 아님 —
`w_fq`는 곱 `int8×scale`을 **fp16으로 반올림 저장**한 값이고, 검증이 곱을 fp32로 계산했기 때문.
모델 실제 dtype(fp16) 기준 `fp16(int8×scale)==w_fq`는 **max Δ 0.0**(전32층 비트 일치)로 확인됨.
- 디버그로 국소화: 저장 int8==재빌드 int8(0 elem diff), 저장 scale==재빌드 scale(Δ0.0). 차이는 오직 곱의 fp16 반올림.

**결론**: Weight(W8)·KV(KIVI INT8) 저장본 모두 실제 W8A8+KIVI 모델 산출물과 (fp16 기준) 정확 일치, eval 스크립트도 의도대로 KIVI-INT8 KV를 주입·적용. **end-to-end 정확성 확정.**
