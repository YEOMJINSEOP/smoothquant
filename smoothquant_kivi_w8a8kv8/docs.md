# `compressed_data/` 데이터 포맷 명세 (검증본)

> 본 문서의 모든 dtype/shape/수식은 **실제 저장 파일(safetensors 헤더·바이트 회계) + unpack 왕복 복원**으로 검증되었습니다.
> 베이스 모델: `meta-llama/Llama-3.1-8B-Instruct` (hidden=4096, layers=32, heads=32, **KV heads=8(GQA)**, head_dim=128, intermediate=14336).

---

## 0. 공통 출처

두 데이터는 **동일한 W8A8 모델**에서 추출됩니다.

```
Llama-3.1-8B-Instruct (fp16)
  → smooth_lm(act_scales=llama-3.1-8b-instruct.pt, alpha=0.85)     # SmoothQuant 등가변환
  → quantize_model(weight=per_channel, act=per_token, quantize_bmm_input=False)
```

- `w8_of_w8a8_smoothquant_llama_31_8b/`  : 위 모델의 **가중치**(W8) 추출.
- `kv_kivi8_of_w8a8_smoothquant_llama_31_8b/` : 위 모델로 prefill한 **KV 캐시**를 KIVI INT8로 양자화.

---

## 1. Weight — `w8_of_w8a8_smoothquant_llama_31_8b/`

### 1.1 추출 방법 (`jsyeom/sq_kivi/save_w8a8_weights.py`)
W8A8Linear의 fake-quant 가중치 `w_fq = round(w/s)·s` (fp16)에서 정수·scale 역추출:

```
scale = clamp(|w_fq|.amax(dim=-1, keepdim=True), 1e-5) / 127     # 출력채널(row)별 1개
int8  = round(w_fq / scale)                ∈ [-127, 127]
─────────────────────────────────────────────────────────────
복원(dequant):  w ≈ int8 × scale
```

- **대칭(symmetric) 양자화**, q_max=127, **zero-point 없음**.
- 채널 최대 가중치가 정확히 ±127에 매핑 → 역추출 오차 **max 0.0** (평가 시 W8A8Linear 가중치와 비트 단위 일치).
- activation(A8, per-token)은 추론 중 동적 적용 → **저장하지 않음**.

### 1.2 형태 (검증된 dtype / shape / 의미)

**(a) `layer_<i>.safetensors` — 레이어당 1파일 (i = 0..31)**

| 키                                 | dtype | shape         | 의미                          |
| ---------------------------------- | ----- | ------------- | ----------------------------- |
| `self_attn.q_proj.weight`          | int8  | (4096, 4096)  | int8 가중치, 값 ∈ [-127,127]  |
| `self_attn.q_proj.scale`           | fp16  | (4096, 1)     | 출력채널(row)별 scale         |
| `self_attn.k_proj.weight`          | int8  | (1024, 4096)  | GQA: 8 KV헤드×128 = 1024행    |
| `self_attn.k_proj.scale`           | fp16  | (1024, 1)     | 출력채널별 scale              |
| `self_attn.v_proj.weight`          | int8  | (1024, 4096)  | GQA: 1024행                   |
| `self_attn.v_proj.scale`           | fp16  | (1024, 1)     | 출력채널별 scale              |
| `self_attn.o_proj.weight`          | int8  | (4096, 4096)  | int8 가중치, 값 ∈ [-127,127]  |
| `self_attn.o_proj.scale`           | fp16  | (4096, 1)     | 출력채널별 scale              |
| `mlp.gate_proj.weight`             | int8  | (14336, 4096) | int8 가중치, 값 ∈ [-127,127]  |
| `mlp.gate_proj.scale`              | fp16  | (14336, 1)    | 출력채널별 scale              |
| `mlp.up_proj.weight`               | int8  | (14336, 4096) | int8 가중치, 값 ∈ [-127,127]  |
| `mlp.up_proj.scale`                | fp16  | (14336, 1)    | 출력채널별 scale              |
| `mlp.down_proj.weight`             | int8  | (4096, 14336) | int8 가중치, 값 ∈ [-127,127]  |
| `mlp.down_proj.scale`              | fp16  | (4096, 1)     | 출력채널별 scale              |
| `input_layernorm.weight`           | fp16  | (4096,)       | 비양자화                      |
| `post_attention_layernorm.weight`  | fp16  | (4096,)       | 비양자화                      |

**(b) `embeddings.safetensors` — 전역 1파일**

| 키                          | dtype | shape           | 의미     |
| --------------------------- | ----- | --------------- | -------- |
| `model.embed_tokens.weight` | fp16  | (128256, 4096)  | 비양자화 |
| `lm_head.weight`            | fp16  | (128256, 4096)  | 비양자화 |
| `model.norm.weight`         | fp16  | (4096,)         | 비양자화 |

### 1.3 on-disk dtype 검증 (int8이 진짜 1바이트인지)
safetensors 헤더와 바이트 회계로 직접 확인:

| 검증 | 결과 |
|---|---|
| safetensors 원본 헤더 dtype | `weight = I8`, `scale = F16`, `layernorm = F16` |
| 바이트/원소 (q_proj.weight) | 16,777,216 원소 = 16,777,216 byte → **1.0 byte/원소 = int8 확정** (fp16이면 2.0) |
| 라이브러리 `get_dtype()` | `I8` |

→ fp16 위장이 아니라 **물리적 1바이트 int8**.

### 1.4 레이아웃
```
w8_of_w8a8_smoothquant_llama_31_8b/
├── layer_0.safetensors … layer_31.safetensors   # 각: 7 linear(weight int8 + scale fp16) + LN 2개
├── embeddings.safetensors                        # embed_tokens, lm_head, model.norm (fp16)
└── config.json                                   # base/alpha/quant방식/storage 메타
```
- 키 형식(레이어 파일): `self_attn.q_proj.weight`, `self_attn.q_proj.scale`, …, `input_layernorm.weight`.
- 총 텐서 515 = int8 224(=7×32) + fp16 291(scale 224 + LN 64 + embed/norm/lm_head 3). 총 ~8.5GB.

### 1.5 복원 예시
```python
from safetensors.torch import load_file
w = load_file("layer_0.safetensors")
real_q_proj = w["self_attn.q_proj.weight"].float() * w["self_attn.q_proj.scale"].float()  # (4096,4096) fp
```

---

## 2. KV cache — `kv_kivi8_of_w8a8_smoothquant_llama_31_8b/`

### 2.1 추출 방법 (`jsyeom/sq_kivi/dump_kv_cache.py` → KIVI `quant/new_pack.py`)
1. W8A8 모델로 각 프롬프트 prefill → `past_key_values.to_legacy_cache()` 로 **post-RoPE fp16 KV** 획득. 레이어당 `K,V` 형태 `[1, 8(KV헤드), T, 128]`.
2. 최근 `residual_length=128` 토큰을 제외한 앞부분만 양자화 (residual은 fp16이며 **저장 안 함**):
   - `n_quant_k = ⌊(T − 128)/32⌋ × 32`  (Key는 토큰축 그룹화 → 32 배수)
   - `n_quant_v = T − 128`               (Value는 head_dim축 그룹화 → T 제약 없음)
3. **비대칭(asymmetric) min-max 양자화** (group_size=32, bits=8):
```
scale = (mx − mn) / 255            # max_int = 2^8 − 1 = 255
code  = round((x − mn)/scale)      ∈ [0, 255]   (부호없는 8bit)
─────────────────────────────────────────────────────────────
복원(dequant):  x ≈ code × scale + mn      ← mn(zero-point) 필수
```
   - **Key = per-channel**: 토큰축을 32개씩 그룹화, 그룹 내 32 토큰에서 min/max → 채널마다 scale/min.
   - **Value = per-token**: head_dim축을 32개씩 그룹화, 그룹 내 32 채널에서 min/max → 토큰마다 scale/min.
4. 정수 코드를 int32에 **4개씩 패킹** (`32 bit / 8 bit = 4`).

### 2.2 형태 (검증된 dtype / shape / 의미) — 예: `ruler_vt/sample_0/layer_0.safetensors`, T=3833

`n_quant_k=3680 (=⌊(3833−128)/32⌋×32=115×32)`, `n_quant_v=3705 (=3833−128)`.

| 키        | dtype | shape (일반식)              | shape (예: T=3833)  | 의미                                       |
| --------- | ----- | -------------------------- | ------------------- | ------------------------------------------ |
| `k_code`  | int32 | (1, 8, n_quant_k/4, 128)   | (1, 8, 920, 128)    | Key 양자화 코드, 토큰축으로 4개 패킹       |
| `k_scale` | fp16  | (1, 8, n_quant_k/32, 1, 128) | (1, 8, 115, 1, 128) | per-channel: (토큰그룹, 1, 채널128)        |
| `k_min`   | fp16  | (1, 8, n_quant_k/32, 1, 128) | (1, 8, 115, 1, 128) | Key zero-point (min)                       |
| `v_code`  | int32 | (1, 8, n_quant_v, 128/4)   | (1, 8, 3705, 32)    | Value 양자화 코드, head_dim축으로 4개 패킹 |
| `v_scale` | fp16  | (1, 8, n_quant_v, 128/32, 1) | (1, 8, 3705, 4, 1)  | per-token: (토큰, 채널그룹4, 1)            |
| `v_min`   | fp16  | (1, 8, n_quant_v, 128/32, 1) | (1, 8, 3705, 4, 1)  | Value zero-point (min)                     |

- dim0=batch(1), dim1=**KV 헤드 8개**(GQA), head_dim=128.
- `_meta.json` (sample당 1개): `{T, n_quant_k, n_quant_v, group_size=32, residual_length=128, k_bits=8, v_bits=8}`.
- **fp16 residual(최근 128토큰 + Key의 부분그룹)은 저장하지 않음** — 사양상 packed INT8 부분만 보존.
- 스냅샷 시점: **prefill 직후, 생성 전**.

### 2.3 레이아웃
```
kv_kivi8_of_w8a8_smoothquant_llama_31_8b/
├── <task>/sample_<n>/layer_<i>.safetensors   # {k_code,k_scale,k_min,v_code,v_scale,v_min}
│                     /_meta.json
├── index.json
└── (task = niah_multikey_1, ruler_vt, ruler_cwe, ruler_fwe, ruler_qa_squad, gsm8k_cot, longbench_hotpotqa)
```
- 7 task × 20 sample × 32 layer = **4480 .safetensors**, 총 ~43GB.
- (각 layer 파일은 flat tensor dict이라 safetensors로 저장 — raw·안전·범용. `_meta.json`은 별도 유지.)
- RULER 5종은 4K 컨텍스트 프롬프트, `gsm8k_cot`(~700tok)·`longbench_hotpotqa`(~14K tok)는 자연 길이.

### 2.4 복원 예시
```python
import sys; sys.path.insert(0,"/SSD/JSY/KIVI")
from safetensors.torch import load_file
from quant.new_pack import unpack_and_dequant_kcache, unpack_and_dequant_vcache
d = load_file(".../ruler_vt/sample_0/layer_0.safetensors")
K = unpack_and_dequant_kcache(d["k_code"], d["k_scale"], d["k_min"], 32, 8)  # (1,8,n_quant_k,128) fp16
V = unpack_and_dequant_vcache(d["v_code"], d["v_scale"], d["v_min"], 32, 8)  # (1,8,n_quant_v,128) fp16
```
검증: unpack 정수 코드 ∈ [0,255], dequant K∈[−10.0,9.9]·V∈[−0.49,0.46] (실제 post-RoPE KV 스케일과 부합). 공식 확인 `dequant == code×scale+min` (fp16 오차 내 일치).

---

## 3. ⚠️ 핵심 구분 — Weight vs KV 양자화 방식이 다름

| 항목        | Weight (W8)                | KV (KIVI INT8)                          |
| ----------- | -------------------------- | --------------------------------------- |
| 양자화 방식 | 대칭 (symmetric)           | 비대칭 (asymmetric min-max)             |
| 정수 범위   | [-127, 127] (signed int8)  | [0, 255] (unsigned 8bit)                |
| 파라미터    | `scale` 만                 | `scale` + `min` (zero-point)            |
| 복원식      | `w = int8 × scale`         | `x = code × scale + min`                |
| 양자화 단위 | 출력채널(row)별 1 scale    | Key=채널·토큰그룹별 / Value=토큰·채널그룹별 |
| 디스크 dtype| int8 (1 byte/원소, raw)    | int32 packed (safetensors, unpack 필요)          |
| 미저장 항목 | A8 (추론 시 동적)          | fp16 residual (128토큰)                 |

> **주의**: KV는 `min`이 없으면 복원 불가하며, `k_code`/`v_code`는 패킹된 int32라 `unpack_*` 없이 직접 쓸 수 없음. Weight는 `scale`만으로, 추가 unpack 없이 `int8×scale`로 복원.

---

## 4. 설정 요약
- SmoothQuant: alpha=0.85, weight=per_channel·act=per_token INT8, `quantize_bmm_input=False`(옵션1 — attention BMM은 fp16, KV는 KIVI 전담).
- KIVI: k_bits=v_bits=8, group_size=32, residual_length=128, Key per-channel / Value per-token, **fake-quant**(KIVI 2/4bit 커널 미사용; 수식은 `new_pack.py`와 비트 일치).
