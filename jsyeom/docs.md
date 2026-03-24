# SmoothQuant Smoothing 동작 원리

> Llama-2-13B 기준으로 설명합니다.
> 코드 참조: `smoothquant/smooth.py`


## 1단계: 레이어 순회

> `smooth.py:75-76`

```python
for name, module in model.named_modules():
```

모델의 모든 모듈을 순회하면서 `LlamaDecoderLayer`를 찾습니다.
Llama-2-13B는 **40개의 디코더 레이어**가 있으므로,
각 레이어마다 아래 과정이 **2번씩** (Attention, FFN) 수행됩니다.


---


## 2단계: 각 레이어에서 Smoothing 대상 추출

> `smooth.py:126-141`

각 `LlamaDecoderLayer`에서 두 쌍을 추출합니다.

### 쌍 A — Attention 블록

```
input_layernorm (RMSNorm)  →  [q_proj, k_proj, v_proj] (Linear)
```

- `act_scales` 키: `"model.layers.{i}.self_attn.q_proj"`
- 의미: q_proj 입력의 채널별 최대 절대값

### 쌍 B — FFN 블록

```
post_attention_layernorm (RMSNorm)  →  [gate_proj, up_proj] (Linear)
```

- `act_scales` 키: `"model.layers.{i}.mlp.gate_proj"`
- 의미: gate_proj 입력의 채널별 최대 절대값

이 두 쌍 각각에 대해 `smooth_ln_fcs_llama_like()`가 호출됩니다.


---


## 3단계: `smooth_ln_fcs_llama_like` 내부 동작

> `smooth.py:48-71`
> Attention 쌍을 예시로 설명합니다. `hidden_dim = 5120` (Llama-2-13B).


### 3-1. act_scales — 캘리브레이션에서 수집된 activation 통계

```python
act_scales  # shape: [5120]
```

각 채널(hidden dim)에 대해, 캘리브레이션 데이터(512개 문장)
전체에서 관찰된 **최대 절대값**입니다.

- 채널 `j`의 값이 크다 = 해당 채널에 outlier가 있다는 뜻


### 3-2. weight_scales 계산

> `smooth.py:58-61`

```python
weight_scales = torch.cat(
    [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
)
weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
```

처리 과정:

```
fc.weight shape         : [out_features, in_features]
.abs().max(dim=0)       : 각 입력 채널(column)의 최대 절대값  → [5120]
q/k/v 3개를 concat      : [3, 5120]
.max(dim=0)             : 3개 중 채널별 최대값               → [5120]
```

결과: `weight_scales[j]` = 채널 `j`에 연결된 weight들 중 가장 큰 절대값


### 3-3. Smoothing scales 계산

> `smooth.py:62-67`

```python
scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
```

핵심 공식:

```
s_j = (a_j ^ alpha) / (w_j ^ (1 - alpha))
```

각 변수의 의미:

```
alpha = 0.85    (Llama-2-13B 최적값)
a_j             채널 j의 activation 최대 절대값
w_j             채널 j의 weight 최대 절대값
s_j             채널 j의 smoothing scale
```

alpha의 의미:

- alpha가 높을수록 → `s_j`가 커짐 → activation을 더 많이 축소, weight를 더 많이 확대
- alpha = 0.85이면 양자화 난이도의 85%를 activation에서 weight로 이전


### 3-4. In-place 변환 적용

> `smooth.py:69-71`

```python
ln.weight.div_(scales)                  # RMSNorm weight  ←  ÷ s
for fc in fcs:
    fc.weight.mul_(scales.view(1, -1))  # Linear  weight  ←  × s
```

**RMSNorm 변환:**

```
원래:     Y  = RMSNorm(X) = X / RMS(X) * gamma
변환 후:  Y' = X / RMS(X) * (gamma / s)
```

gamma를 s로 나누면, RMSNorm의 출력이 채널별로 `1/s`만큼 스케일됩니다.

**Linear 변환:**

```
원래:     Z  = W @ Y
변환 후:  Z' = (W * s) @ Y'
            = (W * s) @ (Y / s)
            = W @ Y
            = Z
```

W에 s를 곱해서, RMSNorm에서 `1/s`로 줄어든 것을 정확히 보상합니다.

**결과: Z' = Z — 출력이 수학적으로 동일합니다.**


---


## 전체 요약

Llama-2-13B의 각 디코더 레이어(40개)에서 변경되는 파라미터:

```
+-----------------------------------------+-----------+-----------------+
| 변환 위치                                | Before    | After           |
+-----------------------------------------+-----------+-----------------+
| input_layernorm.weight                  | gamma     | gamma / s_attn  |
| q_proj.weight                           | W_q       | W_q   * s_attn  |
| k_proj.weight                           | W_k       | W_k   * s_attn  |
| v_proj.weight                           | W_v       | W_v   * s_attn  |
+-----------------------------------------+-----------+-----------------+
| post_attention_layernorm.weight         | gamma     | gamma / s_ffn   |
| gate_proj.weight                        | W_gate    | W_gate * s_ffn  |
| up_proj.weight                          | W_up      | W_up   * s_ffn  |
+-----------------------------------------+-----------+-----------------+
```

정리:

- **변경되는 파라미터** : 레이어당 7개 x 40레이어 = 280개 텐서
- **변경되지 않는 파라미터** : o_proj, down_proj, lm_head, embedding 등
- **모델 출력** : 완전히 동일 (등가 변환)
- **효과** : activation의 outlier 채널이 축소되어 이후 INT8 양자화 시 정보 손실 최소화
