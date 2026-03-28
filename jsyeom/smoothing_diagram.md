# Smoothing in Residual Stream View (Single Decoder Layer)

```
 Residual Stream
 ════════╤═══════════════════════════════════════════════════╤════════
         │                                                   │
         │  ┌─────────────────────────────────────────────┐  │
         │  │            ATTENTION BLOCK                  │  │
         │  │                                             │  │
    READ │  │  input_layernorm (γ / s_attn)   ← ÷ s_attn  │  │
         │  │         │  (activation divided by s_attn)   │  │
         │  │         │                                   │  │
         │  │  ┌──────┼──────┐                            │  │
         │  │  │      │      │                            │  │
         │  │  q_proj k_proj v_proj  (W × s_attn) ← × s   │  │
         │  │  │      │      │          (보상)             │  │
         │  │  └──────┼──────┘                            │  │
         │  │         │                                   │  │
         │  │     attention                               │  │
         │  │         │                                   │  │
         │  │      o_proj  (변경 없음)                      │  │
         │  │         │                                   │  │
         │  └─────────┼───────────────────────────────────┘  │
         │            │  WRITE                               │
 ════════╪════════════╪══════════════════════════════════════╪════════
  (+)  ◄─┘            └─► (+) Residual Stream                │
 ════════╤═══════════════════════════════════════════════════╪════════
         │                                                   │
         │  ┌─────────────────────────────────────────────┐  │
         │  │            FFN BLOCK                        │  │
         │  │                                             │  │
    READ │  │  post_attn_layernorm (γ / s_ffn) ← ÷ s_ffn  │  │
         │  │         │  (activation divided by s_ffn)    │  │
         │  │         │                                   │  │
         │  │    ┌────┴────┐                              │  │
         │  │    │         │                              │  │
         │  │  gate_proj up_proj  (W × s_ffn)  ← × s      │  │
         │  │    │         │         (보상)                │  │
         │  │    └────┬────┘                              │  │
         │  │      SiLU & mul                             │  │
         │  │         │                                   │  │
         │  │     down_proj  (변경 없음)                    │  │
         │  │         │                                   │  │
         │  └─────────┼───────────────────────────────────┘  │
         │            │  WRITE                               │
 ════════╪════════════╪══════════════════════════════════════╪════════
  (+)  ◄─┘            └─► (+) Residual Stream
 ═══════════════════════════════════════════════════════════════════
```

## 핵심 관찰

Smoothing은 **READ 지점**에만 적용된다.

```
  Residual Stream  ──READ──►  RMSNorm(÷s)  →  Linear(×s)  →  ...  ──WRITE──►  Residual Stream
                              ^^^^^^^^^^^     ^^^^^^^^^^^
                              smoothing 적용   smoothing 적용         변경 없음
```

- **READ** : RMSNorm → Linear 사이에서 activation을 s로 축소하고, weight를 s로 확대
- **WRITE** : o_proj, down_proj는 변경 없음 → residual stream에 쓰는 값은 원래와 동일
- **Residual Stream** : 변경 없음 → 다음 레이어로 전달되는 정보가 동일

즉, smoothing은 각 서브블록이 residual stream에서 정보를 **읽어오는 인터페이스**만 조정한다.
서브블록이 residual stream에 **쓰는 값**은 수학적으로 동일하므로, 전체 모델 출력이 보존된다.
