# 다른(빠른 네트워크) 서버에서 compressed_data 재생성 → HF 업로드

이 서버는 외부망이 느려 54GB 직접 업로드가 비현실적이다. 대신 **코드만 git push**하고,
**외부망이 빠른 GPU 서버에서 compressed_data를 재생성한 뒤 거기서 HF에 업로드**한다.
재생성은 결정적이며(코드+시드+버전 동일 → 산출물 동일, `code/etc/verify_final.py`로 본 서버에서 비트일치 검증 완료),
따라서 업로드되는 데이터 = 본 서버에서 검증한 데이터와 동치다.

---

## 0. 빠른 서버로 옮길 것 (작은 것만)

| 항목 | 크기 | 방법 |
|---|---|---|
| 코드 (이 repo + KIVI) | KB | git push / clone |
| `act_scales/llama-3.1-8b-instruct.pt` | 5MB | **NAS 경유 or scp** (gitignore라 push 안 됨). 또는 아래 1-b로 재생성 |
| 베이스 모델 `meta-llama/Llama-3.1-8B-Instruct` | 16GB | 빠른 서버에서 HF 다운로드 |
| lm_eval fork (RULER 태스크) | — | `-ASSIGN-mask_lm_eval` git/NAS, 또는 동등 RULER 지원 lm_eval |

> `compressed_data`(51GB), `dataset/`, `models--*`, `*.pt`는 모두 gitignore → push 제외됨.

---

## 1. 환경 (버전 핀 — 재현성의 핵심)

```bash
conda create -n sqkivi python=3.10 -y && conda activate sqkivi
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.1 flash-attn==2.8.3 datasets==3.6.0 accelerate zstandard
pip install wonderwords nltk jieba fuzzywuzzy rouge python-Levenshtein   # RULER + longbench
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
# lm_eval (커스텀 fork, RULER 내장) editable 설치
pip install -e /path/to/-ASSIGN-mask_lm_eval/src/lm-eval     # == lm_eval 0.4.9.1
# smoothquant repo editable (또는 PYTHONPATH)
pip install -e /path/to/smoothquant   # 또는 export PYTHONPATH=/path/to/smoothquant
```
- 핵심 버전: torch 2.8.0+cu128, **transformers 4.56.1**, flash_attn 2.8.3, lm_eval 0.4.9.1(fork), datasets 3.6.0.
- 경로: 스크립트가 `/SSD/JSY/smoothquant`, `/SSD/JSY/KIVI`를 하드코딩 → 빠른 서버 경로에 맞게 `sys.path`/상수 수정하거나 동일 경로에 배치.
- `HF_HUB_CACHE=<model_dir>` 지정.

---

## 2. 재생성 명령 순서 (Llama-3.1-8B-Instruct, α=0.85, KIVI k=v=8/g32/r128)

```bash
cd <smoothquant_repo>
export HF_HUB_CACHE=<dir> HF_DATASETS_CACHE=<dir> PYTHONPATH=<smoothquant_repo>

# 1-a) act_scales 가져오기(NAS/scp)  또는  1-b) 재캘리브(결정적, Pile-val 512×512 seed42)
python examples/generate_act_scales.py --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output-path act_scales/llama-3.1-8b-instruct.pt \
  --dataset-path dataset/val.jsonl.zst --num-samples 512 --seq-len 512
#   (Pile-val: huggingface_hub hf_hub_download mit-han-lab/pile-val-backup val.jsonl.zst)

# 2) Task1 — W8A8 weight 저장 (int8+scale, layer별)
python code/save_w8a8_weights.py --alpha 0.85 --out compressed_data/w8_of_w8a8_smoothquant_llama_31_8b
python code/split_weights_per_layer.py --dir compressed_data/w8_of_w8a8_smoothquant_llama_31_8b --remove-monolith

# 3) Task2 — KV 덤프 (W8A8 모델 prefill → KIVI-INT8, per-layer)
python code/dump_kv_cache.py --limit 20 --out compressed_data/kv_kivi8_of_w8a8_smoothquant_llama_31_8b
python code/etc/convert_kv_to_safetensors.py --root compressed_data/kv_kivi8_of_w8a8_smoothquant_llama_31_8b

# (선택) 재생성 검증
python code/etc/verify_final.py

# 4) README(영문 카드)·docs는 git에 포함되어 있으니 compressed_data/에 복사
cp <repo>/smoothquant_kivi_w8a8kv8/compressed_data/README.md compressed_data/   # (data card)
```

---

## 3. HF 업로드 (빠른 서버에서)

```bash
huggingface-cli login   # write 토큰
python code/etc/upload_to_hf.py \
  --repo-id jsyeom/smoothquant-kivi-w8a8kv8 \
  --path <...>/compressed_data
# 파일 4656개라 가능하면 tar 샤딩 권장(많은 작은 파일 = 업로드 502/오버헤드). 또는 HF_HUB_ENABLE_HF_TRANSFER=1.
```

---

## 4. 결정성 주의
- 동일 코드+버전+시드면 W8A8 weight·KIVI KV는 재현됨(본 서버 `verify_final.py` 비트일치 확인). 단 **GPU 종류/torch 버전이 다르면 fp16 라운딩으로 ≤1 LSB 차이** 가능(정확성엔 무의미, 여전히 유효한 W8A8+KIVI 산출물).
- 엄밀히 "본 서버와 비트동일"을 원하면 같은 GPU(L40S)+동일 버전 사용.
- 빠른 서버에서 `verify_final.py` 재실행으로 산출물 정합성 재확인 가능.
