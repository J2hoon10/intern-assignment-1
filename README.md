# Qwen2.5-0.5B × PubMedQA — LoRA vs Full FT vs Base

PubMedQA(yes/no/maybe) 과제에서 **Qwen2.5-0.5B(base)** 를 세 가지 방식으로 비교하는 실험입니다.

1. **base** — 사전학습 모델 그대로 (zero-shot)
2. **full_ft** — 전체 파라미터 fine-tuning
3. **lora** — LoRA 어댑터만 fine-tuning

과제는 생성형(Causal LM)으로 정식화하고, 평가는 ` yes`/` no`/` maybe` 세 후보의 로그확률을 비교하는 **제약 스코어링(constrained scoring)** 으로 수행합니다. 지표는 **Accuracy & Macro-F1** 입니다.

---

## 1. 환경 설치 (conda, CUDA 12.1)

```powershell
conda env create -f environment.yml
conda activate lora_pubmedqa

# torch 는 cu121 인덱스에서 별도 설치 (다른 패키지와의 휠 혼선 방지)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성
pip install -r requirements.txt

# 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> bf16 지원 GPU(Ampere 이상)면 자동으로 bf16, 아니면 fp16으로 학습합니다. `bitsandbytes`/`flash-attn`은 사용하지 않으며 attention은 torch 내장 `sdpa`를 씁니다.

---

## 2. 실행

전체 파이프라인:

```powershell
./scripts/run_all.ps1
```

또는 단계별로:

```powershell
# 데이터 준비 (공식 test 500 다운로드 + 학습셋 구성 + 누수 검증)
python src/prepare_data.py --pqaa_size 30000

# 1) base zero-shot 평가
python src/evaluate.py --config configs/base.yaml

# 2) full fine-tuning + 평가
python src/train.py    --config configs/full_ft.yaml
python src/evaluate.py --config configs/full_ft.yaml

# 3) LoRA fine-tuning + 평가
python src/train.py    --config configs/lora.yaml
python src/evaluate.py --config configs/lora.yaml

# 학습 손실 곡선(png) 생성
python src/plot_logs.py --all
```

### 정성적 추론 확인 (입력 vs 예측 vs 정답)
`evaluate.py`는 500개 test 전체에 대한 집계 지표만 남기므로, 개별 사례를 눈으로 확인하려면 `src/infer.py`를 씁니다. `data/infer_sample.jsonl`에 test set에서 뽑은 소규모 샘플(yes/no/maybe 각 3개, 총 9개)을 준비해 두었습니다.

```powershell
python src/infer.py --config configs/lora.yaml
# --input 으로 다른 파일 지정 가능 (기본값: data/infer_sample.jsonl)
python src/infer.py --config configs/full_ft.yaml --input data/infer_sample.jsonl
```

- 콘솔/로그(`logs/<exp>_infer.log`)에 사례별로 질문, 정답(`true`), 예측(`pred`), 정오답 여부, yes/no/maybe 각 후보의 신뢰도(길이정규화 로그확률)를 출력합니다.
- 같은 내용을 `outputs/<exp>/inference_samples.json`에 저장합니다 (pubid, question, contexts, prompt, true_label, pred_label, correct, confidence).

### 학습 중단 & 재시작
학습은 `outputs/<exp>/checkpoint-*`에 자동 저장됩니다. 중단 후 같은 명령을 다시 실행하면 **최신 체크포인트에서 자동으로 이어서** 학습합니다. 처음부터 다시 하려면 `--fresh`를 붙입니다.

```powershell
python src/train.py --config configs/lora.yaml          # 자동 resume
python src/train.py --config configs/lora.yaml --fresh  # 처음부터
```

### 스모크 테스트 (파이프라인 빠른 점검)
```powershell
python src/train.py --config configs/lora.yaml --fresh --max_train_samples 100 --max_steps 20
```

---

## 3. 산출물

| 경로 | 내용 |
|---|---|
| `data/test.jsonl` | 공식 test 500 (question, contexts, label) |
| `data/train.jsonl`, `data/dev.jsonl` | 학습/검증 데이터 (test pubid 제외 검증됨) |
| `data/test_pubids.json`, `data/split_pubids.json` | 누수 검증용 pubid 감사 파일 |
| `outputs/<exp>/param_count.json` | 학습 파라미터 수 (total/trainable/%) |
| `outputs/<exp>/metrics.json` | Accuracy, Macro-F1, per-class F1, confusion matrix |
| `outputs/<exp>/predictions.json` | pubid→label (공식 `evaluation.py` 호환) |
| `outputs/<exp>/inference_samples.json` | 소규모 샘플의 입력/예측/정답/신뢰도 비교 (정성 확인용) |
| `outputs/<exp>/loss_curve.png` | 학습/검증 손실 곡선 |
| `outputs/summary.txt` / `summary.json` | 세 실험 비교 요약 (지표 + 파라미터 수) |
| `logs/*.log` | 데이터 준비 / 학습 / 평가 텍스트 로그 |

> 평가 지표와 파라미터 수는 **로그/JSON으로만** 기록합니다(차트 없음). 시각화는 손실 곡선만 생성합니다.

---

## 4. 데이터 & 누수 방지

- **PQA-L**: 전문가 라벨 1,000개(yes/no/maybe). 공식 test 500 + CV 500.
- **PQA-A**: 자동 생성 ~211k개(yes/no만, maybe 없음). 학습셋은 PQA-A subset + PQA-L CV 450을 혼합해 3클래스를 모두 학습합니다.
- **test 분포**: yes 276 / no 169 / maybe 55 (공식 `test_ground_truth.json` 기준, 총 500).
- **누수 방지(하드/fail-fast)**: 공식 test 500 pubid를 PQA-L·PQA-A 양쪽에서 제거하고, `train/dev ∩ test == 0`을 검증(위반 시 즉시 중단). 추가로 질문 텍스트 해시로 내용 중복까지 이중 확인합니다. 상세 로그는 `logs/prepare_data.log`.

---

## 5. 기대 결과 (정성)

- **base**: Accuracy가 majority(yes 276/500 ≈ 0.55) 근방, Macro-F1은 낮음(maybe/no를 잘 못 맞힘).
- **full_ft / lora**: Macro-F1이 크게 상승. LoRA는 학습 파라미터가 full FT 대비 수 %에 불과하지만 유사한 성능에 근접 → "적은 파라미터로 유사 성능"을 정량 확인.

## 6. (선택) 2단계 순차 학습 + PQA-L 10-fold CV

PQA-A로 먼저 학습한 뒤, 그 가중치를 이어받아 **PQA-L 10-fold 교차검증**으로 다시 학습하는 파이프라인입니다(PubMedQA 공식 프로토콜). 기존 단일 단계와 **공존**합니다.

```
Phase 1 (PQA-A, yes/no)
   └─▶ Phase 2: PQA-L CV 500을 stratified 10-fold(각 450 train / 50 val)
         → 각 fold를 Phase1 위에서 학습·검증 → val Macro-F1 최고 fold 선택 → 500 test 평가
```
- Phase 2 각 fold는 **Phase 1 가중치를 이어받아** 학습 (full_ft: 전체 모델 / lora: 동일 어댑터 계속 학습).
- **최고 fold 모델**을 `outputs/<exp>/{final|adapter}`로 복사 → 기존 evaluate.py가 그 모델로 test.
- 누수 보장: fold 내 train∩val=∅, (train∪val)∩test=∅ 를 assert(fail-fast). Phase1 dev는 PQA-A에서 별도 확보(PQA-L fold와 독립).
- 기본값: PQA-A 100k(+dev 500, 1 epoch) → PQA-L 10-fold(각 4 epoch). full_ft ~6.9시간 / lora ~2.5시간. (epoch은 두 방법 동일, LR만 상이)
- 산출물: `outputs/<exp>/phase2/cv_selection.json`(fold별 val Macro-F1 + 선택 fold), 각 fold `val_metrics.json`. full_ft 비선택 fold 가중치는 디스크 절약을 위해 자동 삭제.

```powershell
# 2단계용 분리 데이터는 새로 준비 필요 (data/mp/)
.\scripts\run_multiphase.ps1
```
개별 실행:
```powershell
python src/prepare_data_multiphase.py --pqaa_size 50000
python src/train_multiphase.py --config configs/full_ft_multiphase.yaml   # phase1(PQA-A) → phase2(PQA-L)
python src/evaluate.py         --config configs/full_ft_multiphase.yaml   # outputs/full_ft_mp/*
python src/train_multiphase.py --config configs/lora_multiphase.yaml
python src/evaluate.py         --config configs/lora_multiphase.yaml
```
- 산출물: `outputs/full_ft_mp/`, `outputs/lora_mp/`(및 `phase1/`,`phase2/` 하위), 최종 지표는 `outputs/summary.txt`에 `full_ft_mp`/`lora_mp`로 함께 표시.
- 단계 중단 시 같은 명령을 다시 실행하면 **완료된 단계는 건너뛰고, 중단된 단계는 체크포인트부터 재개**합니다.
- ⚠️ 단일 단계 실험이 GPU를 쓰는 동안 **동시에 돌리지 마세요.**

## 7. 파일 구조

```
LoRA_exp/
├── environment.yml / requirements.txt / README.md
├── configs/   base / full_ft / lora / lora_smoke / full_ft_multiphase / lora_multiphase (.yaml)
├── src/       common / data / model_utils / metrics / plot_logs
│              prepare_data / train / evaluate                (단일 단계)
│              prepare_data_multiphase / train_multiphase     (2단계 순차)
│              infer                                          (소규모 정성적 추론 확인)
├── scripts/   run_all.ps1 / run_resume.ps1 / run_multiphase.ps1
├── data/ outputs/ logs/  (실행 시 생성; 2단계 데이터는 data/mp/)
```
