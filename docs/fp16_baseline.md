# TinyLLaVA-1.5B FP16 기준 평가 — batch 8

2026-10-28 FP16 / batch 8의 전체 기준 평가를 완료했다.
POPE 9,000문항과 ScienceQA image-test 2,017문항, 총 11,017개 예측을 다시 채점했다.
누락·중복 ID 없이 원본 질문·이미지·정답·입력 hash·batch 조건·시간 metadata와 저장된 summary가 일치했다.
공개 체크포인트의 파인튜닝 전 평가이며, 학습 업데이트는 수행하지 않았다.
이전 batch 1 정식 결과는 사용자 요청대로 삭제한 상태다.

## 전체 평가 결과

| 평가 | 문항 수 | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| POPE random | 3,000 | 90.47 | 92.27 | 88.33 | 90.26 |
| POPE popular | 3,000 | 86.60 | 85.42 | 88.27 | 86.82 |
| POPE adversarial | 3,000 | 82.83 | 79.62 | 88.27 | 83.72 |
| ScienceQA image-test | 2,017 | 59.20 | — | — | — |

POPE macro F1: **86.93217232%** (세 category의 F1 평균).
POPE pooled micro F1: 86.85102197%, 전체 accuracy: 86.63333333% (7,797/9,000).
ScienceQA: 1,194/2,017 정답, 선택지 파싱 실패 0건.

## 메모리와 처리 시간

| 평가 | Peak allocated GiB | Peak reserved GiB | 문항당 처리 시간 ms | 처리량 문항/초 | generation 합계 초 |
| --- | ---: | ---: | ---: | ---: | ---: |
| POPE | 4.1360 | 6.1621 | 230.60 | 4.34 | 2075.36 |
| ScienceQA | 4.9987 | 6.9512 | 183.00 | 5.46 | 369.12 |

GiB는 bytes / 2^30이다. allocated는 CUDA tensor 할당량, reserved는 PyTorch allocator 확보량이다.
문항당 처리 시간은 batch 생성 시간을 실제 batch 크기로 나눈 값이며 개별 요청 latency가 아니다.
처리량과 generation 합계에는 Vision·Projector·LLM 연산을 포함하고 전처리·로드·warmup은 제외한다.
POPE는 1,125개 batch, ScienceQA는 253개 batch(마지막 1문항 포함)를 실행했다.
중복 입력도 각각 생성했으며 재사용한 답변은 없다.

완료 시각(KST): POPE 2026-10-28 14:50:31, ScienceQA 2026-10-28 14:57:11.
질문·답변 길이가 다른 평가끼리 평균 시간을 같은 작업의 속도 차이로 해석하지 않는다.

원자료 검증: `results/baseline/verification.json`.

## 재현

```bash
bash scripts/eval_fp16_all.sh
```

POPE 9,000문항 → ScienceQA image-test 2,017문항을 순서대로 실행한다.
실행 중에는 임시 sleep inhibitor로 절전을 막는다. 같은 run 조건에서는 중단 후 이어받는다.

개별 실행:

```bash
source scripts/env.sh
python scripts/test_inference.py --precision fp16 --benchmark pope --batch-size 8
python scripts/test_inference.py --precision fp16 --benchmark scienceqa --batch-size 8
```

공식 legacy 모델·v1 prompt·greedy·seed 2026·max_new_tokens 32·warmup 1을 사용한다.
길이가 다른 prompt는 left padding과 attention mask로 처리한다.
마지막 batch도 버리지 않고 남은 문항 수만큼 실행한다. 중복 입력은 각각 생성한다.
ScienceQA는 로컬 CQM letter-only image-test 조건이다.

결과 경로:

- `results/baseline/pope/`: predictions.jsonl, run.json, summary.json.
- `results/baseline/scienceqa/`: predictions.jsonl, run.json, summary.json.
- `logs/baseline_batch8/`: 각 benchmark의 실행 로그.

문항별 seconds는 batch generation 시간 / 실제 batch 크기다. 개별 요청 latency가 아니다.
batch_seconds·batch_size·batch_id를 함께 기록한다. summary에는 generation 시간 합계,
처리량(fresh_samples_per_second), CUDA peak allocated/reserved memory를 저장한다.
이후 INT4도 같은 batch 8·입력·generation 조건으로 비교한다.

배치별 별도 메모리 실측은 [배치 메모리 기록](batch_memory.md)을 참고한다.

## 실행 전 GPU 검증

POPE와 ScienceQA 각각 16문항의 batch 8 실행을 통과했다.
ScienceQA prompt가 가장 긴 8문항도 batch 8로 통과했다. 최대 확장 입력 길이는 1302이고,
CUDA peak allocated는 5.00 GiB, peak reserved는 6.06 GiB였다.
원자료는 `results/batch8_validation/`에 저장한다. 정식 전체 점수와 구분한다.
