# KL 기준 weight·activation FP16 유지 — 전체 평가

설정: both. Weight KL ≥ 0.001, activation KL ≥ 0.002를 두 선정 집합에서 모두 만족한 대상을 사용했다.
FP16 weight 레이어 7개, A16 Linear 입력 3개. 학습·LoRA·optimizer update는 없다.
나머지 W4A4 Linear·Conv는 실제 INT4×INT4→INT32를 사용한다. W16A4·W4A16·W16A16 예외는 FP16 GEMM 경로다.
Attention QK/PV·Softmax·RoPE 등 비대상 연산은 기존 정밀도를 유지한다.

| 평가 | FP16 | W4A16 | W4A4 | Mixed |
| --- | ---: | ---: | ---: | ---: |
| POPE macro F1 | 86.93% | 60.13% | 54.50% | 58.82% |
| ScienceQA accuracy | 59.20% | 51.07% | 41.30% | 45.56% |

| 평가·설정 | peak allocated GiB | 문항당 처리 시간 ms | 처리량 문항/초 |
| --- | ---: | ---: | ---: |
| pope / FP16 | 4.1360 | 230.60 | 4.34 |
| pope / W4A16 | 2.0724 | 308.09 | 3.25 |
| pope / W4A4 | 2.1384 | 545.51 | 1.83 |
| pope / Mixed | 2.2428 | 560.70 | 1.78 |
| scienceqa / FP16 | 4.9987 | 183.00 | 5.46 |
| scienceqa / W4A16 | 2.9346 | 185.71 | 5.38 |
| scienceqa / W4A4 | 2.9956 | 538.78 | 1.86 |
| scienceqa / Mixed | 3.1075 | 539.77 | 1.85 |

공통 조건: batch 8, left padding, v1 prompt, greedy, seed 2026, max_new_tokens 32, warmup 1.
문항당 시간은 batch 생성 시간을 실제 batch 크기로 나눈 값이며 개별 요청 latency가 아니다.
Peak allocated는 CUDA tensor 할당량이며 프로세스 전체 메모리와 구분한다. 각 설정은 단일 실행 측정이다.

## 실제 정밀도 구성

| 계산 정밀도 | Linear·Conv 개수 |
| --- | ---: |
| W4A4 | 310 |
| W16A4 | 1 |
| W4A16 | 1 |
| W16A16 | 2 |

FP16 weight는 공개 원본 safetensors를 FP16으로 변환한 값의 hash와 직접 일치했다.
같은 INT4 checkpoint·입력·생성 조건, 예측 재채점, 레이어 호출과 native 명령을 검증했다.
독립 calibration 비교: [보고서](../calibration/report.md). Weight-only·activation-only는 해당 calibration에서 비교했다.
결합 설정의 전체 평가 결과를 기록했으며, threshold를 전체 평가 점수에 맞추어 변경하지 않았다.
