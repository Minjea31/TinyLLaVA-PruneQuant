# 실제 INT4 곱셈 W4A4 평가

모든 가중치는 기존 INT4 RTN 체크포인트의 정수 값과 scale을 그대로 사용한다.
313개 Linear와 SigLIP patch Conv2d 1개는 INT4×INT4 Tensor Core 곱셈·INT32 누산을 수행한다.
activation은 각 입력 행 안에서 원래 weight scale 경계에 맞춘 K 그룹마다 동적으로 INT4 양자화한다.
Attention의 QK/PV 행렬곱·Softmax·Norm·RoPE·embedding은 기존 정밀도를 유지한다.
모델 전체의 모든 연산을 INT4로 수행한 결과가 아니다. 파인튜닝은 없다.

| 평가 | FP16 | 기존 W4A16 | 실제 W4A4 Linear+Conv |
| --- | ---: | ---: | ---: |
| POPE macro F1 | 86.93% | 60.13% | 54.50% |
| ScienceQA accuracy | 59.20% | 51.07% | 41.30% |

| 평가·모드 | peak allocated GiB | 문항당 처리 시간 ms | 처리량 문항/초 |
| --- | ---: | ---: | ---: |
| pope / FP16 | 4.1360 | 230.60 | 4.34 |
| pope / W4A16 | 2.0724 | 308.09 | 3.25 |
| pope / W4A4 Linear+Conv | 2.1384 | 545.51 | 1.83 |
| scienceqa / FP16 | 4.9987 | 183.00 | 5.46 |
| scienceqa / W4A16 | 2.9346 | 185.71 | 5.38 |
| scienceqa / W4A4 Linear+Conv | 2.9956 | 538.78 | 1.86 |

공통 조건은 batch 8, 같은 질문·이미지·greedy·max_new_tokens 32·warmup 1이다.
문항당 시간은 batch generation 시간을 실제 batch 크기로 나눈 값이며 개별 요청 latency가 아니다.
이 커널은 정수 정확성을 먼저 검증한 구현이며 속도 최적화는 완료하지 않았다.
타일 padding·activation quantize/pack·group scale·bias 비용도 포함한다.
원자료: `results/int4_all_w4a4/comparison.json`.
