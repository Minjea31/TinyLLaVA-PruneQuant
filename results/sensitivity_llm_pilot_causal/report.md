# LLM 레이어별 양자화 민감도

8개 calibration 예제, 9개 단일 대상 개입을 완료했다.
집합 구성: {'confirmation': 4, 'discovery': 4}. 양자화 대상 이외의 Vision·Projector·LLM은 FP16으로 고정했다.
각 개입은 하나의 leaf 또는 하나의 블록 안 Attention/FFN 묶음만 변경한다. 전체 INT4 모델에서 복구한 실험은 아니다.
같은 정답 문맥에서 assistant 토큰만 평가한다. 이미지·사용자 prompt·padding 위치는 점수에서 제외한다.
KL은 FP16 다음 토큰 분포와 비교한 값이며, NLL 변화는 정답 토큰에 대한 손실 차이다. 예제별 평균을 다시 평균한다.
이 점수는 POPE F1·ScienceQA accuracy가 아니며, 최종 벤치마크 개선을 보장하지 않는다.
W/A를 분리하기 위해 같은 FP16 연산에서 INT4 양자화·복원을 시뮬레이션했다. 이 실행의 시간·메모리는 속도 비교에 사용하지 않는다.
A4는 Linear 입력에만 적용한다. RMSNorm·embedding의 A4 점수는 정의하지 않았다.
가중치는 기존 group 128 RTN 체크포인트를 읽었으며 학습·LoRA·optimizer update는 없다.
FP16 파라미터 복구와 개입 전후 FP16 출력 일치를 검증했다.

## 개별 레이어 순위

순위는 discovery 집합의 KL로 정하고, 별도 confirmation 집합의 값을 나란히 표시한다.

### W4A16 — 상위 5개

두 집합의 top-5 공통 레이어는 5개다.

| 레이어 | Discovery KL | Confirmation KL | 전체 ΔNLL | 파라미터 M |
| --- | ---: | ---: | ---: | ---: |
| `lm_head` | 0.00809338 | 0.00515579 | +0.0103398 | 65.536 |
| `model.norm` | 0.00182096 | 0.00145068 | +0.00312901 | 0.002 |
| `model.embed_tokens` | 0.000769339 | 0.000148393 | -0.00598053 | 65.536 |
| `model.layers.0.input_layernorm` | 0.000288821 | 5.81344e-05 | -0.00176011 | 0.002 |
| `model.layers.0.self_attn.q_proj` | 5.92077e-06 | 3.38144e-06 | -0.000567371 | 4.194 |

### W16A4 — 상위 2개

두 집합의 top-2 공통 레이어는 2개다.

| 레이어 | Discovery KL | Confirmation KL | 전체 ΔNLL | 파라미터 M |
| --- | ---: | ---: | ---: | ---: |
| `lm_head` | 0.0259772 | 0.014621 | +0.00336442 | 65.536 |
| `model.layers.0.self_attn.q_proj` | 3.09717e-05 | 9.21714e-06 | -0.00106958 | 4.194 |

### W4A4 — 상위 2개

두 집합의 top-2 공통 레이어는 2개다.

| 레이어 | Discovery KL | Confirmation KL | 전체 ΔNLL | 파라미터 M |
| --- | ---: | ---: | ---: | ---: |
| `lm_head` | 0.0343535 | 0.0173415 | +0.0149139 | 65.536 |
| `model.layers.0.self_attn.q_proj` | 3.75637e-05 | 1.08546e-05 | -0.00121192 | 4.194 |

## 블록 안 Attention과 FFN 묶음

각 블록의 Attention projection 전체 또는 FFN projection 전체를 함께 바꾼 경우다.
FFN과 Attention은 파라미터 수가 다르므로 아래 원시 KL은 같은 저장 비용의 비교가 아니다.

| 모드 | 묶음 | 블록 수 | KL 중앙값 | ΔNLL 중앙값 | 블록당 파라미터 M |
| --- | --- | ---: | ---: | ---: | ---: |

## 블록 위치별 비교

개별 projection의 KL 중앙값이다. 앞(0–6), 중간(7–14), 뒤(15–21)를 비교하며 Norm·embedding·head는 제외한다.

| 모드 | 위치 | 개별 projection 수 | KL 중앙값 |
| --- | --- | ---: | ---: |
| W4A16 | 앞 | 1 | 4.65111e-06 |
| W16A4 | 앞 | 1 | 2.00944e-05 |
| W4A4 | 앞 | 1 | 2.42092e-05 |

## 실제 INT4 커널 확인

Discovery W4A4 상위 leaf를 기존 native 커널로 교체해 같은 조건에서 재측정했다. 나머지 모델은 FP16이다.

| 레이어 | QDQ 전체 KL | Native 전체 KL |
| --- | ---: | ---: |
| `lm_head` | 0.0258475 | 0.0258473 |
| `model.layers.0.self_attn.q_proj` | 2.42092e-05 | 2.12526e-05 |

## Heatmap

색은 최종 출력 KL이다. 큰 값일수록 이번 조건에서 출력 변화가 크다. 회색은 해당 조건을 적용하지 않은 칸이다.

![개별 레이어 민감도](layers.png)

![Attention·FFN 묶음 민감도](groups.png)

원자료: `run.json`, `baseline.json`, `measurements.jsonl`, `ranking.csv`, `native_checks.json`, `summary.json`.
집합 간 순위가 안정적인 대상을 후속 calibration 개선 후보로 삼고, 최종 설정은 별도 전체 평가로 확인한다.

## 실행 경로 검증

오른쪽 padding을 사용하고 유효한 정답 위치만 평가하므로, 미래의 padding을 명시적으로 가리는 mask를 생략해 causal SDPA 경로를 사용했다. 모든 개입과 FP16 기준에 같은 경로를 적용했다.
공개 masked forward와 비교한 첫 8개 예제의 평균 KL 차이는 3.43731e-06였다. 어휘 head에 입력하는 행 수도 달라질 수 있어 공개 forward와의 bit 일치를 주장하지 않는다.
재사용한 FP16 prefix와 head 입력은 첫 배치에서 재계산 결과와 bit 단위로 일치했다. 전체 8개 예제의 개입 전후 FP16 출력은 정확히 일치했고 파라미터 hash도 복구됐다.
이 분석은 FP16 나머지 모델에서 단일 대상을 바꾼 민감도이며, 전체 INT4 모델에서 한 레이어를 복구한 성능 변화와는 구분한다.

## Projection별 집계

22개 블록의 개별 레이어를 집계했다. 평균과 중앙값 차이가 크면 일부 블록의 큰 영향이 평균을 끌어올린 경우다.

| 모드 | Projection | KL 중앙값 | KL 평균 | 최대 KL 블록 |
| --- | --- | ---: | ---: | ---: |
| W4A16 | input_layernorm | 0.000173478 | 0.000173478 | 0 |
| W4A16 | q_proj | 4.65111e-06 | 4.65111e-06 | 0 |
| W16A4 | q_proj | 2.00944e-05 | 2.00944e-05 | 0 |
| W4A4 | q_proj | 2.42092e-05 | 2.42092e-05 | 0 |

## Attention·FFN의 예제별 대응 비교

같은 예제에서 22개 블록의 Attention 묶음 KL 평균과 FFN 묶음 KL 평균을 비교했다.
FFN−Attention 값이 양수면 이번 개입에서 FFN의 출력 변화가 더 컸다는 뜻이다. 파라미터 수·저장 비용을 맞춘 비교는 아니다.
95% 구간은 예제를 5,000번 복원 추출한 paired bootstrap이다. 고정된 모델·calibration 분포에 대한 변동이며 전체 과제 정확도의 신뢰구간이 아니다.

| 모드 | 집합 | Attention 평균 KL | FFN 평균 KL | FFN−Attention | 95% 구간 |
| --- | --- | ---: | ---: | ---: | --- |
