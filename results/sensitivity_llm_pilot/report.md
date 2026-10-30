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
| `lm_head` | 0.00809738 | 0.00512483 | +0.0105013 | 65.536 |
| `model.norm` | 0.00180785 | 0.00144468 | +0.00309852 | 0.002 |
| `model.embed_tokens` | 0.000742504 | 0.000147729 | -0.00584799 | 65.536 |
| `model.layers.0.input_layernorm` | 0.000279041 | 5.17187e-05 | -0.00126481 | 0.002 |
| `model.layers.0.self_attn.q_proj` | 5.98427e-06 | 4.28176e-06 | -6.30175e-05 | 4.194 |

### W16A4 — 상위 2개

두 집합의 top-2 공통 레이어는 2개다.

| 레이어 | Discovery KL | Confirmation KL | 전체 ΔNLL | 파라미터 M |
| --- | ---: | ---: | ---: | ---: |
| `lm_head` | 0.0258194 | 0.014708 | +0.00765363 | 65.536 |
| `model.layers.0.self_attn.q_proj` | 2.61582e-05 | 1.11894e-05 | -0.000973922 | 4.194 |

### W4A4 — 상위 2개

두 집합의 top-2 공통 레이어는 2개다.

| 레이어 | Discovery KL | Confirmation KL | 전체 ΔNLL | 파라미터 M |
| --- | ---: | ---: | ---: | ---: |
| `lm_head` | 0.0341144 | 0.0175885 | +0.018983 | 65.536 |
| `model.layers.0.self_attn.q_proj` | 3.61827e-05 | 1.16943e-05 | -0.000957332 | 4.194 |

## 블록 안 Attention과 FFN 묶음

각 블록의 Attention projection 전체 또는 FFN projection 전체를 함께 바꾼 경우다.
FFN과 Attention은 파라미터 수가 다르므로 아래 원시 KL은 같은 저장 비용의 비교가 아니다.

| 모드 | 묶음 | 블록 수 | KL 중앙값 | ΔNLL 중앙값 | 블록당 파라미터 M |
| --- | --- | ---: | ---: | ---: | ---: |

## 블록 위치별 비교

개별 projection의 KL 중앙값이다. 앞(0–6), 중간(7–14), 뒤(15–21)를 비교하며 Norm·embedding·head는 제외한다.

| 모드 | 위치 | 개별 projection 수 | KL 중앙값 |
| --- | --- | ---: | ---: |
| W4A16 | 앞 | 1 | 5.13302e-06 |
| W16A4 | 앞 | 1 | 1.86738e-05 |
| W4A4 | 앞 | 1 | 2.39385e-05 |

## 실제 INT4 커널 확인

Discovery W4A4 상위 leaf를 기존 native 커널로 교체해 같은 조건에서 재측정했다. 나머지 모델은 FP16이다.

| 레이어 | QDQ 전체 KL | Native 전체 KL |
| --- | ---: | ---: |
| `lm_head` | 0.0258515 | 0.025844 |
| `model.layers.0.self_attn.q_proj` | 2.39385e-05 | 2.23785e-05 |

## Heatmap

색은 최종 출력 KL이다. 큰 값일수록 이번 조건에서 출력 변화가 크다. 회색은 해당 조건을 적용하지 않은 칸이다.

![개별 레이어 민감도](layers.png)

![Attention·FFN 묶음 민감도](groups.png)

원자료: `run.json`, `baseline.json`, `measurements.jsonl`, `ranking.csv`, `native_checks.json`, `summary.json`.
집합 간 순위가 안정적인 대상을 후속 calibration 개선 후보로 삼고, 최종 설정은 별도 전체 평가로 확인한다.
