# 전체 가중치 INT4 / activation FP16 — 파인튜닝 전 결과

공개 TinyLLaVA-1.5B의 모든 parameter를 RTN INT4로 압축한 W4A16 실험이다.
학습·gradient update·activation 양자화·AWQ/GPTQ 보정은 수행하지 않았다.

| 평가 | FP16 Accuracy | INT4 Accuracy | FP16 F1 | INT4 F1 | F1 변화(pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| POPE random | 90.47 | 71.30 | 90.26 | 60.88 | -29.38 |
| POPE popular | 86.60 | 71.10 | 86.82 | 60.72 | -26.10 |
| POPE adversarial | 82.83 | 68.70 | 83.72 | 58.80 | -24.92 |
| POPE macro F1 | — | — | 86.93 | 60.13 | -26.80 |
| ScienceQA image-test | 59.20 | 51.07 | — | — | — |

**GPU 메모리·시간**

| 평가 | FP16 / INT4 peak allocated GiB | allocated 절감률 | FP16 / INT4 평균 시간(ms) | 시간 변화율 |
| --- | ---: | ---: | ---: | ---: |
| pope | 4.1360 / 2.0724 | 49.89% | 230.60 / 308.09 | +33.61% |
| scienceqa | 4.9987 / 2.9346 | 41.29% | 183.00 / 185.71 | +1.48% |

ScienceQA accuracy 변화: -8.13pp. FP16에서 오답→INT4에서 정답 182개, 정답→오답 346개.

**적용 범위와 해석**

Vision·Projector·LLM의 626개 parameter tensors, 총 1,504,353,824개 값을 모두 INT4로 변환했다.
Embedding 2개, Linear 313개, Conv2d 1개, LayerNorm 53개, RMSNorm 45개와 해당 bias를 포함한다.
연속된 128개 값마다 FP16 absmax/7 scale을 저장하고, FP32에서 round-to-nearest 후 [-7,7]로 제한한다.
두 integer code를 uint8 한 개에 packing한다. [-7,7]의 15개 수준을 쓰는 대칭 narrow-range INT4다.
activation·KV cache는 원래 정밀도를 유지하고, softmax·normalization 내부 FP32 누산 등도 원래 동작을 유지한다.
RoPE 등 비학습 arithmetic buffer와 FP16 scale metadata는 4-bit parameter 범위에 포함하지 않는다.
packed payload는 775,682,460 bytes (0.7224 GiB), safetensors 파일은 775,918,724 bytes다.
각 연산 직전에 필요한 weight를 Triton으로 FP16 복원하고 원래 FP16 연산을 실행한다.
FP16 weights 전체를 계속 보관하는 fake-quant 방식은 아니지만, 정수 GEMM 전용 최적화 커널도 아니다.
따라서 시간은 이 reference backend의 실측값이며 최적화된 INT4의 일반적인 속도 효과로 해석하지 않는다.
각 문항 1회 측정이며 답변 길이가 달라질 수 있으므로 출력 길이·반복 측정을 추가해 속도를 평가한다.
FP16과 INT4 모두 batch 8·left padding을 사용한다. 문항당 시간은 batch 생성 시간을 실제 batch 크기로 나눈 값이다.
ScienceQA는 image-test 2,017문항과 로컬 CQM letter-only prompt다.

**원자료와 재현**

```bash
bash scripts/eval_int4_all.sh
python scripts/report_int4.py
```

- `results/int4_all_w4a16/comparison.json`: 재채점·조건 검증·정밀 수치.
- `results/int4_all_w4a16/{pope,scienceqa}/`: 예측·summary·run·quantization audit.
- `model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128/`: packed checkpoint·manifest.
- `docs/int4_implementation.md`: 구현 방식과 적용 범위.
- `docs/fp16_baseline.md`: 비교에 사용한 FP16 기준 성능.

이 결과는 전체 가중치를 단순 RTN으로 압축했을 때의 출발점이다. 후속 AWQ/GPTQ·혼합 정밀도·복구 학습 결과와 구분한다. W4A4는 아직 실행하지 않았다.
