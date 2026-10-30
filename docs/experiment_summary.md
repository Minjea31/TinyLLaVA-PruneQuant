# TinyLLaVA 실험 정리 — FP16·INT4·혼합 정밀도

정리일: 2026-10-30. FP16·W4A16 네 건은 2026-10-28, W4A4 두 건은 2026-10-29에 완료했다.
LLM 레이어 민감도 분석과 KL 기준 혼합 정밀도의 두 전체 평가는 2026-10-30에 완료했다.
아래 배치 8 작업은 공개 체크포인트의 FP16 평가와 파인튜닝 전 INT4 평가다.
이 평가에 학습용 checkpoint나 gradient update를 적용하지 않았다. 실제 가중치를 갱신한 짧은 학습 검증은 별도 기록으로 구분한다.

## 완료한 실험과 고정 조건

- 모델: TinyLLaVA-1.5B, SigLIP 실제 26개 encoder blocks + Projector + TinyLlama 22개 decoder blocks.
- FP16: 공개 원본 체크포인트를 기존 loader로 FP16 변환해 평가.
- W4A16: Vision·Projector·LLM의 모든 parameter를 uniform RTN INT4로 양자화. embedding·convolution·normalization·bias 포함.
- 총 626개 parameter tensors, 1,504,353,824개 값. Group size 128, FP16 scale, [-7,7] 대칭 정수 양자화.
- W4A16의 Activation·KV cache는 원래 정밀도를 유지하며, 원래 연산 내부의 FP32 누산도 유지한다.
- W4A4 Linear/Conv: 원래 INT4 weight 정수 값·scale을 유지하고 입력 activation을 행·K 그룹별로 동적 INT4 양자화한다. 313개 Linear와 patch Conv2d 1개는 INT4×INT4 곱셈·INT32 누산을 수행한다.
- W4A4에서도 Attention QK/PV 행렬곱·Softmax·Norm·RoPE·embedding lookup 등은 기존 연산 정밀도를 유지한다. 모델 전체의 모든 산술 연산이 INT4인 것은 아니다.
- 세 기준 설정의 RTN weight는 calibration·AWQ/GPTQ 보정 없이 만들었다. 추가 파인튜닝은 없다.
- Mixed: calibration의 KL로 weight 7개 레이어·activation 3개 Linear 입력의 FP16 예외를 골랐다. 나머지 INT4 weight 값·scale은 기존 RTN과 같다.
- 공통 조건: batch 8, left padding, v1 prompt, greedy, seed 2026, max_new_tokens 32, warmup 1.
- POPE 9,000문항, ScienceQA image-test 2,017문항. ScienceQA는 로컬 CQM letter-only 조건이다.
- 네 설정 모두 전체 문항을 각각 생성했다. 총 44,068개 예측이며 답변 재사용은 없다.

## 정확도 비교

| 지표 | FP16 | 전체 W4A16 | W4A4 Linear/Conv | Mixed | Mixed vs W4A4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| POPE macro F1 | 86.93% | 60.13% | 54.50% | 58.82% | +4.33pp |
| ScienceQA accuracy | 59.20% | 51.07% | 41.30% | 45.56% | +4.26pp |

| POPE category | FP16 F1 | W4A16 F1 | W4A4 F1 | FP16 recall | W4A16 recall | W4A4 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 90.26% | 60.88% | 56.60% | 88.33% | 44.67% | 42.33% |
| popular | 86.82% | 60.72% | 54.76% | 88.27% | 44.67% | 40.67% |
| adversarial | 83.72% | 58.80% | 52.13% | 88.27% | 44.67% | 40.73% |

ScienceQA 정답 수: FP16 1,194/2,017 → INT4 1,030/2,017.
선택지 파싱 실패는 0건 → 5건이다.
같은 문항끼리 비교하면 FP16 오답→INT4 정답 182개, FP16 정답→INT4 오답 346개다.

W4A4의 ScienceQA 정답 수는 833/2,017, 선택지 파싱 실패는 10건이다.
FP16 정답→W4A4 오답은 574개, FP16 오답→W4A4 정답은 213개다.
W4A16과 비교하면 POPE macro F1이 추가로 5.64pp, ScienceQA accuracy가 9.77pp 하락했다.

Mixed의 ScienceQA 정답 수는 919/2,017, 파싱 실패는 3건이다.
W4A4 오답→Mixed 정답은 367개, W4A4 정답→Mixed 오답은 281개로 순증 86개였다.

## GPU 메모리와 처리 시간

| 평가·모드 | peak allocated GiB | 문항당 처리 시간 ms | 처리량 문항/초 |
| --- | ---: | ---: | ---: |
| POPE / FP16 | 4.1360 | 230.60 | 4.34 |
| POPE / W4A16 | 2.0724 | 308.09 | 3.25 |
| POPE / W4A4 Linear/Conv | 2.1384 | 545.51 | 1.83 |
| POPE / Mixed | 2.2428 | 560.70 | 1.78 |
| ScienceQA / FP16 | 4.9987 | 183.00 | 5.46 |
| ScienceQA / W4A16 | 2.9346 | 185.71 | 5.38 |
| ScienceQA / W4A4 Linear/Conv | 2.9956 | 538.78 | 1.86 |
| ScienceQA / Mixed | 3.1075 | 539.77 | 1.85 |

문항당 처리 시간은 batch 생성 시간을 실제 batch 크기로 나눈 값이며 개별 요청 latency가 아니다.
Generation에는 Vision·Projector·LLM이 포함되고, 전처리·모델 로드·warmup은 제외한다.
Peak allocated는 CUDA tensor 할당량이며 nvidia-smi의 프로세스 전체 사용량과 구분한다.
W4A16 구현은 packed 가중치를 필요한 연산 직전에 Triton으로 FP16 복원하고 원래 연산을 수행한다.
W4A4 Linear/Conv는 signed INT4 WMMA 커널을 사용한다. 입력 quantize/pack·타일 padding·그룹 scale 적용 비용을 포함하며 성능 튜닝은 미완료다.
답변 길이도 달라질 수 있으며 각 모드의 단일 실행 측정이다.

## 결과 해석

- 전체 parameter INT4 RTN W4A16은 메모리를 약 41~50% 줄였지만 두 평가의 정확도가 하락했다.
- W4A4 Linear/Conv도 FP16보다 메모리를 POPE 48.30%, ScienceQA 40.07% 줄였다. 다만 기존 W4A16보다 각각 3.19%, 2.08% 더 사용했다.
- POPE recall은 FP16 약 88% → W4A16 44.67% → W4A4 약 41~42%로 하락했다. 실제 존재하는 물체에 no로 답하는 false negative가 크게 늘었다.
- W4A16의 POPE precision은 FP16보다 높다. W4A4의 precision은 W4A16보다 낮아져 추가적인 false positive도 발생했다.
- 현재 결과만으로 Vision·Projector·LLM 중 어느 부위가 손실의 주원인인지는 확정할 수 없다.
- W4A16 처리 시간은 FP16 대비 POPE 약 33.61%, ScienceQA 약 1.48% 늘었다. 현재 W4A4 구현은 각각 FP16의 2.37배, 2.94배 시간이 걸렸다. 이번 구현에서 메모리 감소가 속도 향상으로 이어지지는 않았다.
- Mixed는 W4A4 대비 두 평가에서 약 4.3pp를 회복했다. Peak allocated는 POPE 107MiB, ScienceQA 115MiB 증가했고, 처리 시간은 각각 2.79%, 0.18% 늘었다.
- Mixed의 FP16 대비 메모리 감소는 각각 45.77%, 37.83%다. 정확도는 FP16 및 전체 W4A16보다 낮아 아직 회복할 손실이 남아 있다.

## LLM 레이어 민감도 — 2026-10-30 완료

LLM의 201개 leaf와 블록 안 Attention·FFN 묶음 44개, 총 643개 W/A 조건을 calibration 128개에서 측정했다.
Discovery 64개·confirmation 64개는 서로 다른 예제이며, 각 예제에서 정답 assistant 토큰을 최대 32개 골랐다.
양자화 대상 이외의 모델은 FP16으로 고정했고, 이미지 인코더·Projector의 FP16 출력과 변경되지 않는 LLM 앞부분을 재사용했다.
학습·LoRA·optimizer update는 없었다. 같은 FP16 연산의 QDQ 시뮬레이션으로 W/A를 분리하고 상위 3개 W4A4 leaf를 실제 INT4 커널로 재확인했다.

| 대상 | W4A16 KL | W16A4 KL | W4A4 KL | 관찰 |
| --- | ---: | ---: | ---: | --- |
| lm_head | 0.007081 | 0.016410 | 0.023737 | Activation 영향이 더 큼 |
| layers.2.mlp.down_proj | 0.009355 | 0.000248 | 0.009446 | Weight 영향이 더 큼 |
| layers.21.mlp.down_proj | 0.001429 | 0.004824 | 0.006178 | Activation 영향이 더 큼 |
| layers.20.mlp.down_proj | 0.000622 | 0.002447 | 0.003096 | Activation 영향이 더 큼 |
| layers.8.input_layernorm | 0.003268 | 해당 없음 | 해당 없음 | RMSNorm 가중치도 민감 후보 |

숫자는 FP16 대비 최종 다음 토큰 분포의 평균 KL이며, 높을수록 이번 개입의 출력 변화가 컸다는 뜻이다. 정확도 하락률은 아니다.
W4A4 개별 projection의 KL 중앙값은 앞(0–6번) 0.000176, 중간(7–14번) 0.000596, 뒤(15–21번) 0.000879로 후반이 초반의 약 5배였다.
다만 weight에 민감한 2번 down_proj 같은 초반 예외가 있으므로 모든 초반 레이어를 덜 중요하다고 취급하지 않는다.

W4A4 블록 묶음 평균 KL은 Attention 0.002377, FFN 0.003183이었다. 같은 예제에서의 FFN−Attention 차이 평균은 0.000807, paired bootstrap 95% 구간은 [0.000551, 0.001029]였다.
Attention 묶음은 약 9.44M, FFN은 34.60M parameters이므로 이는 저장 비용이나 파라미터 수를 맞춘 비교가 아니다.
Activation-only에서는 Attention 묶음의 중앙값이 더 크고 FFN 묶음의 평균이 더 커, 지표·블록 위치에 따라 해석이 달라진다.

세 정밀도 조건 모두 discovery·confirmation top-10의 공통 레이어는 9개였다.
기존 native 커널로 재확인한 W4A4 KL도 lm_head 0.023735, layers.2.down_proj 0.009423, layers.21.down_proj 0.006179로 같은 주요 후보의 민감도가 유지됐다.
643개 조건 × 128개 예제의 ID·fold·집계·source hash를 검증했고, 마지막 FP16 가중치 hash와 출력은 시작 전과 정확히 일치했다.

이 결과는 단일 대상 민감도다. 전체 INT4에서의 회복 효과나 POPE·ScienceQA 정확도 개선을 확정한 결과는 아니다.
후속 비교 후보는 2번 down_proj의 weight group·clipping, lm_head와 마지막 FFN의 activation group·scale, 입력 RMSNorm 양자화 설정이다.

- [전체 분석·bootstrap·실행 조건](../results/sensitivity_llm/report.md)
- [개별 레이어 히트맵](../results/sensitivity_llm/layers.png)
- [Attention·FFN 묶음 히트맵](../results/sensitivity_llm/groups.png)
- [전체 순위 CSV](../results/sensitivity_llm/ranking.csv)

## KL 기준 혼합 정밀도 — 2026-10-30 완료

선정 기준은 weight의 W4A16 KL ≥ 0.001, activation의 W16A4 KL ≥ 0.002다.
민감도 선정에 쓴 두 64개 집합에서 모두 기준을 넘는 레이어를 사용했다.
Weight FP16은 layers.2/21.mlp.down_proj, lm_head, layers.6/8/21.input_layernorm, model.norm의 7개다.
Activation FP16은 layers.20/21.mlp.down_proj와 lm_head의 3개다. Norm의 A4는 이 실험에서 정의하지 않았다.

| Linear·Conv 조건 | 개수 | 계산 경로 |
| --- | ---: | --- |
| W4A4 | 310 | native INT4×INT4→INT32 |
| W16A4 | 1 | layers.2.down_proj, A4 QDQ 후 FP16 GEMM |
| W4A16 | 1 | layers.20.down_proj, W4 DQ 후 FP16 GEMM |
| W16A16 | 2 | layers.21.down_proj·lm_head, FP16 GEMM |

모든 target 이외의 모델은 기존 W4A4 구성을 유지했다. FP16 weight는 RTN을 복원한 근삿값 대신 공개 원본 FP16 값을 사용했고 원본 safetensors의 FP16 변환 hash와 일치했다.

선정에 사용한 128개와 ID·이미지 SHA256이 겹치지 않는 64개에서 별도 비교했다.
이미지 인코더·Projector·LLM을 각 모드에서 다시 계산했으며 FP16 teacher prefix를 재사용하지 않았다.

| 설정 | FP16 대비 KL | 정답 토큰 NLL |
| --- | ---: | ---: |
| FP16 | 0 | 1.088557 |
| W4A4 | 0.507639 | 1.617922 |
| Weight만 FP16 유지 | 0.455714 | 1.551770 |
| Activation만 FP16 유지 | 0.467118 | 1.575689 |
| 둘 다 유지 | 0.407886 | 1.518524 |

Weight-only·activation-only의 위 비교는 calibration 지표다. 전체 POPE·ScienceQA는 결합 설정을 평가했다.
결합 설정의 POPE F1은 random 59.80%, popular 60.05%, adversarial 56.61%이며 macro F1은 58.82%다.
W4A4 대비 recall은 약 41~42%에서 약 46%로 회복했지만 FP16 약 88%와는 차이가 남았다.
추가 학습·LoRA·optimizer update는 없었다. 평가 결과를 보고 threshold를 바꾸지 않았다.

검증은 25개 테스트, 짧은 생성 및 완료 결과 이어받기, 전체 예측 재채점·동일 입력 조건·314개 weighted 레이어의 경로·원본 FP16 값·기존 INT4 checkpoint hash를 포함한다.

- [전체 혼합 정밀도 평가](../results/mixed_precision/both/report.md)
- [독립 calibration의 W/A 분리 비교](../results/mixed_precision/calibration/report.md)
- [추가 검증·정답 전환 기록](../results/mixed_precision/both/verification.json)
- [실행 도구](../scripts/eval_mixed_precision.sh)

## 다음 실험 계획 — 전체 모델의 블록·레이어 민감도

아래 설계 중 LLM의 개별 레이어와 블록 안 Attention·FFN 단일 대상 비교는 위 결과처럼 실행했다.
선정한 여러 레이어를 함께 FP16으로 복구하는 혼합 정밀도 평가도 위처럼 완료했다.
Vision·Projector까지 확장한 비교와 전체 INT4 배경에서 모든 레이어를 하나씩 복구해 순위를 구하는 실험은 후속 계획이다. 모든 측정은 optimizer·gradient update·LoRA 없이 수행한다.

### 1. 먼저 블록 단위로 손실 위치 찾기

SigLIP encoder 26개, TinyLlama decoder 22개, Projector 전체 1개를 우선 49개 블록으로 본다. 블록 밖의 patch Conv·position/token embedding·최종 Norm·lm_head도 별도 대상으로 기록한다.

- **하나만 양자화:** 나머지는 FP16으로 두고 선택한 블록만 양자화한다. FP16 대비 최종 출력이 얼마나 변하는지 측정한다.
- **하나만 복구:** 기존 전체 INT4 모델에서 선택한 블록만 원본 FP16으로 복구한다. 전체 INT4 대비 최종 출력이 얼마나 회복하는지 측정한다.
- 두 순위가 다른 경우에는 양자화된 앞뒤 블록 사이의 상호작용이나 오차 누적 가능성을 추가 확인한다. 개별 블록의 손실·회복량은 단순 합산하지 않는다.

복구는 원인 분석을 위한 개입이다. 최종 모델을 반드시 mixed precision으로 만든다는 뜻은 아니다. W4 조건의 weight는 현재 packed RTN 체크포인트의 값을 사용하고 group 128·bias/Norm 양자화 정책을 고정한다.

### 2. 가중치와 activation의 영향을 분리하기

| 선택한 블록 조건 | Weight | Linear/Conv 입력 activation | 목적 |
| --- | --- | --- | --- |
| W16A16 | 원본 FP16 | 원본 정밀도 | 기준 |
| W4A16 | 기존 RTN INT4 | 원본 정밀도 | 가중치 양자화 영향 |
| W16A4 | 원본 FP16 | 현재 방식으로 INT4 양자화 | activation 양자화 영향 |
| W4A4 | 기존 RTN INT4 | 현재 방식으로 INT4 양자화 | 결합 영향 |

Norm·embedding 등에는 Linear/Conv의 A4 정의를 그대로 적용하지 않는다. 가중치만 바꾸는 별도 항목으로 분석한다.
현재 W4A16→native W4A4 결과에는 activation 양자화와 누산 경로 변경이 함께 포함돼 있어, 그 차이를 activation만의 손실로 단정하지 않는다.
분리 실험에서는 양자화→복원한 값을 같은 FP16 연산에 넣어 산술 경로를 통일할 수 있다. 이는 정확도 분석용 시뮬레이션이며 INT4 속도·메모리 결과로 사용하지 않는다. 상위 후보와 최종 설정은 기존 실제 INT4 커널로 다시 확인한다.

### 3. 작은 calibration에서 순위를 구한 뒤 세부 레이어 분석

- 기존 학습 데이터 기반 calibration 512개 중 source·길이를 나누어 64~128개를 순위 산출에 사용하고, 겹치지 않는 별도 부분집합에서 순위 안정성을 확인한다. 샘플 ID와 hash를 고정하며 POPE·ScienceQA test로 순위를 고르지 않는다.
- 같은 정답 prefix를 입력하는 teacher forcing으로 assistant 응답 위치의 `KL(p_FP16 || p_perturbed)`와 정답 토큰 NLL 변화를 측정한다. KL의 softmax temperature는 1로 고정하고 padding·image placeholder·사용자 prompt 위치는 점수에서 제외한다. Teacher forcing은 forward 평가이며 학습이 아니다.
- 레이어의 동일한 FP16 입력에 대한 상대 출력 오차 `sum((y_quant-y_FP16)^2) / (sum(y_FP16^2)+eps)`와 cosine도 기록한다. 이 값만으로 최종 답변의 중요도를 판단하지 않는다.
- 민감한 LLM 블록은 `q/k/v/o_proj`, `gate/up/down_proj`, 두 RMSNorm으로 분해한다. Vision 블록은 Attention의 projection·FFN `fc1/fc2`·LayerNorm으로 분해하고 Projector의 두 Linear도 분리한다.
- 실제 forward 호출과 최종 출력으로 이어지는 경로를 확인한다. 예를 들어 현재 SigLIP Tower는 `hidden_states[-1]`을 사용하므로, post_layernorm의 큰 로컬 오차가 그대로 최종 답변 손실을 의미하지 않는다.
- 결과는 블록 번호 × W/A 조건의 heatmap, 세부 레이어 순위, 원본 대비 최종 KL/NLL, 사용한 파라미터 수로 정리한다. 마지막 후보 설정만 POPE·ScienceQA 전체로 확인한다.

### 4. 민감도 결과를 이용해 양자화 개선

- Weight에 민감한 레이어: group 128→64→32, clipping과 보정된 rounding을 비교한다. GPTQ 같은 2차 정보 기반 weight PTQ는 비교 대상으로 검토한다. [GPTQ](https://arxiv.org/abs/2210.17323)
- Activation에 민감한 레이어: outlier 통계·scale·clipping을 살피고, LLM에 대해서는 rotation 방식의 적용 가능성을 검토한다. TinyLLaVA 전체에 QuaRot을 바로 적용할 수 있다고 가정하지 않는다. [QuaRot](https://arxiv.org/abs/2404.00456)
- 모든 weight를 INT4로 유지하려면 민감한 레이어에 더 작은 group이나 개선된 양자화 파라미터를 적용한다. scale 저장량과 padding을 포함한 실제 byte 증가도 함께 비교한다.
- 혼합 정밀도를 별도 실험으로 허용할 경우 민감한 레이어를 INT8/FP16으로 남기고, 회복량을 추가 저장 byte와 함께 비교한다.

Hessian trace를 이용한 민감도·혼합 정밀도 배정은 HAWQ-V2의 참고 방향이다. 첫 단계는 gradient와 Hessian 추정이 필요 없는 위 forward 개입 실험으로 진행하는 것을 제안한다. 민감도 분석 자체를 새로운 방법으로 주장하지 않고, W/A별 순위 차이와 블록 간 상호작용을 확인한다. [HAWQ-V2](https://arxiv.org/abs/1911.03852)

## 별도 전체 모델 학습 도구

이전에 요청한 전체 SFT 도구와 짧은 기능 검증 기록은 [전체 모델 학습 가이드](full_training.md)에 보존한다.
위 FP16·INT4 전체 평가는 공개 원본 체크포인트를 사용했으며 해당 학습 checkpoint를 사용하지 않았다.

## 검증과 원자료

report_int4.py·report_w4a4.py·report_mixed_precision.py로 전체 예측의 ID·질문·이미지·정답·source hash·run 조건·재채점 점수·batch 시간·memory metadata와 INT4 checkpoint hash를 검증했다.
네 모드가 같은 batch 8·데이터·생성 조건을 사용하는 것도 확인했다.
W4A4에서는 두 평가 모두 실제 314개 레이어가 호출됐으며, CUDA 커널의 `IMMA.8832.S4.S4` 명령과 구현 source hash를 확인했다.
기존 packed INT4 checkpoint의 SHA256도 평가 후 일치했다. 상세 검증은 `results/int4_all_w4a4/verification.json`에 기록했다.

- [INT4 상세 결과](int4_all_w4a16_results.md)
- [실제 INT4 곱셈 W4A4 결과](int4_all_w4a4_results.md)
- [W4A4 비교·검증 원자료](../results/int4_all_w4a4/comparison.json)
- [혼합 정밀도 비교·검증 원자료](../results/mixed_precision/both/comparison.json)
- [FP16 기준 성능](fp16_baseline.md)
- [비교·검증 원자료](../results/int4_all_w4a16/comparison.json)
- [전체 모델 학습 검증](full_training.md)

이전 batch 1 정식 평가 결과는 삭제된 상태이며 이 요약에는 포함하지 않았다.
