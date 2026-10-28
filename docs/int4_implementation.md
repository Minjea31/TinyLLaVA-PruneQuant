# 전체 가중치 INT4 평가 구현

사용자가 선택한 첫 실험은 모든 구성요소의 가중치를 INT4로 만드는 **W4A16**이다.
activation도 줄이는 W4A4는 후속 실험이며, 이번 실행에는 fine-tuning을 적용하지 않는다.

공개 체크포인트를 기존 FP16 loader로 읽은 뒤 weight-only RTN을 적용한다.
LLM만 바꾸는 설정과 달리 Vision 421개, Projector 4개, LLM 201개 parameter tensors를
모두 변환한다. 총 1,504,353,824개 값이며 embedding·Conv2d·Linear·LayerNorm·RMSNorm과
bias를 포함한다. FP16으로 남기는 원본 parameter는 없다.

모델의 device/dtype 조회를 위해 zero-element FP16 parameter를 사용하지만 가중치 값은
포함하지 않는다. activation·KV cache·RoPE 등의 arithmetic buffer는 원래 동작을 유지한다.
softmax/RMSNorm 내부 FP32 누산과 최종 FP32 logits 역시 그대로다.

**양자화 설정**

- 방법: weight-only round-to-nearest(RTN), calibration·학습 없음.
- Group: tensor를 펼친 순서로 연속 128개 값. 마지막 group은 0으로 padding.
- Scale: 각 group의 absmax / 7을 FP16으로 저장하고, 그 실제 scale로 FP32 양자화.
- 범위: [-7, 7], 15개 수준을 쓰는 대칭 narrow-range INT4.
- Encoding: q + 8을 4-bit code로 저장, 두 code를 uint8 한 개로 packing.
- 복원: (code - 8) × scale. 필요한 tensor만 연산 직전에 FP16으로 복원.
- 연산: 원래 PyTorch FP16 Linear/Conv/Embedding/Norm. INT4 GEMM 전용 최적화는 없음.
- GPU 복원: Triton 2.1 기반 fused unpack/scale kernel, torch reference와 정확히 일치함을 검사.
- NF4·FP4 등의 4-bit floating codebook이 아닌 균일 정수 양자화다.

고정된 방식으로 한 번 압축한 checkpoint를 두 벤치마크가 공유한다. 기존 모델 파일과
FP16 평가 결과는 변경하지 않는다. packed payload는 775,682,460 bytes(약 0.7224 GiB),
metadata header를 포함한 safetensors는 775,918,724 bytes다. scale metadata가 있으므로
실효 저장량은 weight 하나당 약 4.125 bits다.

**실행**

```bash
bash scripts/eval_int4_all.sh
```

ScienceQA image-test 2,017문항, 이후 POPE 9,000문항을 차례로 실행한다. 같은 GPU에서
동시에 두 평가를 실행하지 않는다. batch 8, left padding, v1 prompt, greedy, max_new_tokens 32,
warmup 1, seed 2026과 FP16의 질문 파일 hash를 유지한다.
systemd-inhibit를 사용할 수 있으면 실행 중에만 sleep을 막는 임시 inhibitor를 사용한다.
시스템의 영구 절전 설정은 바꾸지 않는다. 평가가 종료되면 inhibitor도 해제된다.

기본 평가에서는 POPE의 중복 이미지·질문 입력도 각각 생성한다.
FP16과 INT4 모두 batch 8로 실행하고, 배치 생성 시간을 실제 batch 크기로 나눈 문항당
처리 시간과 전체 처리량을 기록한다. 각 예측에는 batch_id·batch_size·batch_seconds가 포함된다.

개별 실행:

```bash
source scripts/env.sh
python scripts/test_inference.py --precision int4-all --benchmark scienceqa --batch-size 8
python scripts/test_inference.py --precision int4-all --benchmark pope --batch-size 8
python scripts/report_int4.py
```

결과는 `results/int4_all_w4a16/<benchmark>/`의 predictions.jsonl, summary.json,
run.json, quantization.json에 저장한다. 기존 run 조건이 같은 경우 예측 파일에서 이어받으며,
조건이 다르면 새 output을 요구한다. 비교 보고서는 두 전체 평가가 모두 완료된 뒤 생성한다.

checkpoint는 `model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128/`에 model.safetensors와
manifest.json으로 저장한다. 일반 FP16 HF loader로 직접 불러오는 파일이 아니며,
`scripts/int4.py`의 packed module 복원 경로를 사용한다. 재사용 시 hash·모델 revision·
전체 parameter topology를 검사한다. 이 inference-only checkpoint를 기존 full training
코드에 그대로 넣어 학습하는 기능은 제공하지 않는다.

**검증**

```bash
/home/a/anaconda3/envs/LLAVA/bin/python -m unittest discover -s tests -p 'test_int4.py' -v
```

4개 검사를 통과했다: zero/홀수 길이/마지막 group 처리, layer별 parameter 완전 변환,
CPU dense RTN과 실제 legacy multimodal 모델 logits의 정확한 일치,
GPU Triton 복원과 torch reference 복원의 정확한 일치. 공개 전체 모델의 실제 이미지 추론과
checkpoint 재로딩도 확인했다.

측정 시간에는 unpack·FP16 weight 복원 비용도 포함한다. 따라서 메모리는 줄어도
FP16보다 느릴 수 있으며, 이 수치를 최적화된 INT4 backend의 속도라고 보고하지 않는다.
정확도 결과는 단순 RTN의 출발점으로 해석하고 AWQ/GPTQ 등 최적화 기법과 구분한다.

**평가 기준 변경**

2026-10-28 호스트 절전 복귀 이후 발생한 CUDA 장애는 사용자의 재부팅으로 복구됐다.
이후 사용자 요청으로 배치 1 정식 평가 결과를 삭제하고 FP16·INT4 기본값을 배치 8로 변경했다.
FP16 전체 평가는 사용자가 직접 실행한다. 기존 INT4 평가는 중지했으며,
새 FP16 기준 평가가 완료된 뒤 같은 배치 8로 새로 수행한다.

참고 API: [PyTorch 2.1 Linear](https://docs.pytorch.org/docs/2.1/generated/torch.nn.functional.linear.html),
[Triton vector kernel tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).
