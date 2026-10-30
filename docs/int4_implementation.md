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

**실제 INT4 곱셈 — Linear·Conv2d W4A4 구현**

`--precision int4-w4a4`는 `scripts/w4a4.py`, `scripts/int4_cuda.py`,
`kernels/int4_gemm.cu`를 사용한다. 기존 W4A16 코드와 결과 경로를 분리한다.
공개 모델의 313개 Linear와 SigLIP patch Conv2d 1개가 INT4×INT4 곱셈·INT32 누산을 수행한다.
가중치를 FP16으로 복원한 뒤 GEMM을 호출하는 fallback은 없다. bias·scale 처리는 FP16/FP32다.

- 원래 RTN 체크포인트의 정수 weight와 FP16 scale을 그대로 사용하며 재학습·재양자화하지 않는다.
- Native weight는 `q+8` encoding을 signed INT4 two's-complement로 바꿔 저장한다.
- 각 입력 행의 K 그룹마다 absmax/7을 FP16으로 저장하고, 그 scale을 사용해 activation을
  [-7,7] INT4로 round-to-nearest-even 양자화한다. 영점 그룹 scale은 1이다.
- 그룹 길이는 `gcd(in_features, original_weight_group_size)`다. group size 128 기준으로
  주요 LLM·Projector·Vision Linear는 128, Vision fc2(K=4304)는 16, patch Conv(K=588)는 4다.
- 원래 flat weight scale 경계를 보존하기 위해 작은 그룹을 Tensor Core의 K=32 타일로
  zero padding한다. 각 정수 부분합에 해당 weight·activation scale을 곱하고 FP32로 합산한다.
- 마지막 출력은 FP16으로 돌려준다. Residual·Attention QK/PV·Softmax·Norm·RoPE·embedding은
  기존 정밀도를 유지한다. 따라서 모델의 모든 연산을 INT4로 수행하는 구현은 아니다.
- Conv2d는 FP16 unfold로 patch를 배열한 후 실제 INT4 GEMM을 수행한다. 현재 groups=1,
  zero/valid padding을 지원한다. 공개 SigLIP patch convolution이 이 범위에 해당한다.

Tensor Core의 정수 곱셈에는 `nvcuda::wmma::experimental::precision::s4`를 사용한다.
현재 RTX 3070 Laptop(SM86)과 nvcc 11.5에서 device-only cubin을 만들고 CUDA Driver API로
PyTorch의 현재 stream에 올린다. PyTorch C++ 확장이나 별도의 CUDA runtime을 링크하지
않으므로 기존 torch 2.1.2+cu121/Conda 환경은 변경하지 않았다.

컴파일 결과에서 raw·scaled GEMM 모두 `IMMA.8832.S4.S4` 명령을 확인했다.
빌드 산출물·compiler 기록은 `model/.cache/int4_cuda/<source+compiler+SM hash>/`에 둔다.
backend는 SM75/SM8x를 대상으로 작성됐고 실제 검증 기기는 SM86이다. 다른 기기는 맞는
nvcc 지원과 별도 검증이 필요하다. 현재 커널은 정확성을 먼저 확인한 구현이며 성능 튜닝은 미완료다.
CUDA 실행 오류 시 FP16 GEMM으로 자동 전환하지 않는다.

```bash
# 두 전체 평가: batch 8, ScienceQA → POPE; 완료 후 세 모드 비교 보고서 생성
bash scripts/eval_w4a4_all.sh

# 작은 별도 확인; 정식 전체 점수와 구분
source scripts/env.sh
python scripts/test_inference.py --precision int4-w4a4 --benchmark scienceqa \
  --batch-size 8 --limit 8 --output results/w4a4_check/scienceqa
```

전체 결과는 `results/int4_all_w4a4/{scienceqa,pope}/`에 저장한다. 원본 packed checkpoint는
기존 `model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128/`을 재사용한다.
`quantization.json`에는 native 적용 범위·패딩 저장량·CUDA 명령 증거를, summary에는 현재
프로세스의 native 호출 수를 기록한다. run에는 구현 파일의 SHA256도 포함해 버전이 다른
실행이 같은 예측 파일에 섞이지 않도록 한다. 끝난 작업을 재실행해 모두 건너뛴 경우 현재
프로세스 호출 수는 0일 수 있다.
`report_w4a4.py`는 두 전체 평가가 완료된 후 FP16·W4A16과 동일 조건인지 검증하고
`results/int4_all_w4a4/comparison.json`, `docs/int4_all_w4a4_results.md`를 생성한다.

2026-10-29 검증:

- GPU의 exact INT32 결과가 독립적인 CPU 정수 행렬곱과 일치했다. -8·7 경계와 tail tile 포함.
- 그룹 경계·bias·영점·비연속 입력·Conv2d·현재 CUDA stream·실제 multimodal forward를 검사했다.
- FP16 weight dequantize API를 강제로 실패시키는 테스트에서도 native Linear가 실행됐다.
- 기존 검사를 포함한 테스트 15개를 통과했다.
- 공개 1.5B 모델의 가장 긴 ScienceQA 입력 8개(batch 8, 확장 길이 최대 1302)에서 모든
  `[8,1302,32000]` logits가 유한했고, 전체 314개 native 레이어 호출을 확인했다.
- ScienceQA·POPE 각각 첫 8문항을 batch 8·max_new_tokens 32로 실행했다.
  이는 실행 검증용이며 전체 성능 점수는 아니다. 원자료는 `results/w4a4_validation/`이다.

초기 per-row scale 1개(v1)보다 K 그룹별 scale(v2)로 activation 오차를 줄이도록 설계했다.
최종 구현은 `rtn_original_weights_dynamic_group_a4_native_wmma_v2`다. 초기 실험과 최종 검증은
서로 다른 output 폴더에 기록했다. 정확도·속도 개선 여부는 전체 평가와 추가 PTQ 실험으로 확인한다.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/a/anaconda3/envs/LLAVA/bin/python \
  -m unittest discover -s tests -p 'test_w4a4.py' -v
```

근거: [NVIDIA CUDA 11.5 sub-byte WMMA](https://docs.nvidia.com/cuda/archive/11.5.0/cuda-c-programming-guide/index.html#wmma-subbyte),
[NVIDIA Ampere Tensor Core 지원](https://docs.nvidia.com/cuda/ampere-tuning-guide/).

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
사용자가 FP16과 INT4 전체 평가를 배치 8로 새로 실행했고, 두 정밀도의 전체 평가가 모두 완료됐다.
결과는 `docs/int4_all_w4a16_results.md`, 전체 실험 요약은 `docs/experiment_summary.md`에 정리했다.

참고 API: [PyTorch 2.1 Linear](https://docs.pytorch.org/docs/2.1/generated/torch.nn.functional.linear.html),
[Triton vector kernel tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).
