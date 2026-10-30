# TinyLLaVA Quantization 연구 환경

공개 TinyLLaVA-1.5B에 추가 학습 없이 양자화를 적용해 정확도·Peak VRAM·처리 시간을
비교하는 워크스페이스다. FP16·전체 가중치 W4A16·실제 INT4 곱셈 W4A4·KL 기준 혼합 정밀도의
배치 8 전체 평가를 완료했다. [전체 실험 요약](docs/experiment_summary.md)에 조건과 결과를 정리했다.

기존 Conda `LLAVA`를 사용한다. 모델 구성과 코드 탐색은
[모델 구조](docs/model_architecture.md), 데이터 준비 상태는 [데이터 검사](docs/data_inventory.json)를 참고한다.

## 시작

워크스페이스 루트에서 실행한다.

```bash
source scripts/env.sh       # conda activate LLAVA 포함
python scripts/check_environment.py
# 다른 작업 공간에서 기록된 third_party commits를 복원할 때:
python scripts/bootstrap_sources.py
```

새 Conda 환경은 만들지 않았다. 기존 Python 3.10.21, PyTorch 2.1.2+cu121,
torchvision 0.16.2+cu121, Transformers 4.37.2, accelerate 0.21.0을 재사용한다.
실제 import 충돌이 있던 PEFT만 공식 legacy 요구 버전 0.4.0으로 맞췄다.
추가한 `hf-transfer==0.1.9`는 공식 모델 다운로드의 선택 전송 모듈이다.
[패키지 분석](docs/dependency_analysis.md), [pip freeze](docs/environment_requirements.txt),
[Conda 명시 목록](docs/conda_packages.txt), [환경 검사 결과](docs/environment.json)를 참고한다.

`pip freeze`에는 기존 ROS 및 다른 프로젝트의 editable `llava` 경로도 포함되므로 다른 기기에
그대로 일괄 설치하는 명령으로 사용하지 않는다. 이 프로젝트의 핵심 버전은
[requirements_core.txt](configs/requirements_core.txt)에 따로 기록했다. 시스템 드라이버와
PyTorch를 일괄 업그레이드하지 않는다. `third_party` 소스는 PYTHONPATH로 불러온다.

## 모델과 구조

- [bczhou/TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B): TinyLlama 1.1B,
  22 decoder layers, hidden 2048, GQA heads 32/4.
- [bczhou/TinyLLaVA-1.5B-SigLIP](https://huggingface.co/bczhou/TinyLLaVA-1.5B-SigLIP):
  384×384 SigLIP, 729 patches, hidden 1152, 설정 27 layers / 실제 legacy encoder 26 layers.
- Projector: Linear(1152,2048) → GELU → Linear(2048,2048).
- 전체 1,504,353,824 parameters. 원본 safetensors는 FP32이며 baseline 로더에서 FP16으로 변환한다.

요청 체크포인트는 legacy 형식이므로 공식 Factory main과 함께 `tinyllava_bench` 브랜치를
별도 보관한다. [구조·Recipe 분석](docs/model_architecture.md)과 `docs/*_snapshot.json`에
Revision/파일별 SHA256, `docs/*_analysis.json`에 tensor 구조를 기록한다.
원본 파일을 수정하거나 FP16/quantized 파일로 덮어쓰지 않는다.

```bash
python scripts/download_model.py --parallel
python scripts/inspect_model.py
# 다운로드 완료 후 GPU 없이 로컬 파일·전처리 검사:
python scripts/test_inference.py --validate-load --image <실제_이미지_경로>
```

실행 시 메모리의 Vision Encoder 참조만 로컬 경로로 바꾸며 network 없이 로딩한다.
일반 공식 CLI에 사용할 경우 `python scripts/create_local_model_view.py`로 원본 weights를
symlink한 별도 local Config view를 만들 수 있다.

## 데이터

논문의 LLaVA 558K / mix665K 및 ShareGPT4V 1.246M / mix665K Annotation을 보존한다.
공개 1.5B는 논문 표/모델 카드 대응상 ShareGPT4V share recipe에 해당한다.
LAION 이미지 전체와 COCO SFT의 실제 이미지 부분집합을 초기 연구에 사용한다.
SAM/WebData 등 전체 ShareGPT4V 이미지가 모두 준비되었다고 간주하지 않는다.

```bash
python scripts/download_data.py --group annotations
python scripts/download_data.py --group pretrain
python scripts/download_data.py --group training-subset
python scripts/download_data.py --group initial-eval
python scripts/map_data_paths.py
python scripts/prepare_calibration.py --count 512 --seed 2026
```

큰 파일은 이어받기와 공간 검사를 지원한다. 정상 파일은 재사용하고 실패 원인은 logs에 기록한다.
Calibration은 원본 image-text 연결, 고정 seed, source/길이 strata, unique image SHA256,
다운로드한 평가 이미지와의 분리를 유지한다. 목록은 `data/calibration/calibration.json`,
hash·추출 기준은 `data/manifests/calibration.json`에 기록한다.

[학습 데이터 출처·경로](docs/datasets.md), [평가 데이터·Metric·실행법](docs/evaluation.md)에
전체 데이터 다운로드 스크립트와 추가 준비 조건이 있다. 우선 POPE와 ScienceQA image-test를
준비하고 다른 벤치마크는 `scripts/download_remaining.sh`로 받는다.

## FP16 baseline

2026-10-28 FP16 배치 8의 POPE 9,000문항과 ScienceQA image-test 2,017문항 평가를 완료했다.
POPE macro F1은 **86.93%**, ScienceQA accuracy는 **59.20%**다.
전체 11,017개 예측을 재채점하고 원본 입력·run 조건과 일치함을 검증했다.
상세 점수·메모리·처리 시간은 [FP16 기준 평가](docs/fp16_baseline.md)에 정리했다.

```bash
source scripts/env.sh
python scripts/check_environment.py
python scripts/test_inference.py
bash scripts/eval_fp16_all.sh
# 개별 실행:
python scripts/test_inference.py --benchmark pope --batch-size 8
python scripts/test_inference.py --benchmark scienceqa --batch-size 8
```

공식 legacy loader, 모델 generate, image preprocessing, `v1` prompt를 사용한다.
greedy / batch 8 / left padding / seed 2026 / max_new_tokens 32 / warmup 1이다.
입력 JSONL hash와 모델 revision을 결과에 함께 저장한다. 결과 경로는 `results/baseline/`이며
예측, 정확도, 추론 시간, CUDA peak allocated/reserved VRAM을 분리해 기록한다.
GPU가 없으면 `status.json`을 남기고 종료 코드 2로 종료한다. baseline 점수를 만들어내지 않는다.
ScienceQA의 로컬 CQM letter-only 조건은 논문 eval.zip 원본 질문 조건과 구분한다.

## 전체 가중치 INT4 평가

Vision·Projector·LLM의 embedding·convolution·normalization·bias를 포함한 전체 가중치에
INT4 RTN을 적용하는 W4A16 평가를 추가했다. activation은 원래 정밀도를 유지하며
파인튜닝은 하지 않는다. 실행은 `bash scripts/eval_int4_all.sh`이다.

FP16과 같은 배치 8로 두 전체 평가를 완료하고 재검증했다. POPE macro F1은
**86.93% → 60.13%**, ScienceQA accuracy는 **59.20% → 51.07%**다.
Peak allocated VRAM은 POPE 49.89%, ScienceQA 41.29% 줄었다.
[전체 실험 요약](docs/experiment_summary.md), [상세 결과](docs/int4_all_w4a16_results.md)와
[구현·이어 실행 안내](docs/int4_implementation.md)를 참고한다.

## 실제 INT4 곱셈 — W4A4 Linear·Conv2d

`--precision int4-w4a4`로 313개 Linear와 SigLIP patch Conv2d 1개를 실제
INT4×INT4 Tensor Core 곱셈·INT32 누산 경로로 실행한다. activation은 입력 행의 K 그룹마다
동적 INT4 양자화하고 원래 INT4 weight·scale은 그대로 사용한다. 추가 학습은 없다.
Attention QK/PV·Softmax·Norm·RoPE·embedding은 기존 정밀도를 유지한다.

```bash
bash scripts/eval_w4a4_all.sh
```

2026-10-29 POPE 9,000문항과 ScienceQA image-test 2,017문항을 모두 완료하고 재검증했다.
POPE macro F1은 **54.50%**, ScienceQA accuracy는 **41.30%**다.
Peak allocated는 각각 **2.14 GiB / 3.00 GiB**, 문항당 처리 시간은 **545.51 ms / 538.78 ms**다.
기존 W4A16 결과와 별도로 `results/int4_all_w4a4/`에 저장했다.
[세 정밀도 비교 보고서](docs/int4_all_w4a4_results.md)에 전체 점수·메모리·처리 시간을 정리했다.
GPU 정수 기준값 일치·native INT4 명령·공개 전체 모델의 finite logits와 batch 8 실행을
확인했다. 커널 성능 튜닝은 미완료이며, [구현·검증 범위](docs/int4_implementation.md)를 참고한다.

## LLM 레이어별 민감도

2026-10-30 추가 학습 없이 LLM 개별 레이어 201개와 블록별 Attention·FFN 묶음 44개를
128개 calibration 예제에서 비교했다. W4A16·W16A4·W4A4의 643개 조건을 측정했고,
나머지 모델은 FP16으로 고정했다. 출력 KL·정답 토큰 NLL을 사용한 단일 대상 분석이다.
가중치·activation 영향을 분리하기 위해 같은 FP16 연산에서 양자화·복원을 시뮬레이션하고,
상위 W4A4 레이어 3개는 실제 INT4 커널로 다시 확인했다.

가중치는 `layers.2.mlp.down_proj`, activation은 `lm_head`와 마지막 FFN의 `down_proj`가
주요 민감 후보였다. [분석 보고서](results/sensitivity_llm/report.md),
[레이어 히트맵](results/sensitivity_llm/layers.png), [원자료](results/sensitivity_llm/ranking.csv)를 참고한다.
POPE·ScienceQA 정확도는 이 분석으로 다시 측정하지 않았다.

```bash
bash scripts/eval_llm_sensitivity.sh
# 완료한 원자료의 검증·표·그림만 다시 생성:
python scripts/report_llm_sensitivity.py
```

## KL 기준 weight·activation FP16 유지

2026-10-30 weight KL ≥ 0.001, activation KL ≥ 0.002를 두 선정 집합에서 모두 넘은
weight 7개 레이어·activation 3개 Linear 입력을 FP16으로 유지해 평가했다. 추가 학습은 없다.
나머지 310개 Linear·Conv는 실제 INT4 곱셈을 사용한다. W16A4·W4A16·W16A16 예외는
FP16 GEMM 경로이며, FP16 weight는 공개 원본에서 양자화 전에 보관한 값이다.

선정에 사용하지 않은 calibration 64개에서 weight만 유지·activation만 유지·둘 다 유지를
분리 비교했다. 결합 설정의 전체 batch 8 평가에서는 POPE F1 **54.50% → 58.82%**,
ScienceQA accuracy **41.30% → 45.56%**로 W4A4 대비 회복했다.
Peak allocated는 각각 **2.24 GiB / 3.11 GiB**다. 최종 설정의 전체 평가는 11,017문항이다.

```bash
bash scripts/eval_mixed_precision.sh
# 완료된 전체 평가의 비교·검증 보고서만 다시 생성:
python scripts/report_mixed_precision.py
```

[전체 평가 보고서](results/mixed_precision/both/report.md),
[독립 calibration 분리 비교](results/mixed_precision/calibration/report.md),
[고정한 기준과 후보](results/sensitivity_llm/fp16_threshold_candidates.json)를 참고한다.
Weight-only·activation-only의 비교는 calibration에서 수행했고, 전체 벤치마크는 결합 설정을 평가했다.

## 별도 전체 모델 학습 도구

이전에 요청한 전체 모델 학습 도구는 별도로 보존한다. 위 FP16·INT4 평가에는 사용하지 않았다.
Vision Encoder·Projector·LLM 전체를 학습하려면 다음을 실행한다.
기존 LLAVA 환경에서 FP16 GPU 계산과 CPU FP32 master/gradient/Adafactor offload를 사용한다.

```bash
bash scripts/train_full.sh
# 짧은 전체 구성요소 업데이트 확인:
bash scripts/train_full.sh --max-steps 2 --max-samples 4 --gradient-accumulation 2
```

기본 학습 데이터는 실제 이미지가 준비된 COCO SFT 부분집합이며, 이미지 토큰을 포함한
길이·assistant label 조건에 맞는 항목만 사용한다. 실행 설정, 이어 학습, 체크포인트와
검증 범위는 [전체 학습 가이드](docs/full_training.md)에 정리했다.

## 폴더

```text
model/        TinyLLaVA-1.5B/, TinyLLaVA-1.5B-SigLIP/, quantized/, checkpoints/, model_study/
data/         train/{llava,sharegpt4v,annotations}/
              eval/{pope,scienceqa,textvqa,gqa,vqav2,mme,mmvet,mmmu}/
              calibration/, manifests/
third_party/  TinyLLaVA_Factory/, TinyLLaVA_Factory_legacy/, MegaTrain/
configs/      핵심 버전과 연구 조건
scripts/      환경 검사·다운로드·구조 분석·baseline·검증
results/      baseline/, int4_all_w4a16/, int4_all_w4a4/, sensitivity_llm/, mixed_precision/
logs/         다운로드 로그
docs/         모델 구조·데이터·평가 실행법·실험 결과
```

`.gitignore`는 데이터셋(`data/`), 모델(`model/`), 학습 체크포인트와 가중치·학습 상태 파일을
제외한다. 공부용 `model/model_study/`와 `model/README.md`, 코드·문서·로그·평가 결과는 포함한다.
HF 모델 cache는 model 아래, datasets cache는 data 아래로 고정한다. snapshot local_dir를
사용하여 global HF hub cache에 같은 대형 weights를 중복 저장하지 않는다.
경로 alias는 symlink를 사용하며 원본 모델 안의 Vision weights와 별도 Vision snapshot의
중복은 공식 로딩 경로 재현을 위해 유지한다.

## 다음 연구 단계

FP16·W4A16·W4A4 Linear/Conv의 전체 평가를 기준으로 구성요소별 양자화 민감도를 분석하고,
그룹·클리핑·calibration을 이용한 PTQ 개선과 정수 커널 최적화를 진행한다.
현재 W4A16 구현은 packed INT4 가중치를 레이어별로 FP16 복원해 계산한다.
실제 INT4 곱셈은 별도 `int4-w4a4` 경로의 CUDA WMMA 커널로 실행한다.
