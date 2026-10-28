# TinyLLaVA Quantization 연구 환경

TinyLLaVA의 Vision Encoder / Projector / LLM을 분석하고 FP16을 기준으로 향후 INT8·INT4,
layer-wise·channel-wise PTQ의 정확도, Peak VRAM, 추론 시간을 비교하는 워크스페이스다.
초기 환경 구축 후 드라이버 복구와 실제 이미지 LoRA+Projector 학습 smoke test까지 확인했다.
FP16 배치 8의 전체 기준 평가를 완료했고, 같은 조건의 전체 가중치 INT4 RTN 비교를 준비한다. 전체 모델 fine-tuning 코드는
추가했으며, 실제 이미지 4개로 2-step 업데이트를 확인했다. 장기 재학습은 아직 실행하지 않았다.

**기존 Conda `LLAVA`만 사용한다. 2026-10-27 드라이버 595.99.02에서 GPU 추론과 4-step 학습을 통과했다.**
검증 범위와 실측값은 [학습 확인 결과](docs/training_smoke.md)에 기록했다.
모델·데이터의 실제 완료 여부, 다운로드 바이트와 최종 여유 공간은
[완료 보고서](docs/completion_report.md)와 [데이터 검사](docs/data_inventory.json)에서 확인한다.
원본 Annotation만 받은 데이터와 실제 이미지가 연결된 데이터를 구분한다.

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
[Conda 명시 목록](docs/conda_packages.txt), [GPU 진단](docs/system.md)을 참고한다.

`pip freeze`에는 기존 ROS 및 다른 프로젝트의 editable `llava` 경로도 포함되므로 다른 기기에
그대로 일괄 설치하는 명령으로 사용하지 않는다. 이 프로젝트의 핵심 버전은
[requirements_core.txt](configs/requirements_core.txt)에 따로 기록했다. 시스템 드라이버와
PyTorch를 일괄 업그레이드하지 않는다. `third_party` 소스는 PYTHONPATH로 불러온다.

## 모델과 구조

- [bczhou/TinyLLaVA-1.5B](https://huggingface.co/bczhou/TinyLLaVA-1.5B): TinyLlama 1.1B,
  22 decoder layers, hidden 2048, GQA heads 32/4.
- [bczhou/TinyLLaVA-1.5B-SigLIP](https://huggingface.co/bczhou/TinyLLaVA-1.5B-SigLIP):
  384×384 SigLIP, 729 patches, hidden 1152, 27 layers.
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

이전 배치 1의 INT4 평가 결과는 삭제했다. FP16 배치 8 기준 평가가 완료됐으며,
같은 조건의 INT4 전체 평가는 대기 중이다. [현재 상태](docs/int4_all_w4a16_results.md)와
[구현·이어 실행 안내](docs/int4_implementation.md)를 참고한다.
FP16·INT4의 배치 1·4·8 메모리와 처리량은 별도 [배치 실측 기록](docs/batch_memory.md)에 정리했다.

## 짧은 학습 확인

```bash
source scripts/env.sh
python scripts/test_training.py --steps 4
```

실제 COCO SFT 이미지 2개와 공식 legacy 전처리를 사용한다. FP16 base에 LLM LoRA(r=8)와
FP32 Projector를 학습하고 Vision은 고정한다. batch 1, 이미지 729 patches, 총 sequence ≤1024,
gradient checkpointing 조건이다. loss, 유한한 nonzero gradient, 두 학습 영역의 실제 parameter
변경을 검사한다. 출력은 새 `results/training_smoke/<실행시각>/` 아래에 저장한다.
샌드박스에서 GPU가 숨겨진 환경이라면 GPU 접근이 가능한 호스트 터미널에서 실행한다.

검증 실행은 4 step 모두 통과했고 최대 allocated VRAM은 3.33 GiB였다. 전체 모델 fine-tuning,
Vision 학습, MegaTrain 호환성 또는 장기 수렴을 확인한 결과는 아니다.

## 전체 모델 학습

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

## MegaTrain

공식 [MegaTrain](https://github.com/DLYuanGod/MegaTrain)을 third_party에 보관했다.
CPU parameter/optimizer offload, layer streaming, double buffering, checkpointing을 검토했지만
현재 TinyLLaVA에 직접 호환되지 않는다. image sentinel 확장과 component adapter,
Vision/Projector gradient 경로 및 Transformers 버전 충돌이 남아 있다.

```bash
python scripts/check_megatrain.py
```

[코드 수준 검토](docs/megatrain.md)에 수정 필요 사항과 RAM 한계를 정리했다.
MegaTrain은 학습 후보 backend로만 분리하며 일반 추론·평가 경로에는 적용하지 않는다.
이번 작업에서 MegaTrain 학습/전체 backward를 실행하지 않았다.

## 폴더

```text
model/        TinyLLaVA-1.5B/, TinyLLaVA-1.5B-SigLIP/, quantized/, checkpoints/
data/         train/{llava,sharegpt4v,annotations}/
              eval/{pope,scienceqa,textvqa,gqa,vqav2,mme,mmvet,mmmu}/
              calibration/, manifests/
third_party/  TinyLLaVA_Factory/, TinyLLaVA_Factory_legacy/, MegaTrain/
configs/      핵심 버전과 연구 조건
scripts/      환경 검사·다운로드·구조 분석·baseline·검증
results/      baseline/ 및 향후 실험 결과
logs/         다운로드 로그
docs/         환경·모델·데이터·호환성·완료 보고서
```

`.gitignore`는 데이터셋(`data/`), 모델(`model/`), 학습 체크포인트와 가중치·학습 상태 파일을
제외한다. 공부용 `model/model_study/`와 `model/README.md`, 코드·문서·로그·평가 결과는 포함한다.
HF 모델 cache는 model 아래, datasets cache는 data 아래로 고정한다. snapshot local_dir를
사용하여 global HF hub cache에 같은 대형 weights를 중복 저장하지 않는다.
경로 alias는 symlink를 사용하며 원본 모델 안의 Vision weights와 별도 Vision snapshot의
중복은 공식 로딩 경로 재현을 위해 유지한다.

## 다음 연구 단계

배치 8 FP16 baseline 검증 완료 → 같은 배치의 전체 INT4 평가 → Calibration manifest 고정 → LLM/Projector/Vision 단위
INT8·INT4 실험 → layer-wise·channel-wise 정밀도 배분 → 동일 Split 정확도·VRAM·시간 비교.
메모리 부족 시 batch/생성 길이·context 조정, 학습용 checkpointing과 검증된 CPU offload를
단계적으로 검토한다. 전체 INT4 RTN은 구현됐으며 복구 학습은 후속 단계다.
