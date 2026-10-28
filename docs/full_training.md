# TinyLLaVA 전체 fine-tuning

`scripts/train_full.sh`는 현재 공개 TinyLLaVA-1.5B에서 Vision Encoder,
Projector, LLM 전체를 unfreeze한 이미지 SFT를 실행한다. 공개 체크포인트에서
이어 학습하며, 무작위 초기화 pretraining이나 논문의 전체 학습 recipe 재현과는 구분한다.
기존 Conda `LLAVA`와 공식 legacy 전처리·v1 assistant loss mask를 재사용한다.

## 실행

GPU를 사용할 수 있는 호스트 터미널에서 워크스페이스 루트를 기준으로 실행한다.

```bash
bash scripts/train_full.sh
```

기본값은 실제 COCO 이미지가 연결된 `llava_sft_coco1024.json`에서 사용 가능한
single-image records를 골라 1 epoch 학습한다. 이미지 729 patches를 포함해
1024 tokens를 넘거나 assistant 정답이 없는 항목은 제외하며, 사용·제외 개수를
`training_manifest.json`에 남긴다. 전체 mix665K 데이터가 준비됐다는 뜻은 아니다.

짧은 전체 모델 업데이트 확인:

```bash
bash scripts/train_full.sh --max-steps 2 --max-samples 4 --gradient-accumulation 2
```

설정 변경:

```bash
EPOCHS=3 GRAD_ACCUM=8 LEARNING_RATE=1e-5 VISION_LEARNING_RATE=1e-6 \
  bash scripts/train_full.sh

bash scripts/train_full.sh \
  --data-path data/train/annotations/llava_sft_coco1024.json \
  --image-folder data/train/llava \
  --epochs 3 --max-length 1024 --save-every 100

bash scripts/train_full.sh --help
```

`--max-steps`는 microbatch 수가 아니라 성공한 optimizer update 수다. 0이면 지정한
epoch까지 실행한다. batch는 1이며 기본 gradient accumulation은 8이다. epoch 끝의
짧은 accumulation window는 실제 microbatch 수로 나누어 학습한다.

## 8GB GPU를 위한 실행 방식

- GPU: 전체 FP16 모델의 forward/backward, Vision과 LLM의 non-reentrant gradient checkpointing.
- CPU: FP32 master weights, 누적 gradients, Transformers Adafactor의 factored optimizer state.
- backward 중 각 parameter의 gradient가 준비되면 CPU로 옮기고 GPU gradient를 해제한다.
- 모든 microbatch가 끝난 후 overflow를 검사하고 CPU에서 한 번만 업데이트한다.
  갱신된 master를 GPU 모델에 복사한다. 재계산 중 weights를 바꾸지 않는다.
- FP16 loss scale 기본값은 128이며 overflow 발생 시 전체 update를 건너뛰고 절반으로 낮춘다.
  Adafactor 자체 RMS update clipping을 쓰고 별도 global norm clipping은 적용하지 않는다.

약 1.5B 모델의 CPU master+gradient만 약 11.2 GiB다. 실행 전 여기에 3 GiB 여유를
더해 가용 RAM을 검사한다. 실제 CPU 연산·복사 때문에 GPU-only 학습보다 느릴 수 있다.
8GB 적합성은 아래 실제 검증 범위를 참고한다. 추가 패키지나 DeepSpeed/MegaTrain은 쓰지 않는다.

공식 SigLIP wrapper는 `hidden_states[-1]`를 특징으로 사용한다. 이 값은
`post_layernorm` 이전 출력이므로 해당 norm의 weight/bias 두 tensor는 원래 SFT loss에
연결되지 않는다. 모든 parameter를 unfreeze하되 이 두 tensor의 gradient 부재만 허용한다.
나머지 parameter가 loss에 연결되지 않으면 실패한다. 기준 특징 추출 동작을 바꾸지 않는다.

구현 근거: [PyTorch post-accumulate gradient hook 설명](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html),
[Transformers 4.37.2 Adafactor](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/optimizer_schedules#transformers.Adafactor).
이 스크립트는 hook에서 optimizer step을 실행하지 않고 gradient만 offload한다.

## 결과·체크포인트

실행마다 새 `results/full_training/<시각>_<PID>/`와
`logs/full_training/<시각>_<PID>.log`를 만든다. 기존 모델과 이전 실행을 덮어쓰지 않는다.

- `arguments.json`: 실행 설정.
- `training_manifest.json`: annotation hash, 사용한 이미지 hash, 길이와 제외 사유.
- `metrics.jsonl`: update별 loss, 세 구성요소의 gradient norm, overflow, 시간·VRAM.
- `status.json`: 종료 상태, 세 구성요소의 대표 가중치 실제 변경 여부.
- `final/`: main safetensors, tokenizer, 학습한 `vision_encoder-SigLIP/`, `training_state.pt`.

전체 학습된 Vision은 별도로 저장한다. legacy loader는 Vision을 별도 snapshot에서
로드하므로 main weights만 저장하면 이전 Vision을 다시 불러올 수 있기 때문이다.
`runtime.load(model_path=...)`는 체크포인트 내부 Vision을 우선 사용하며, 폴더를 옮겨도
현재 경로로 다시 매핑한다. `training_state.pt`에는 FP32 master, optimizer, 데이터 위치,
loss scale, RNG를 저장한다. 정상 종료 후 기본적으로 final 한 개만 저장하며,
`--save-every N`으로 중간 checkpoint를 추가할 수 있다. 하나당 약 9–10 GiB의 디스크가
필요하고 자동 삭제는 하지 않는다. `.incomplete` 폴더는 저장 실패·중단 상태다.

중간 저장 지점에서 정확한 optimizer state를 복원해 이어 학습:

```bash
bash scripts/train_full.sh \
  --resume-from results/full_training/<이전실행>/checkpoint-000100 \
  --epochs 3
```

같은 annotation·선택 이미지·seed·accumulation·optimizer 설정을 사용해야 한다.
이전 실행이 epoch를 완료했다면 `--epochs`를 더 크게 지정한다. `--max-steps`는 이어받은
global step을 포함한 상한이다. resume도 새 출력 폴더를 만든다. 학습 도중 중단하면
마지막으로 저장이 끝난 checkpoint에서만 복원할 수 있다. 신뢰하는 로컬 실행이 만든
`training_state.pt`를 사용한다.

optimizer 없이 weights만 시작점으로 사용할 때:

```bash
bash scripts/train_full.sh --model-path results/full_training/<이전실행>/final
```

## 검증

```bash
/home/a/anaconda3/envs/LLAVA/bin/python -m unittest discover -s tests -p 'test_train_full.py' -v
```

CPU 수치 검사는 누적 update와 FP32 Adafactor의 일치, 짧은 마지막 window,
overflow 시 전체 update 취소, optimizer 복원, 실제 legacy SigLIP→Projector→Llama의
gradient 경로를 검사한다.

2026-10-27~28, 실제 RTX 3070 Laptop / driver 595.99.02에서 다음을 확인했다.

| 항목 | 결과 |
| --- | --- |
| CPU 수치·gradient 검사 | 3개 통과 |
| 전체 모델 학습 | 실제 이미지 4개, accumulation 2, 2 optimizer steps 통과 |
| 학습 대상으로 등록한 parameters | Vision 397,747,744 / Projector 6,557,696 / LLM 1,100,048,384 |
| 역전파 | 매 update 세 구성요소 모두 유한한 nonzero gradient, 연결 누락은 위에 설명한 norm 2개만 |
| 실제 FP16 가중치 변경 | 대표 4,096개 요소 중 Vision 822 / Projector 2,199 / LLM 503개 변경 |
| Peak allocated / reserved | 3.3547 / 3.7930 GiB |
| optimizer update 시간 | 2 microbatches 포함 약 11.2–11.6초 |
| 이어 학습 | FP32 master·optimizer·RNG·데이터 위치 복원 후 global step 2→3 통과 |
| 최종 재로딩 | 626개 parameter tensors, 총 1,504,353,824 parameters 모두 저장한 master의 FP16 변환과 정확히 일치 |
| 재로딩 후 이미지 forward | 유한한 loss 0.5828772 확인 |

원자료는 `results/full_training/validation_full_2steps/status.json`,
`results/full_training/validation_full_resume/status.json`,
`results/full_training/validation_full_resume/reload_validation.json`에 있다.
이는 짧은 기능 검증이며 전체 epoch의 수렴·벤치마크 정확도·모든 입력에서의 최대 메모리를
검증한 결과는 아니다. 검증용 checkpoint 두 개는 각각 약 9.2 GiB이며 별도 보존했다.

체크포인트 재로딩 검사를 재실행하려면 GPU가 보이는 호스트에서 실행한다.

```bash
/home/a/anaconda3/envs/LLAVA/bin/python tests/verify_full_checkpoint.py \
  results/full_training/validation_full_resume/final
```
