# 드라이버 복구 후 학습 확인

2026-10-27 20:55 KST, 기존 Conda `LLAVA`에서 실제 GPU 실행으로 확인했다.
드라이버/패키지 변경 없이 진행했다.

| 항목 | 결과 |
| --- | --- |
| GPU / driver | RTX 3070 Laptop 8192 MiB / 595.99.02 |
| PyTorch / CUDA runtime | 2.1.2+cu121 / 12.1 |
| FP16 이미지 추론 | 통과, 8-token 요청, 0.688초 (warmup 0) |
| 학습 | 실제 COCO SFT 이미지 2개, batch 1, AdamW 4 step |
| 학습 영역 | LLM q/v LoRA r=8: 1,126,400 parameters; Projector: 6,557,696 parameters |
| dtype | base FP16; trainable parameters FP32; FP16 autocast + GradScaler |
| 입력 | 공식 v1 전처리/assistant loss mask; 이미지 729 patches, 총 866/861 tokens |
| 메모리 설정 | gradient checkpointing, sequence limit 1024, frozen Vision |
| learning rate / seed | 1e-4 / 2026 |
| 샘플 1 loss (학습 전 → 후, eval mode) | 1.506135 → 1.116451 |
| 샘플 2 loss (학습 전 → 후, eval mode) | 1.240571 → 0.981148 |
| 역전파 | 매 step LoRA와 Projector 모두 유한한 nonzero gradient |
| 가중치 변경 | LoRA 88/88 tensors, Projector 4/4 tensors |
| Peak CUDA allocated / reserved | 3.3266 / 3.4473 GiB |
| step 소요 시간 | 0.61–0.74초 (gradient 검사 포함) |

원자료:

- [학습 상태·loss·gradient·샘플 해시](../results/driver_recheck/training/status.json)
- [학습 로그](../logs/driver_recheck_training.log)
- [추론 결과](../results/driver_recheck/inference/summary.json)
- [추론 로그](../logs/driver_recheck_inference.log)

LoRA adapter와 Projector는 `results/driver_recheck/training/checkpoint/`에 별도 저장했다.
원본 모델과 학습 annotation은 수정하지 않았다. 체크포인트 재로딩은 이번 검사 범위에 포함하지 않았다.
이 실행은 학습 경로와 업데이트가 작동하는지 확인하는 짧은 검사다.
전체 모델 fine-tuning, Vision 학습, 장기 수렴/일반화, MegaTrain 실행과 전체 평가 점수는 미검증이다.
MegaTrain의 기존 호환성 제약은 [별도 검토](megatrain.md)를 참고한다.

호스트 터미널에서 재실행:

```bash
source scripts/env.sh
python scripts/test_training.py --steps 4
```

새 `results/training_smoke/<실행시각>/`에 기록한다. GPU가 숨겨진 샌드박스에서는 CUDA를 사용할 수
없으므로 호스트 실행이 필요하다. `scripts/test_training.py`는 기존 모델/전처리/PEFT를 사용하며
학습 데이터를 추가 다운로드하지 않는다.
