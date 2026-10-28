# MegaTrain 검토

근거: [논문](https://arxiv.org/abs/2604.05091),
[공식 코드](https://github.com/DLYuanGod/MegaTrain).
체크아웃 commit은 `environment.json`에 기록한다. `check_megatrain.py`를 다시 실행하면
현재 코드의 근거 파일·줄번호를 `megatrain_compatibility.json`으로 출력한다.

## 제공 구조

CPU RAM에 master parameters, gradient 및 optimizer state를 두고 GPU에는 현재 계산할
층을 전송한다. layer template에 weights를 bind하며 double buffering, CUDA stream,
prefetch와 gradient offload를 겹친다. `checkpoint_interval`마다 activation을 보관하고
backward에서 층을 재계산한다. CPU Adam/DeepSpeed CPUAdam을 지원한다.
따라서 RAM 용량과 PCIe 대역폭이 핵심 자원이다. CUDA와 정상 NVIDIA Driver는 여전히 필요하다.

주요 파일은 `infinity/model/cpu_master.py`, `infinity/model/mp_worker.py`,
`infinity/adapters/hf_decoder.py`, `infinity/config/training.py`, optimizer 모듈이다.
README의 VLM 지원 선언을 TinyLLaVA 지원 검증으로 해석하지 않는다.

## TinyLLaVA에 직접 적용할 수 없는 이유

1. Legacy config `model_type=tinyllava`는 HF 표준 `llava`가 아니다. 직접 모델 클래스 등록과
   로더가 필요하다. 같은 TinyLLaVA 이름의 Factory main도 state_dict 구조가 다르다.
2. 모델 구성 검색은 HF LLaVA의 `language_model`, `vision_tower`, `multi_modal_projector`를
   예상한다. Legacy는 `model.vision_tower`, `model.mm_projector`; Factory는 `connector`다.
3. TinyLLaVA의 -200 sentinel은 embedding lookup 전에 729개 patch embedding으로 확장하며
   labels에 IGNORE_INDEX를 삽입하고 attention mask/position을 늘린다. MegaTrain의
   `_merge_vision_embeddings`는 고정 위치 치환을 가정한다. 음수 sentinel과 길이 변화에 맞지 않는다.
4. `_process_vision`은 Vision과 Projector 모두 `torch.no_grad()` 아래에서 실행한다.
   이 경로는 frozen vision 특징 추출에 해당하며 share recipe의 vision/projector 역전파를 제공하지 않는다.
5. `merged[img_mask][:n_use] = ...`는 boolean indexing으로 생성한 임시 tensor를 수정하는 코드다.
   원본 hidden에 실제로 반영되는지에 대한 수정과 별도 검증이 필요하다.
6. 현재 `requirements.txt`는 Transformers ≥5.0.0을 요구한다. `setup.py`의 ≥4.30.0과도
   선언이 다르며, 기존 4.37.2 환경에서 최신 backend API 사용을 보장할 수 없다.

결론: **직접 호환되지 않음**. 현재 장비에서 TinyLLaVA MegaTrain forward/backward smoke test를
지원한다고 표시하거나 실행 가능한 것처럼 설정을 만들지 않는다. repo 보관·코드 검토까지만 수행했다.

## 후속 어댑터에 필요한 작업

정확한 model/component adapter, tokenizer/image processor와 VLM collator, 음수 sentinel
확장 및 loss mask, Vision/Projector gradient·optimizer 등록, state_dict 저장/복원과
RNG/activation recompute를 구현해야 한다. 공식 TinyLLaVA에 대해 logits/loss 및 trainable
각 구성의 gradient를 비교해야 하며, 이미지 변경 시 logits도 변하는지 확인해야 한다.
이는 이번 초기 환경 구축 범위를 넘는 후속 작업이다.

그 이후 smoke 후보는 batch 1, 실제 이미지 2개, 이미지 729 tokens를 포함한 총 sequence
1024, backward 1~2 step, checkpoint interval 2, 최소 slab 4이다. 자원 계산기 및 측정 후
결정하며, 현재 실행 가능한 MegaTrain 설정으로 취급하지 않는다.

## 메모리와 VRAM 단계

1. 추론: FP16, batch 1, greedy, max_new_tokens 32부터 Peak VRAM을 측정한다.
2. Sequence는 이미지 729 tokens를 포함한다. text budget만 줄인다고 이미지 토큰 수가 줄지는 않는다.
3. 향후 학습은 trainable 영역 축소 → sequence/batch 조정 → activation checkpointing →
   검증된 CPU parameter/optimizer offload 순으로 검토한다.
4. 약 1.504B parameters에 FP32 master/gradient/Adam m/v를 모두 두면 16 bytes/parameter,
   약 22.4 GiB이다. 실제 dtype/optimizer 구현에 따라 달라지며 추가 모델 copy, buffers,
   dataloader, activation, OS RAM이 필요하다. RAM 31 GiB/available 약 25 GiB에서는
   전체 학습이 넉넉하다고 판단할 수 없다. Swap 2 GiB는 성능 가능한 RAM의 대체가 아니다.

MegaTrain은 향후 학습 후보로만 분리한다. 일반 추론과 평가는 공식 TinyLLaVA 경로를 유지한다.
