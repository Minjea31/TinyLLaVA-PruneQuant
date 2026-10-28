# 고정 Split 평가 가이드

근거: [TinyLLaVA Evaluation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html),
[논문](https://arxiv.org/abs/2402.14289).
논문 v1에는 POPE/ScienceQA/VQAv2/GQA/TextVQA/MM-Vet/MME와 LLaVA-W/MMBench가 포함된다.
MMMU는 현재 Factory 문서의 지원 범위에 추가된 것으로, 원논문 실험 결과와 구분한다.

| 벤치마크 | 목적 / 사용할 Split·규모 | Metric | 이미지·Annotation / 공식 출처 |
| --- | --- | --- | --- |
| POPE | 객체 hallucination, COCO val2014 random/popular/adversarial 각 3,000문항, 같은 500장 | accuracy, precision, recall, F1, yes ratio; category별 보고 | [공식 POPE output/coco](https://github.com/RUCAIBox/POPE/tree/main/output/coco), [COCO](https://cocodataset.org/#download) JPEG |
| ScienceQA | 과학 추론; test 전체 4,241 / image-only 2,017문항 | 선택지 accuracy; SQA-IMG 별도 | [공식 repo](https://github.com/lupantech/ScienceQA), problems.json, pid_splits.json, 공식 test.zip |
| TextVQA | 장면 텍스트/OCR 추론; val 5,000문항, 3,166장 | official VQA soft accuracy | [공식 데이터](https://textvqa.org/dataset/), val JSON, train_val_images, TinyLLaVA eval.zip의 OCR 포함 질문 |
| GQA | compositional visual reasoning; testdev balanced 통상 12,578문항 | balanced accuracy, official evaluator의 세부 지표 | [공식 GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), images.zip, questions1.2.zip, eval.zip; 실제 배포본 count를 재확인 |
| VQAv2 | 일반 VQA; test2015 전체 447,793문항, test-dev 별도 subset | official VQA soft accuracy, EvalAI | [공식 VQA](https://visualqa.org/download.html), COCO test2015, v2 test question ZIP; 공개 test 정답 없음 |
| MME | perception 10 + cognition 4 tasks; task별 이미지 항목 합계 1,187개 / paired yes/no 2,374문항 | 각 task accuracy + accuracy+, perception/cognition 합계 | [저자 공식 안내](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), 공식 안내가 링크한 MME ZIP과 eval_tool |
| MM-Vet | OCR/공간/지식/수학 등 통합 능력; v1 218장, 300문항 | 공식 grader의 0~100 점수 | [공식 MM-Vet](https://github.com/yuweihao/MM-Vet), mm-vet.zip + question JSON; GPT-4 grading/API 또는 외부 평가 서비스 |
| MMMU | 대학 수준 다분야; 전체 약 11.5K, dev 150 / val 900 / test 약 10.5K | official accuracy, subject별 지표 | [공식 MMMU](https://mmmu-benchmark.github.io/), [HF MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU), multi-image 포함 |

MMMU 공식 사이트는 2026-02-12에 test 정답 공개를 알렸다. 과거 외부 서버 전용 안내를
현재에도 무조건 적용하지 않는다. 이번 초기 비교 split은 val 900으로 고정하는 것이 후속 후보이며,
실제 MMMU 다운로드·image adapter·평가 실행은 이번 초기 POPE/ScienceQA 구성과 별개다.

MME 개수는 [공식 논문](https://arxiv.org/pdf/2306.13394)의 task별 수를 합산한 값이다:
coarse 4×30 + fine (147+170+200+200+200) + OCR 20 + cognition (70+20+20+20) =
1,187개 task-image 항목, 각 2문항이다. 서로 다른 task 사이의 동일 이미지 가능성과 구분하며,
추가 다운로드 시 실제 배포본의 row 수를 다시 검증한다.

## 지금 사용할 초기 평가

POPE `questions.jsonl`은 원본 세 파일의 순서를 category별로 유지하고, category+원본 id로
고유 question_id를 만든다. 모든 COCO 이미지가 실제로 존재하고 PIL로 열리는지 검사한다.
질문/라벨은 원본에서 가져오며 랜덤하게 새 평가 split을 생성하지 않는다.

ScienceQA는 원본 test split과 image-only test split을 각각 생성한다. 다운받은
`problems.json`에 train 정답도 포함되지만 Calibration source로 사용하지 않는다.
`questions_test_image.jsonl`은 **로컬 CQM letter-only 프로토콜**이다. 공개 eval.zip의
`llava_test_CQM-A.json`과 질문 문자열을 완전히 같다고 주장하지 않으며, 논문 SQA 점수와
직접 비교하려면 원래 eval.zip 질문을 별도로 사용해야 한다.
FP16/INT8/INT4 간에는 이 동일 JSONL·hash·generation 설정을 그대로 공유한다.

```bash
source scripts/env.sh
python scripts/download_data.py --group initial-eval
# GPU 복구 이후 간단한 subset smoke (정식 full score로 보고하지 않음):
python scripts/test_inference.py --benchmark pope --batch-size 8 --limit 16 --output results/batch8_validation/pope
python scripts/test_inference.py --benchmark scienceqa --batch-size 8 --limit 16 --output results/batch8_validation/scienceqa
# 전체 고정 split:
bash scripts/eval_fp16_all.sh
# 또는 개별 실행:
python scripts/test_inference.py --benchmark pope --batch-size 8
python scripts/test_inference.py --benchmark scienceqa --batch-size 8
```

Greedy decoding, batch 8, left padding, `v1` prompt, max_new_tokens 32, seed 2026, FP16, warmup 1을 기본값으로
고정한다. `run.json`이 다르면 같은 output에 섞어 실행하지 못하게 막는다.
predict JSONL로 이어받기를 지원한다. POPE의 category별 scorer는 공식 첫 문장 no/not
정규화 규칙을 따른다. ScienceQA에서 선택지 파싱 실패를 정답으로 추정하지 않고 invalid로 기록한다.

Peak allocated/reserved VRAM은 CUDA synchronize와 reset_peak_memory_stats를 사용한다.
`nvidia-smi`와 torch CUDA가 모두 정상일 때만 GPU 실행을 허용한다. 호스트에서 torch CUDA
metadata가 조회되더라도 NVML 오류가 남아 있으면 사용자 지침에 따라 차단한다.
시간은 Vision/Projector/LLM generation 전체를 포함하며 이미지 전처리와 모델 로딩은 분리한다.
warmup은 제외한다. 문항별 seconds는 batch 생성 시간을 실제 batch 크기로 나눈 처리 시간이고,
개별 요청 latency와 구분한다. batch_seconds·실제 batch_size·batch_id를 예측마다 저장하고,
summary에 generation_seconds와 fresh_samples_per_second를 기록한다.
마지막 batch는 남은 문항 수만큼 실행하며, 기본 평가는 중복 입력도 각각 생성한다.
단일 반복/기기 상태에 영향을 받는 측정이므로 본격적인 논문용 latency
비교에서는 반복 횟수와 전원/클럭·동시 프로세스 조건도 고정해야 한다.

## 다른 벤치마크의 다운로드와 공식 evaluator

```bash
bash scripts/download_remaining.sh textvqa
bash scripts/download_remaining.sh gqa
bash scripts/download_remaining.sh vqav2
bash scripts/download_remaining.sh mme
bash scripts/download_remaining.sh mmvet
bash scripts/download_remaining.sh mmmu
```

위 스크립트는 data와 공식 도구를 받는다. 모든 benchmark의 모델 adapter까지 준비됐다는
의미는 아니다. TinyLLaVA 문서에서 지정한 [eval.zip](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html)
및 MMMU.zip에는 OCR 질문, 변환 코드 등 추가 자료가 있다. 필요한 경우 해당 문서의 Google Drive
링크에서 받아 `data/eval/` 아래 지정 구조로 해제한다. 임의의 다른 mirror 질문으로 바꾸지 않는다.

원본을 건드리지 않는 official CLI용 로컬 경로 view를 만들 수 있다.

```bash
python scripts/create_local_model_view.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL_PATH="$PROJECT_ROOT/model/checkpoints/TinyLLaVA-1.5B-local"
```

예: TextVQA의 OCR 질문이 준비되었을 때 공식 추론/평가 명령은 다음과 같다.

```bash
python -m tinyllava.eval.model_vqa_loader \
  --model-path "$MODEL_PATH" \
  --question-file data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
  --image-folder data/eval/textvqa/train_images \
  --answers-file results/baseline/textvqa/answers.jsonl --conv-mode v1 --temperature 0
python -m tinyllava.eval.eval_textvqa \
  --annotation-file data/eval/textvqa/TextVQA_0.5.1_val.json \
  --result-file results/baseline/textvqa/answers.jsonl
```

동일하게 upstream `scripts/eval/gqa.sh`, `vqav2.sh`, `mme.sh`, `mmvet.sh`, `mmmu.sh`의
MODEL_PATH / MODEL_NAME / EVAL_DIR / conv-mode를 위 경로 및 `v1`으로 지정한다.
legacy 브랜치의 스크립트 이름이 다른 경우 `rg --files third_party/TinyLLaVA_Factory_legacy/scripts`
에서 대응 항목을 확인한다. shell 파일은 third_party 원본을 수정하지 말고 workspace `scripts/`
아래에 복사해서 매핑한다.

GQA는 질문 ID에 맞춰 prediction을 변환하고 `eval.py --tier testdev`로 채점한다.
VQAv2는 `convert_vqav2_for_submission.py`로 답을 변환해 공식 EvalAI에 제출한다.
MME는 pair 순서를 유지해 convert_answer_to_mme.py 이후 공식 `eval_tool/calculation.py`를 실행한다.
MM-Vet는 `convert_mmvet_for_eval.py`의 JSON을 공식 grader/서비스에 전달하며, grader 모델 버전과
API 조건이 점수에 영향을 주므로 기록해야 한다. API를 이번 작업에서 호출하거나 비용을 발생시키지 않았다.
MMMU는 공식 evaluator의 answer parsing/subject aggregation을 사용해야 하고 legacy 모델에 맞는
multi-image 입력 처리가 추가로 필요하다. 미준비 benchmark를 평가 완료로 보고하지 않는다.
