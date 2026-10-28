# 학습 데이터, 경로와 다운로드 정책

근거: [TinyLLaVA 데이터 준비 문서](https://tinyllava-factory.readthedocs.io/en/latest/Prepare%20Datasets.html),
[LLaVA Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain),
[LLaVA SFT](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K),
[ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V).

## 원논문 Annotation

| 파일 | 공식 바이트 크기 | 데이터 의미 |
| --- | ---: | --- |
| blip_laion_cc_sbu_558k.json | 180,506,011 | LAION-CC-SBU 558K alignment |
| llava_v1_5_mix665k.json | 1,029,887,963 | LLaVA 665K visual instructions + text-only data |
| share-captioner_coco_lcs_sam_1246k_1107.json | 1,492,479,912 | Share-Captioner 1.246M pretraining |
| sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json | 1,230,418,359 | ShareGPT4V 665K SFT |

네 파일은 `data/train/annotations/`에 원본 이름으로 저장한다. 고정 HF revision은
`scripts/download_data.py` 및 파일별 `.receipt.json`에, SHA256/바이트 수는 receipt에 기록한다.
최종 실제 row 수와 image 연결 수는 `data_inventory.json`에 기록한다.

## 이미지 구성

LLaVA 558K의 공식 `images.zip`은 **27,356,108,382 bytes (25.48 GiB)**다.
다운로드 전에 압축본 약 25.5 GiB와 JPEG 중심 해제본 약 26~30 GiB, 모델·Annotation,
10 GiB 이상의 여유를 고려했다. ZIP을 확보한 뒤 실제 central directory의 uncompressed
size로 압축 해제 공간을 다시 검사한다. 전체 SHA256은 HF LFS 값과 비교하고 해제 시 CRC를 확인한다.

이미지 경로는 `data/train/llava/llava_pretrain/images/<annotation.image>`다.
원본 ZIP은 숫자 디렉터리로 시작하며 `images/` 상위 폴더를 포함하지 않는다. 스크립트는
압축 해제 대상을 명시적으로 위 `images/`로 지정한다. 실제 해제 용량은 27,753,772,218 bytes다.
ShareGPT4V가 참조하는 LAION 부분은 symlink로 같은 파일을 사용한다.

초기 SFT 실험용으로 원본 LLaVA 665K에서 COCO train2017 이미지가 있는 서로 다른
1,024개 샘플을 seed 2026으로 reservoir sampling한다. POPE COCO image IDs는 제외한다.
공식 COCO 이미지 서버의 같은 S3 버킷에서 실제 JPEG를 다운로드하고 PIL로 검증한다.
원본 대화·image 상대 경로를 유지한 `llava_sft_coco1024.json`을 따로 저장한다.
이 부분집합은 **전체 665K SFT 데이터가 아니다**.

COCO의 `https://images.cocodataset.org/`는 실측 시 hostname TLS mismatch가 발생했다.
인증서 검증을 끄지 않고 같은 bucket의 `https://s3.amazonaws.com/images.cocodataset.org/`
경로로 접근한다. POPE도 같은 방식으로 실제 필요한 val2014 이미지만 받는다.

| 전체 SFT에 추가로 필요한 원본 | 다운로드/설명 | 로컬 경로 | 이번 정책 |
| --- | --- | --- | --- |
| COCO train2017 | [COCO](https://cocodataset.org/#download) | llava/coco/train2017 | 1,024-image subset 우선; 전체 ZIP script 제공 |
| GQA images | [공식 images.zip](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) | llava/gqa/images | 전체 archive script 제공 |
| OCR-VQA | [공식 페이지](https://ocr-vqa.github.io/)의 원본 URL 다운로드 도구 | llava/ocr_vqa/images | 개별 외부 이미지 URL 수집 필요; script 안내 제공 |
| TextVQA | [train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) | llava/textvqa/train_images | script 제공, eval과 공유 |
| Visual Genome | [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) | llava/vg/VG_100K, VG_100K_2 | script 제공 |
| ShareGPT4V SAM | [공식 ShareGPT4V 안내](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) | sharegpt4v/sam/images | 전체 PT는 SAM tar 000000~000050까지 추가 필요; 수백 GiB~TB 규모로 현 여유 공간에서 전체 준비 보류 |
| ShareGPT4V WebData / Share_TextVQA | 위 공식 안내의 Google Drive 자료 | sharegpt4v/web-*, wikiart, share_textvqa | 외부 수집 및 academic usage 조건; 보류 |

전체 ShareGPT4V PT의 이미지가 모두 준비된 것으로 보고하지 않는다. SAM 9K subset은
SFT용 일부만 충족하며 1.246M PT 전체를 대체하지 않는다. 선정 checkpoint의 Recipe는
`model_architecture.md`에서 근거와 한계를 구분한다.

```bash
source scripts/env.sh
python scripts/download_data.py --group annotations
python scripts/download_data.py --group pretrain
python scripts/download_data.py --group training-subset
python scripts/map_data_paths.py
# 전체 LLaVA SFT 이미지 archive를 추가할 때:
bash scripts/download_remaining.sh llava-images
bash scripts/download_remaining.sh ocr-vqa
# 별도 SAM/WebData 취득 안내:
bash scripts/download_remaining.sh sharegpt4v
```

Archive 재실행 시 검증된 완성 파일을 재사용한다. large pretrain archive는 64 MiB range
완료 목록을 보존한다. HTTP 부분 파일, HF local-dir metadata, CRC extraction marker와
SHA256 receipt가 이어받기와 검증의 근거다. 정상 원본을 다시 다운로드하거나 덮어쓰지 않는다.
실패 원인은 `logs/`에 남는다. `data/manifests/failure_*`는 과거 실패일 수 있으므로 최종
`data_inventory.json` 및 각 성공 manifest와 함께 확인한다.

## 공식 학습 경로 매핑

LLaVA pretrain은 annotation + `image_folder=data/train/llava/llava_pretrain/images`;
LLaVA SFT는 `image_folder=data/train/llava`;
ShareGPT4V는 `image_folder=data/train/sharegpt4v`를 사용한다.
`map_data_paths.py`가 source hierarchy alias를 만들며 실제 이미지는 중복 복사하지 않는다.
Calibration JSON의 image 경로는 workspace 상대 경로이므로 그 파일을 사용할 때는
`image_folder=<workspace 절대경로>`로 지정한다. 다른 upstream shell의 하드코딩된
`IMAGE_PATH`, `DATA_PATH`에는 이 매핑을 대입한다. 이번 작업은 재학습을 시작하지 않는다.

## PTQ Calibration

`prepare_calibration.py --seed 2026 --count 512`는 존재하는 원본 train 이미지/대화만 사용한다.
LLaVA/ShareGPT4V pretrain 및 SFT subset을 source와 text character bin
(≤256, ≤512, ≤1024, >1024)으로 나누어 reservoir를 만들고 strata를 순회해 선택한다.
같은 이미지 SHA256은 중복 선택하지 않으며 다운로드한 POPE/ScienceQA 평가 이미지의
SHA256도 제외한다. 모든 평가 데이터에 대한 전 세계적 중복 부재를 주장하지 않는다.
최종 source별 수, token 길이, 이미지 크기, 원본 id/hash는 `data/manifests/calibration.json`에 기록한다.

고정된 Calibration manifest를 실험 도중 다시 생성하지 않는다. tokenizer·split·seed·sample
목록을 baseline/INT8/INT4에 똑같이 적용하고 평가 데이터는 Calibration에 사용하지 않는다.
Quantization 알고리즘은 구현하지 않았다.
