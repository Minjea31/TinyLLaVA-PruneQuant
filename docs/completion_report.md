# 실제 완료 보고서

검사 시각: 2026-10-27T11:11:55.156267+00:00. 기존 LLAVA 환경만 사용했다.

- [x] 시스템 환경 확인
- [x] 폴더 구조 구성
- [x] TinyLLaVA 및 별도 Vision snapshot 다운로드/해시 검증
- [x] safetensors 구조/인덱스 검증
- [x] 학습 Annotation 네 종류
- [x] 사용 가능한 학습 이미지/텍스트 연결
- [x] POPE·ScienceQA 초기 평가 데이터
- [x] Calibration 512 samples
- [x] 로컬 모델 CPU load/실제 이미지 전처리
- [x] MegaTrain 코드 호환성 검토
- [ ] FP16 GPU 추론/평가
- [x] README 및 실행 가이드

검증된 다운로드 파일 합계: **36.68 GiB** (39,390,196,176 bytes). HTTP/API 소량 metadata traffic은 제외한다.

실제 보관 model: 7.09 GiB, data (압축본+해제본 포함): 55.69 GiB. 디스크 여유: **226.13 GiB**.

데이터 파일별 출처/해시는 `data_sources.json`, 모델별 해시는 `*_snapshot.json`에 보존했다.

| Snapshot | Revision | 파일 크기 |
| --- | --- | --- |
| TinyLLaVA-1.5B | `e1fe174363732086223af5ca9dab6f4309f66278` | 5.61 GiB |
| TinyLLaVA-1.5B-SigLIP | `0a95c431e2c8636d003f5db08adf9ea7dc3b6081` | 1.48 GiB |

| 학습 Annotation | 실제 records | 이미지가 연결된 records | 상태 |
| --- | ---: | ---: | --- |
| blip_laion_cc_sbu_558k.json | 558,128 | 558,128 | ready |
| llava_v1_5_mix665k.json | 665,298 | 4,235 | partial_images |
| share-captioner_coco_lcs_sam_1246k_1107.json | 1,246,901 | 559,152 | partial_images |
| sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json | 665,058 | 7,036 | partial_images |

Annotation만 있는 records는 사용 가능한 학습 데이터로 집계하지 않았다. text-only records는 이미지 미연결과 구분한다.

- pope: {"questions": 9000, "images": 500, "split": "COCO val2014 POPE"}
- scienceqa: {"split": "test", "questions": 4241, "image_questions": 2017}

저장소 commit:

- MegaTrain: `9d88c6898d83da6a92705ebe1a16872a4bea0d3b` (수정 상태: clean)
- TinyLLaVA_Factory: `6c57c267383837d5fdbf1f8b943f2c00b680cefc` (수정 상태: clean)
- TinyLLaVA_Factory_legacy: `2d2105fc7d44781464e7f503986bbfac270a8e22` (수정 상태: clean)

## 제약과 미완료 항목

호스트 NVML 595.99 / loaded module 595.91.07 불일치가 남아 있어 사용자 지침에 따라 GPU 평가를 보류했다. 호스트 CUDA metadata 조회는 성공했고 GPU VRAM은 8,222,998,528 bytes로 확인했다. FP16 generation과 benchmark 점수는 미실행이다. 드라이버/시스템 CUDA는 변경하지 않았다. 원본 모델 FP16 메모리 크기 약 2.80 GiB는 Peak VRAM 실측값이 아니다.

전체 ShareGPT4V PT는 추가 SAM/WebData와 상당한 디스크가 필요해 보류했다. 전체 LLaVA SFT 이미지도 초기 1,024-image subset과 구분하며 추가 archive script를 제공했다. 다른 6개 benchmark는 다운로드 도구·공식 평가 절차를 제공했으나 이번 초기 실행 완료 대상으로 집계하지 않는다. MegaTrain은 TinyLLaVA의 image token/gradient 경로가 직접 호환되지 않아 학습 backend 설치/실행을 보류했다.

GPU 정상화 후 `source scripts/env.sh` → `python scripts/check_environment.py` → `python scripts/test_inference.py` → POPE/ScienceQA full baseline을 실행한다. Quantization 및 재학습은 실행하지 않았다.
