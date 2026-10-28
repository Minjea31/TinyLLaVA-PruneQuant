# 배치별 추론 메모리 실측

2026-10-28 RTX 3070 Laptop 8 GiB에서 POPE 첫 16문항을 같은 모델·이미지·질문으로 실행했다.
greedy, max_new_tokens=32, batch별 warmup 1회, 명시적 attention mask와 left padding을 사용한다.
본 전체 평가와 별도 출력에 저장한 소규모 메모리·처리량 실험이다.

| 모델 | 배치 | GPU 전체 사용량 관측(GiB) | peak allocated(GiB) | peak reserved(GiB) | 문항/초 | 기존 배치 1과 답변 문자열 일치 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 1 | 3.47 | 2.99 | 3.21 | 2.32 | 16/16 |
| FP16 | 4 | 4.44 | 3.49 | 4.17 | 3.58 | 15/16 |
| FP16 | 8 | 6.43 | 4.13 | 6.16 | 4.23 | 14/16 |
| INT4 W4A16 | 1 | 1.67 | 1.02 | 1.40 | 1.51 | 16/16 |
| INT4 W4A16 | 4 | 2.41 | 1.42 | 2.14 | 2.90 | 16/16 |
| INT4 W4A16 | 8 | 4.30 | 2.07 | 4.04 | 3.57 | 16/16 |

GPU 전체 사용량은 각 batch의 generate가 끝난 직후 nvidia-smi로 관측한 최댓값이다.
전체 구간을 연속 샘플링한 peak는 아니다. PyTorch peak allocated/reserved는 generation 구간에서 측정한다.
전체 사용량에는 PyTorch reserved memory·CUDA context·다른 GPU 프로세스가 포함되어 allocated보다 크다.
처리량은 16 / 전체 generation 시간이며, 이미지 전처리·모델 로드·warmup·nvidia-smi 호출은 제외한다.
batch별로 CUDA cache를 비운 뒤 warmup하며, 모델 가중치는 같은 정밀도를 유지한다.

INT4는 batch 1·4·8 모두 기존 batch 1 답변과 16/16개 문자열이 같았다.
FP16은 batch 4에서 15/16개, batch 8에서 14/16개 문자열이 같았다.
이 16문항의 POPE yes/no 판정과 metric은 모든 batch에서 각 모델의 기존 batch 1 결과와 같았다.
따라서 batch를 바꿔도 모든 문항의 생성 문자열이 동일하다고 가정하지 않는다.
ScienceQA나 더 긴 입력의 최대 메모리를 검증한 결과는 아니다.

## 재현

```bash
source scripts/env.sh
python scripts/benchmark_inference_batch.py --precision fp16 --batch-sizes 1 4 8
python scripts/benchmark_inference_batch.py --precision int4-all --batch-sizes 1 4 8
```

원자료: `results/batch_probe/fp16_pope.json`, `int4_pope.json`, `int4_pope_batch8.json`.
이 표는 별도 16문항 메모리 실험의 기록이다. 이후 사용자 요청으로 배치 1 정식 평가 결과를 삭제하고 전체 평가 기본값을 배치 8로 변경했다.
