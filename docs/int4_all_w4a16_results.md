# 전체 가중치 INT4 / activation FP16 — batch 8 평가 대기

2026-10-28 사용자 요청으로 이전 batch 1의 INT4 POPE 부분 예측·ScienceQA 전체 결과·비교 요약을 삭제했다.
FP16 batch 8 기준 평가는 완료됐으며, INT4 batch 8 전체 평가는 대기 중이다. 현재 확정된 INT4 batch 8 전체 점수는 없다.
전체 가중치 RTN INT4 체크포인트는 `model/quantized/TinyLLaVA-1.5B-W4A16-all-RTN-g128/`에 보존돼 있다.

```bash
# FP16 기준 평가 완료 후 실행:
bash scripts/eval_int4_all.sh
```

Vision·Projector·LLM의 모든 parameter를 W4A16으로 압축하며 fine-tuning은 적용하지 않는다.
두 전체 평가를 마친 뒤 report_int4.py가 같은 batch 8 조건과 전체 예측을 검증해 이 문서를 갱신한다.
구현·압축 범위는 [INT4 구현](int4_implementation.md), 새 기준 조건은 [FP16 평가](fp16_baseline.md)를 참고한다.
