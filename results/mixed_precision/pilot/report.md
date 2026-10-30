# FP16 예외 기준 — 독립 calibration 검증

민감도 선정에 쓰지 않은 8개 예제에서 같은 정답 문맥·응답 위치를 비교했다.
모든 quantized 모드의 Vision·Projector·embedding·LLM을 다시 계산했다. 학습은 없다.
W4A4는 native INT4×INT4→INT32, 예외는 명시한 FP16 GEMM 경로다.
이 값은 teacher-forced calibration 지표이며 전체 벤치마크 정확도가 아니다.

| 설정 | FP16 대비 KL | 정답 토큰 NLL | FP16 대비 ΔNLL |
| --- | ---: | ---: | ---: |
| fp16 | 0.000000 | 0.951205 | +0.000000 |
| none | 0.484663 | 1.440533 | +0.489328 |
| weights | 0.442943 | 1.370334 | +0.419129 |
| activations | 0.431633 | 1.376602 | +0.425397 |
| both | 0.403227 | 1.355994 | +0.404789 |
