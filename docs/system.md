# 시스템 점검 및 GPU 상태

## 2026-10-27 20:55 KST 재확인

호스트 `nvidia-smi`에서 드라이버 **595.99.02**, RTX 3070 Laptop GPU **8192 MiB**가 정상 조회된다.
기존 LLAVA의 PyTorch 2.1.2+cu121 / CUDA runtime 12.1로 실제 이미지 FP16 추론과
LoRA+Projector 4-step 학습을 통과했다. 두 학습 영역의 gradient와 parameter 업데이트를 확인했다.
최대 allocated VRAM은 3.33 GiB였다. [검증 결과 및 실행 방법](training_smoke.md)을 참고한다.

샌드박스 내부에서는 여전히 GPU 장치가 숨겨져 일반적인 driver 통신 오류가 발생한다.
이는 현재 호스트 드라이버 상태와 구분한다. 아래 내용은 복구 전 진단 기록이다.

## 복구 전 기록

실측 날짜: 2026-10-27. 원자료는 `environment.json`; GPU 호스트 진단은 아래와 같다.

| 항목 | 확인 결과 |
| --- | --- |
| OS / kernel | Ubuntu 22.04.5 LTS / 6.8.0-138-generic |
| Conda | 기존 `/home/a/anaconda3/envs/LLAVA`만 사용 |
| Python | 3.10.21 |
| torch / torchvision | 2.1.2+cu121 / 0.16.2+cu121 |
| torch CUDA runtime | 12.1 |
| 시스템 nvcc | 11.5.119 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, PCI 10de:249d |
| VRAM | 호스트 torch device properties로 8,222,998,528 bytes (약 7.66 GiB) 확인; NVML 조회는 실패 |
| RAM / 초기 available | 약 31 GiB / 약 25 GiB |
| Swap | 2 GiB, 초기 사용량 0 |
| 디스크 | 475 GiB 파티션, 초기 여유 약 291 GiB; 최종 여유는 completion_report 참조 |

## 호스트에서 재확인한 문제

샌드박스 내부에서는 `/dev/nvidia*`가 보이지 않아 `nvidia-smi`가 일반적인 통신 오류를 냈다.
그와 별도로 호스트에서 읽기 전용 재검사를 수행했고 실제로 다음 오류를 확인했다.

```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 595.99
```

로드된 `/proc/driver/nvidia/version`은 **595.91.07**,
디스크의 `modinfo nvidia`와 `libnvidia-ml.so.1`은 **595.99.02**다.
호스트의 `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`은 존재한다.
호스트 torch CUDA metadata 조회는 성공했고 `cuda.is_available()`은 True, capability는 8.6이다.
샌드박스 내부의 False와 구분한다. 사용자 지침에 따라 `nvidia-smi` 오류가 남아 있는 동안
GPU 추론/평가는 보류한다. CUDA metadata 조회 성공을 FP16 실행 성공으로 간주하지 않는다.
드라이버 업데이트 후 이전 kernel module이 메모리에 남아 있는 상황과 일치한다.
정확한 패키지 설치 이력까지 조사하거나 시스템 드라이버를 변경하지는 않았다.

사용자가 다른 작업을 저장하고 적절한 시점에 재부팅한 뒤 `nvidia-smi`와 torch CUDA를
다시 확인하는 것이 첫 복구 단계다. 이 작업에서는 재부팅, module unload, driver 삭제/설치,
시스템 CUDA 변경을 실행하지 않았다. 재부팅 후에도 버전이 다르면 관리자가 설치된
kernel module과 user-space driver 패키지를 같은 버전으로 정렬해야 한다.

일반 CUDA 12.1 Linux 호환 최소 드라이버 기준은 525.60.13이다
([NVIDIA CUDA 호환성 문서](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)).
현재 번호는 이를 넘지만 module/library 불일치 때문에 정상 사용 가능하다는 뜻은 아니다.
nvcc 11.5/torch CUDA 12.1의 차이는 추론 runtime 버전을 바꾸는 이유가 되지 않는다.
CUDA 확장 설치는 GPU 정상화 후 별도 검토한다.

```bash
source scripts/env.sh
nvidia-smi
python scripts/check_environment.py
python scripts/test_inference.py
```

GPU 오류가 있는 동안 모델·데이터 다운로드, 파일 해시 검사, 구조 분석,
CPU local load/이미지 전처리 검증을 수행한다. CPU FP16 연산 결과를 GPU baseline으로
대신 보고하지 않는다. `results/baseline/smoke/status.json`은 차단 상태를 기록한다.
