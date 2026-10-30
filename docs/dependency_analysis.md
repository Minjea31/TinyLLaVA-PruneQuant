# 기존 LLAVA 환경 분석

기존 Conda `/home/a/anaconda3/envs/LLAVA`만 사용한다. 새 환경을 생성하지 않는다.
변경 전 목록: `environment_requirements.before.txt`. 최종 목록: `environment_requirements.txt`.

기존 Python 3.10.21 / torch 2.1.2+cu121 / torchvision 0.16.2+cu121 / NumPy 1.26.4를 재사용한다.
선택 체크포인트의 `transformers_version`과 공식 `tinyllava_bench` 브랜치가 요구하는
Transformers 4.37.2 / tokenizers 0.15.1 / accelerate 0.21.0은 이미 설치되어 있다.
Factory main의 일괄 설치는 Transformers 4.40.1 등으로 바꾸므로 실행하지 않는다.
공식 legacy 소스를 PYTHONPATH로 사용해 다른 프로젝트의 editable `llava` 설치도 보존한다.

`pip check`는 성공했지만 PEFT 0.18.1 import는 `accelerate.utils.memory.clear_device_cache`
부재로 실패했다. 이는 선언된 최소 의존성 검사만으로 탐지되지 않는 API 충돌이다.
필요 변경 후보는 legacy가 지정한 `peft==0.4.0`으로 맞추는 것이다. 이 작업에서 LoRA는
사용하지 않지만 환경의 실제 import 호환성을 확보하기 위해 이 패키지만 조정한다.
torch / torchvision / transformers / accelerate는 변경하지 않는다.

현재 평가는 추가 CUDA 확장·DeepSpeed·FlashAttention을 사용하지 않는다.
초기 점검에서 시스템 nvcc 11.5와 torch 내장 CUDA runtime 12.1을 확인했다.
사전 빌드 torch 추론은 nvcc가 필요 없으며, 실제 INT4 연산용 CUDA 확장을 추가할 때는
당시의 compiler·PyTorch CUDA·커널 지원 버전을 다시 확인한다.

다운로드 성능을 위해 HF 공식 optional transport `hf-transfer==0.1.9`를 추가한다.
ML runtime 의존성을 변경하지 않는 전송 모듈이며 `--no-deps`로 설치한다.
기본 HTTP의 이미 완성된 파일은 그대로 재사용한다. fast transport를 선택하면 미완성
HTTP 파일의 일부는 다시 전송될 수 있다. 기본 이어받기 모드는 계속 제공한다.
기존 hf-xet transport는 이 장비에서 CAS reconstruction 416 오류가 발생해 기본 비활성화한다.
실제 대형 파일은 `download_model.py --parallel`로 HF API의 고정 URL/LFS SHA256을 사용해
timeout과 완료 블록 목록이 있는 range 전송으로 받았으며, 마지막에 `snapshot_download`로
전체 파일·metadata를 확정한다. HF native fast transport의 지연/재시도 기록은 logs에 보존한다.
