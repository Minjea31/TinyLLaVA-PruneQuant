#!/usr/bin/env bash
# Source from any working directory; always use the existing environment.
source /home/a/anaconda3/etc/profile.d/conda.sh
conda activate LLAVA
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$PROJECT_ROOT/model/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$PROJECT_ROOT/model/.cache/xet"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$PROJECT_ROOT/third_party/TinyLLaVA_Factory_legacy${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
