#!/usr/bin/env bash
# Full Vision + Projector + LLM fine-tuning in the existing LLAVA environment.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
set -u
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec python "$SCRIPT_DIR/train_full.py" --help
fi

# Unique run directories; user-supplied Python options override these defaults.
FULL_TRAIN_RUN="${OUTPUT_DIR:-$PROJECT_ROOT/results/full_training/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$PROJECT_ROOT/logs/full_training"
FULL_TRAIN_LOG="$PROJECT_ROOT/logs/full_training/$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'Log: %s\n' "$FULL_TRAIN_LOG"
python "$SCRIPT_DIR/train_full.py" \
    --output "$FULL_TRAIN_RUN" \
    --epochs "${EPOCHS:-1}" \
    --gradient-accumulation "${GRAD_ACCUM:-8}" \
    --learning-rate "${LEARNING_RATE:-1e-5}" \
    --vision-learning-rate "${VISION_LEARNING_RATE:-1e-6}" \
    --max-length "${MAX_LENGTH:-1024}" \
    "$@" 2>&1 | tee "$FULL_TRAIN_LOG"
