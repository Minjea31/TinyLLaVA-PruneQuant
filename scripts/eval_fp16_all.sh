#!/usr/bin/env bash
# Fixed FP16 baseline: both full splits, batch 8, one GPU job at a time.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${TINYLLAVA_FP16_INHIBITED:-0}" != "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    export TINYLLAVA_FP16_INHIBITED=1
    exec systemd-inhibit --what=sleep --mode=block \
        --why="TinyLLaVA FP16 batch 8 evaluation" bash "$SCRIPT_DIR/eval_fp16_all.sh" "$@"
fi
source "$SCRIPT_DIR/env.sh"
set -u
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$PROJECT_ROOT/logs/baseline_batch8"
for BENCHMARK in pope scienceqa; do
    python "$SCRIPT_DIR/test_inference.py" --precision fp16 --batch-size 8 \
        --benchmark "$BENCHMARK" "$@" \
        2>&1 | tee "$PROJECT_ROOT/logs/baseline_batch8/$BENCHMARK.log"
done
