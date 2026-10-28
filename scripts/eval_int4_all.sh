#!/usr/bin/env bash
# No fine-tuning: packed uniform INT4 weights across all three components.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${TINYLLAVA_INT4_INHIBITED:-0}" != "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    export TINYLLAVA_INT4_INHIBITED=1
    exec systemd-inhibit --what=sleep --mode=block \
        --why="TinyLLaVA INT4 evaluation: keep CUDA available until completion" \
        bash "$SCRIPT_DIR/eval_int4_all.sh" "$@"
fi
source "$SCRIPT_DIR/env.sh"
set -u
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$PROJECT_ROOT/logs/int4_all_w4a16"
for BENCHMARK in scienceqa pope; do
    python "$SCRIPT_DIR/test_inference.py" \
        --precision int4-all --benchmark "$BENCHMARK" --batch-size 8 "$@" \
        2>&1 | tee "$PROJECT_ROOT/logs/int4_all_w4a16/$BENCHMARK.log"
done
python "$SCRIPT_DIR/report_int4.py"
