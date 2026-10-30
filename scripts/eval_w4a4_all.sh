#!/usr/bin/env bash
# True INT4 weighted GEMM; non-GEMM operations retain original precision.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${TINYLLAVA_W4A4_INHIBITED:-0}" != "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    export TINYLLAVA_W4A4_INHIBITED=1
    exec systemd-inhibit --what=sleep --mode=block \
        --why="TinyLLaVA native INT4 W4A4 evaluation" bash "$SCRIPT_DIR/eval_w4a4_all.sh" "$@"
fi
source "$SCRIPT_DIR/env.sh"
set -u
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PROJECT_ROOT/logs/int4_all_w4a4"
for BENCHMARK in scienceqa pope; do
    python "$SCRIPT_DIR/test_inference.py" --precision int4-w4a4 --benchmark "$BENCHMARK" \
        --batch-size 8 "$@" 2>&1 | tee "$PROJECT_ROOT/logs/int4_all_w4a4/$BENCHMARK.log"
done
python "$SCRIPT_DIR/report_w4a4.py"
