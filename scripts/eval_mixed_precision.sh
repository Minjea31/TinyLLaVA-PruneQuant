#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${TINYLLAVA_MIXED_INHIBITED:-0}" != "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    export TINYLLAVA_MIXED_INHIBITED=1
    exec systemd-inhibit --what=sleep --mode=block \
        --why="TinyLLaVA KL selected mixed precision evaluation" bash "$SCRIPT_DIR/eval_mixed_precision.sh" "$@"
fi
source "$SCRIPT_DIR/env.sh"
set -u
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PROJECT_ROOT/logs/mixed_precision"
python "$SCRIPT_DIR/validate_mixed_precision.py" 2>&1 | tee "$PROJECT_ROOT/logs/mixed_precision/calibration.log"
for BENCHMARK in scienceqa pope; do
    python "$SCRIPT_DIR/test_inference.py" --precision int4-mixed --mixed-variant both \
        --benchmark "$BENCHMARK" --batch-size 8 2>&1 | tee "$PROJECT_ROOT/logs/mixed_precision/$BENCHMARK.log"
done
python "$SCRIPT_DIR/report_mixed_precision.py"
