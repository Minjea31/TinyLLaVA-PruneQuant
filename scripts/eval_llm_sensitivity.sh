#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${TINYLLAVA_SENSITIVITY_INHIBITED:-0}" != "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    export TINYLLAVA_SENSITIVITY_INHIBITED=1
    exec systemd-inhibit --what=sleep --mode=block \
        --why="TinyLLaVA layer sensitivity analysis" bash "$SCRIPT_DIR/eval_llm_sensitivity.sh" "$@"
fi
source "$SCRIPT_DIR/env.sh"
set -u
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
SENSITIVITY_OUTPUT="$PROJECT_ROOT/results/sensitivity_llm"
SENSITIVITY_ARGS=("$@")
for ((SENSITIVITY_INDEX=0; SENSITIVITY_INDEX<${#SENSITIVITY_ARGS[@]}; SENSITIVITY_INDEX++)); do
    case "${SENSITIVITY_ARGS[SENSITIVITY_INDEX]}" in
        --output-dir) SENSITIVITY_OUTPUT="${SENSITIVITY_ARGS[SENSITIVITY_INDEX+1]}" ;;
        --output-dir=*) SENSITIVITY_OUTPUT="${SENSITIVITY_ARGS[SENSITIVITY_INDEX]#*=}" ;;
    esac
done
SENSITIVITY_LOG="$PROJECT_ROOT/logs/$(basename -- "$SENSITIVITY_OUTPUT")"
mkdir -p "$SENSITIVITY_LOG"
python "$SCRIPT_DIR/analyze_llm_sensitivity.py" "$@" 2>&1 | tee "$SENSITIVITY_LOG/run.log"
python "$SCRIPT_DIR/report_llm_sensitivity.py" --output-dir "$SENSITIVITY_OUTPUT" \
    2>&1 | tee -a "$SENSITIVITY_LOG/run.log"
