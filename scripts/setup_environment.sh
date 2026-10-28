#!/usr/bin/env bash
# Do not install the upstream all-in-one requirements or recreate Conda.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
python scripts/check_environment.py
python -m pip check
python -c 'import peft; print("PEFT:", peft.__version__)'
printf '%s\n' 'See docs/dependency_analysis.md before any package change.'
