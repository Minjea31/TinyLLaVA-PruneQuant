#!/usr/bin/env bash
# Explicit stage selection, never launch all enormous archives implicitly.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
case "${1:-help}" in
  llava-images)
    for group in coco-train gqa-images textvqa vg1 vg2; do
      python scripts/download_data.py --group "$group"
    done
    python scripts/map_data_paths.py
    ;;
  gqa|vqav2|textvqa|mmvet)
    if [ "$1" = gqa ]; then
      python scripts/download_data.py --group gqa-images
      python scripts/download_data.py --group gqa-questions
    else
      python scripts/download_data.py --group "$1"
    fi
    python scripts/download_optional_eval.py --dataset "$1"
    ;;
  mme|mmmu|ocr-vqa|sharegpt4v)
    python scripts/download_optional_eval.py --dataset "$1"
    ;;
  *)
    printf '%s\n' 'Usage: bash scripts/download_remaining.sh {llava-images|gqa|vqav2|textvqa|mmvet|mme|mmmu|ocr-vqa|sharegpt4v}'
    ;;
esac
