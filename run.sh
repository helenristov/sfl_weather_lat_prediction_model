#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run.sh  —  SFL Scientific Take-Home Challenge
#
# Usage:
#   bash run.sh [--ps1 <path>] [--ps2 <path>] [--output <file>]
#
# This script:
#   (a) Builds the Docker image
#   (b) Launches a container that runs the prediction pipeline
#   (c) Copies prediction_results.csv to the host working directory
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE_NAME="sfl_latitude_prediction"
CONTAINER_NAME="sfl_run"
OUTPUT_FILE="prediction_results.csv"

# Default data paths (relative to this script)
PS1_DIR="./data/PS1"
PS2_DIR="./data/PS2"

# ── Parse optional arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --ps1)    PS1_DIR="$2";    shift 2 ;;
        --ps2)    PS2_DIR="$2";    shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " SFL Scientific — Latitude Prediction Pipeline"
echo "  PS1 dir   : $PS1_DIR"
echo "  PS2 dir   : $PS2_DIR"
echo "  Output    : $OUTPUT_FILE"
echo "============================================================"

# ── (a) Build Docker image ────────────────────────────────────────────────────
echo ""
echo "[1/3] Building Docker image: $IMAGE_NAME ..."
docker build -t "$IMAGE_NAME" .

# ── (b) Run container ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] Running prediction pipeline ..."

# Remove stale container if it exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run \
    --name "$CONTAINER_NAME" \
    -v "$(realpath "$PS1_DIR")":/app/data/PS1:ro \
    -v "$(realpath "$PS2_DIR")":/app/data/PS2:ro \
    -v "$(pwd)":/app/output \
    -e PS1_DIR=/app/data/PS1 \
    -e PS2_DIR=/app/data/PS2 \
    "$IMAGE_NAME" \
    python run_pipeline.py \
        --ps1 /app/data/PS1 \
        --ps2 /app/data/PS2 \
        --output /app/output/"$OUTPUT_FILE"

# ── (c) Confirm output ────────────────────────────────────────────────────────
echo ""
echo "[3/3] Checking output ..."
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓  Results saved to: $(pwd)/$OUTPUT_FILE"
    echo ""
    echo "Preview:"
    head -5 "$OUTPUT_FILE"
else
    echo "✗  Output file not found: $OUTPUT_FILE"
    exit 1
fi

echo ""
echo "Pipeline complete."
