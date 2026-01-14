#!/usr/bin/env bash
set -euo pipefail

# Run the detection benchmark inside the GPU-enabled Docker container
# This gives access to CUDA-enabled OpenCV and ONNX Runtime

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$BACKEND_DIR")"
RESULTS_DIR="${PROJECT_DIR}/benchmark_results"
LOG_FILE="${RESULTS_DIR}/benchmark.log"

# Default arguments
ARGS="${*:-}"

# Ensure results directory exists
mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "Detection Benchmark (GPU)"
echo "========================================"
echo "Arguments: ${ARGS:-<default>}"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Results will be in: ${RESULTS_DIR}/"
echo "========================================"
echo ""

docker run --rm \
    --gpus all \
    --name ronin-benchmark \
    -v /opt3/ronin/storage:/opt3/ronin/storage:ro \
    -v /opt3/ronin/ml_models:/opt3/ronin/ml_models:ro \
    -v "${BACKEND_DIR}:/app/backend" \
    -v "${RESULTS_DIR}:/app/benchmark_results" \
    -w /app/backend \
    -e PYTHONPATH=/app/backend \
    ronin-nvr-live-detection:latest \
    python tools/run_detection_benchmark.py \
        --output /app/benchmark_results \
        --log-file /app/benchmark_results/benchmark.log \
        ${ARGS}

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Results: ${RESULTS_DIR}/"
echo "Log: ${LOG_FILE}"
echo "========================================"
