#!/usr/bin/env bash
set -euo pipefail

# Start ML workers using the Python virtual environment
#
# Usage:
#   ./start_ml_workers.sh          # Start with default settings
#   ./start_ml_workers.sh -w 4     # Start 4 workers
#   ./start_ml_workers.sh --help   # Show help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for virtual environment
if [[ ! -f "venv/bin/python" ]]; then
    echo "Error: Virtual environment not found. Run ./setup_venv.sh first."
    exit 1
fi

# Run the worker script with the venv Python
exec ./venv/bin/python ml_worker.py "$@"
