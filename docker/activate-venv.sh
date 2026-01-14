#!/usr/bin/env bash
# Activate the nextgen-dev venv
# Usage: source docker/activate-venv.sh

export PATH="/opt/venv/bin:$PATH"
export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"

echo "Activated nextgen-dev environment"
python --version
