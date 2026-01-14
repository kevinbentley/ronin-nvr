#!/usr/bin/env bash
set -euo pipefail

# Install script for RoninNVR System Watchdog
# Creates a Python virtual environment and installs as a systemd service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="ronin-watchdog"
VENV_DIR="${SCRIPT_DIR}/venv"
USER="${USER:-$(whoami)}"

echo "=========================================="
echo "RoninNVR System Watchdog Installer"
echo "=========================================="
echo "Directory: ${SCRIPT_DIR}"
echo "User: ${USER}"
echo ""

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "${VENV_DIR}"

# Activate and install dependencies
echo "Installing dependencies..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"

echo "Dependencies installed successfully"

# Create systemd service file
SERVICE_FILE="/tmp/${SERVICE_NAME}.service"
cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=RoninNVR System Watchdog
Documentation=https://github.com/kevinbentley/ronin-nvr
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${VENV_DIR}/bin/python watchdog_daemon.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment
Environment=WATCHDOG_LLM_ENDPOINT=http://192.168.1.125:9001/v1
Environment=WATCHDOG_CHECK_INTERVAL=60
Environment=WATCHDOG_COMPOSE_PATH=$(dirname "${SCRIPT_DIR}")

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "Service file created at: ${SERVICE_FILE}"
echo ""
echo "To install the systemd service, run:"
echo ""
echo "  sudo cp ${SERVICE_FILE} /etc/systemd/system/${SERVICE_NAME}.service"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable ${SERVICE_NAME}"
echo "  sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "To check status:"
echo "  sudo systemctl status ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "To test manually (dry run):"
echo "  cd ${SCRIPT_DIR}"
echo "  ${VENV_DIR}/bin/python watchdog_daemon.py --dry-run --once -v"
echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
