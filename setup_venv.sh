#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly VENV_DIR="${SCRIPT_DIR}/backend/venv"
readonly REQUIREMENTS="${SCRIPT_DIR}/backend/requirements.txt"

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [options]

Setup and activate Python virtual environment for RoninNVR.

Options:
    -h, --help      Show this help message
    -r, --recreate  Remove existing venv and create fresh
EOF
}

create_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "Creating virtual environment at ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    else
        echo "Virtual environment already exists at ${VENV_DIR}"
    fi
}

install_deps() {
    echo "Installing dependencies..."
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS}"
}

main() {
    local recreate=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -r|--recreate)
                recreate=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    if [[ "${recreate}" == true ]] && [[ -d "${VENV_DIR}" ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "${VENV_DIR}"
    fi

    create_venv
    install_deps

    echo ""
    echo "Virtual environment ready!"
    echo "To activate, run:"
    echo "  source ${VENV_DIR}/bin/activate"
    echo ""
    echo "Or run backend directly with:"
    echo "  ${VENV_DIR}/bin/uvicorn app.main:app --reload"
}

main "$@"
