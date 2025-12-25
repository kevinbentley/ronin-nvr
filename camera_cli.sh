#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly API_BASE="${API_BASE:-http://localhost:8000/api}"

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} <command> [options]

Camera management CLI for RoninNVR.

Commands:
    list                        List all cameras
    get <id>                    Get camera by ID
    add <name> <host> [opts]    Add a new camera
    update <id> [opts]          Update a camera
    delete <id>                 Delete a camera
    test <id>                   Test camera connection

Recording Commands:
    rec-start <id>              Start recording for camera
    rec-stop <id>               Stop recording for camera
    rec-status <id>             Get recording status for camera
    rec-status-all              Get recording status for all cameras

Storage Commands:
    storage-stats               Get storage statistics
    storage-cleanup             Run retention cleanup

Add/Update Options:
    --host <host>               Camera IP or hostname
    --port <port>               RTSP port (default: 554)
    --path <path>               RTSP path (default: /cam/realmonitor)
    --user <username>           Username for authentication
    --pass <password>           Password for authentication
    --transport <tcp|udp>       Transport protocol (default: tcp)
    --recording <true|false>    Enable recording (default: true)

Environment:
    API_BASE                    API base URL (default: http://localhost:8000/api)

Examples:
    ${SCRIPT_NAME} list
    ${SCRIPT_NAME} add "Front Door" 192.168.1.100
    ${SCRIPT_NAME} add "Backyard" 192.168.1.101 --user admin --pass secret --port 8554
    ${SCRIPT_NAME} get 1
    ${SCRIPT_NAME} test 1
    ${SCRIPT_NAME} update 1 --recording false
    ${SCRIPT_NAME} delete 1
    ${SCRIPT_NAME} rec-start 1
    ${SCRIPT_NAME} rec-stop 1
    ${SCRIPT_NAME} rec-status 1
    ${SCRIPT_NAME} rec-status-all
    ${SCRIPT_NAME} storage-stats
    ${SCRIPT_NAME} storage-cleanup
EOF
}

# Check if jq is available for pretty printing
if command -v jq &> /dev/null; then
    json_format() { jq .; }
else
    json_format() { cat; }
fi

list_cameras() {
    curl -s "${API_BASE}/cameras" | json_format
}

get_camera() {
    local id="$1"
    curl -s "${API_BASE}/cameras/${id}" | json_format
}

add_camera() {
    local name="$1"
    local host="$2"
    shift 2

    local port=554
    local path="/cam/realmonitor"
    local username=""
    local password=""
    local transport="tcp"
    local recording="true"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            --path) path="$2"; shift 2 ;;
            --user) username="$2"; shift 2 ;;
            --pass) password="$2"; shift 2 ;;
            --transport) transport="$2"; shift 2 ;;
            --recording) recording="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    local json
    json=$(cat <<EOJSON
{
    "name": "${name}",
    "host": "${host}",
    "port": ${port},
    "path": "${path}",
    "transport": "${transport}",
    "recording_enabled": ${recording}
EOJSON
)

    if [[ -n "${username}" ]]; then
        json="${json}, \"username\": \"${username}\""
    fi
    if [[ -n "${password}" ]]; then
        json="${json}, \"password\": \"${password}\""
    fi
    json="${json}}"

    curl -s -X POST "${API_BASE}/cameras" \
        -H "Content-Type: application/json" \
        -d "${json}" | json_format
}

update_camera() {
    local id="$1"
    shift

    local json="{"
    local first=true

    while [[ $# -gt 0 ]]; do
        if [[ "${first}" != "true" ]]; then
            json="${json},"
        fi
        first=false

        case "$1" in
            --name) json="${json} \"name\": \"$2\""; shift 2 ;;
            --host) json="${json} \"host\": \"$2\""; shift 2 ;;
            --port) json="${json} \"port\": $2"; shift 2 ;;
            --path) json="${json} \"path\": \"$2\""; shift 2 ;;
            --user) json="${json} \"username\": \"$2\""; shift 2 ;;
            --pass) json="${json} \"password\": \"$2\""; shift 2 ;;
            --transport) json="${json} \"transport\": \"$2\""; shift 2 ;;
            --recording) json="${json} \"recording_enabled\": $2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    json="${json} }"

    curl -s -X PUT "${API_BASE}/cameras/${id}" \
        -H "Content-Type: application/json" \
        -d "${json}" | json_format
}

delete_camera() {
    local id="$1"
    local response
    local http_code

    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "${API_BASE}/cameras/${id}")

    if [[ "${http_code}" == "204" ]]; then
        echo "Camera ${id} deleted successfully"
    elif [[ "${http_code}" == "404" ]]; then
        echo "Camera ${id} not found"
        exit 1
    else
        echo "Failed to delete camera ${id} (HTTP ${http_code})"
        exit 1
    fi
}

test_camera() {
    local id="$1"
    curl -s -X POST "${API_BASE}/cameras/${id}/test" | json_format
}

start_recording() {
    local id="$1"
    curl -s -X POST "${API_BASE}/cameras/${id}/recording/start" | json_format
}

stop_recording() {
    local id="$1"
    curl -s -X POST "${API_BASE}/cameras/${id}/recording/stop" | json_format
}

get_recording_status() {
    local id="$1"
    curl -s "${API_BASE}/cameras/${id}/recording/status" | json_format
}

get_all_recording_status() {
    curl -s "${API_BASE}/cameras/recording/status" | json_format
}

get_storage_stats() {
    curl -s "${API_BASE}/storage/stats" | json_format
}

run_storage_cleanup() {
    curl -s -X POST "${API_BASE}/storage/cleanup" | json_format
}

main() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi

    local command="$1"
    shift

    case "${command}" in
        list)
            list_cameras
            ;;
        get)
            if [[ $# -lt 1 ]]; then
                echo "Error: get requires camera ID"
                exit 1
            fi
            get_camera "$1"
            ;;
        add)
            if [[ $# -lt 2 ]]; then
                echo "Error: add requires name and host"
                echo "Usage: ${SCRIPT_NAME} add <name> <host> [options]"
                exit 1
            fi
            add_camera "$@"
            ;;
        update)
            if [[ $# -lt 2 ]]; then
                echo "Error: update requires camera ID and at least one option"
                echo "Usage: ${SCRIPT_NAME} update <id> [options]"
                exit 1
            fi
            update_camera "$@"
            ;;
        delete)
            if [[ $# -lt 1 ]]; then
                echo "Error: delete requires camera ID"
                exit 1
            fi
            delete_camera "$1"
            ;;
        test)
            if [[ $# -lt 1 ]]; then
                echo "Error: test requires camera ID"
                exit 1
            fi
            test_camera "$1"
            ;;
        rec-start)
            if [[ $# -lt 1 ]]; then
                echo "Error: rec-start requires camera ID"
                exit 1
            fi
            start_recording "$1"
            ;;
        rec-stop)
            if [[ $# -lt 1 ]]; then
                echo "Error: rec-stop requires camera ID"
                exit 1
            fi
            stop_recording "$1"
            ;;
        rec-status)
            if [[ $# -lt 1 ]]; then
                echo "Error: rec-status requires camera ID"
                exit 1
            fi
            get_recording_status "$1"
            ;;
        rec-status-all)
            get_all_recording_status
            ;;
        storage-stats)
            get_storage_stats
            ;;
        storage-cleanup)
            run_storage_cleanup
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "Unknown command: ${command}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
