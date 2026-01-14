# RoninNVR System Watchdog

An intelligent system watchdog that uses an LLM to analyze system state and take remediation actions when anomalies are detected.

## Features

- Monitors memory usage, CPU, and process counts
- Tracks FFmpeg processes (transcoding can spawn too many)
- Monitors Docker container resource usage
- Uses LLM to analyze system state and recommend actions
- Can stop Docker containers or kill runaway processes
- Safety features: rate limiting, cooldowns, protected processes
- Dry-run mode for testing

## Quick Start

```bash
# Install dependencies
cd watchdog
./install.sh

# Test with dry-run (no actions taken)
./venv/bin/python watchdog_daemon.py --dry-run --once -v

# Run continuously
./venv/bin/python watchdog_daemon.py

# Install as systemd service (see install.sh output)
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WATCHDOG_LLM_ENDPOINT` | `http://192.168.1.125:9001/v1` | LLM API endpoint |
| `WATCHDOG_LLM_MODEL` | `default` | Model name |
| `WATCHDOG_CHECK_INTERVAL` | `60` | Seconds between checks |
| `WATCHDOG_MEMORY_WARNING` | `80.0` | Memory warning threshold % |
| `WATCHDOG_MEMORY_CRITICAL` | `95.0` | Memory critical threshold % |
| `WATCHDOG_MAX_FFMPEG` | `10` | Max ffmpeg processes before warning |
| `WATCHDOG_DRY_RUN` | `false` | Don't execute actions |
| `WATCHDOG_LOG_LEVEL` | `INFO` | Logging level |

## CLI Options

```
python watchdog_daemon.py [OPTIONS]

Options:
  --dry-run           Log actions without executing
  --once              Run single check and exit
  --interval SECONDS  Check interval (default: 60)
  --endpoint URL      LLM API endpoint
  --model NAME        LLM model name
  -v, --verbose       Enable debug logging
```

## How It Works

1. **Collect**: Gathers system metrics (memory, CPU, processes, containers)
2. **Analyze**: Sends state to LLM for anomaly detection
3. **Execute**: Performs recommended actions (with safety checks)

### LLM Analysis

The LLM receives a formatted system state and returns:
- Analysis of current situation
- Severity level (normal/warning/critical)
- Recommended actions (if any)

### Safety Features

- **Rate limiting**: Max 3 actions per 10 minutes
- **Cooldown**: Same action can't repeat within 5 minutes
- **Protected processes**: postgres, systemd, sshd, etc.
- **Dry-run mode**: Test without executing
- **Fallback**: Rule-based analysis if LLM unavailable

## Actions

| Action | Description |
|--------|-------------|
| `stop_container` | `docker compose stop <service>` |
| `restart_container` | `docker compose restart <service>` |
| `kill_process` | Send SIGTERM/SIGKILL to PID |

## Logs

When running as systemd service:
```bash
journalctl -u ronin-watchdog -f
```

## Files

```
watchdog/
├── watchdog_daemon.py    # Main entry point
├── collectors.py         # System metrics
├── llm_client.py         # LLM API client
├── actions.py            # Remediation actions
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── install.sh            # Installation script
└── README.md             # This file
```
