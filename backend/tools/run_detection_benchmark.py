#!/usr/bin/env python3
"""Entry point for running detection benchmarks.

This script compares multiple detection methods (YOLO, MOG2, frame differencing, etc.)
on security camera footage and uses a Vision LLM to establish ground truth.

Usage:
    python run_detection_benchmark.py [options]

Examples:
    # Run full benchmark with 50 videos
    python run_detection_benchmark.py

    # Quick test with 5 videos, no VLM
    python run_detection_benchmark.py --videos 5 --skip-vlm

    # Resume interrupted benchmark
    python run_detection_benchmark.py --resume abc123

    # Test specific methods
    python run_detection_benchmark.py --methods yolov8n yolo11l --videos 10

For full options:
    python run_detection_benchmark.py --help
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.benchmark.cli import main

if __name__ == "__main__":
    main()
