"""Detection Benchmark Framework.

Automated comparison of detection methods using Vision LLM ground truth.
"""

from .models import (
    BenchmarkResult,
    CandidateEvent,
    Detection,
    MethodMetrics,
    VideoInfo,
)

__all__ = [
    "BenchmarkResult",
    "CandidateEvent",
    "Detection",
    "MethodMetrics",
    "VideoInfo",
]
