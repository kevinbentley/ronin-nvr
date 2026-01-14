"""Metrics calculation and analysis for benchmark results."""

import logging
from collections import defaultdict

from .models import (
    BenchmarkResult,
    CandidateEvent,
    DetectionMethod,
    EventType,
    MethodMetrics,
    VLMLabel,
)

logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """Analyzes benchmark results and calculates metrics."""

    def __init__(self, result: BenchmarkResult):
        """Initialize analyzer.

        Args:
            result: Benchmark result to analyze
        """
        self.result = result

    def calculate_metrics(self) -> dict[DetectionMethod, MethodMetrics]:
        """Calculate performance metrics for each detection method.

        Returns:
            Dictionary mapping methods to their metrics
        """
        metrics: dict[DetectionMethod, MethodMetrics] = {}

        # Get all methods that were used
        all_methods = set()
        for event in self.result.candidate_events:
            all_methods.update(event.methods_that_detected)

        # Initialize metrics for each method
        for method in all_methods:
            metrics[method] = MethodMetrics(method=method)

        # Analyze each event
        for event in self.result.candidate_events:
            label = event.ground_truth_label
            if label is None or label == VLMLabel.ERROR:
                continue

            is_positive = label == VLMLabel.TRUE_POSITIVE
            methods_detected = event.methods_that_detected

            for method in all_methods:
                method_detected = method in methods_detected

                if method_detected:
                    metrics[method].total_detections += 1

                    if is_positive:
                        metrics[method].true_positives += 1
                    else:
                        metrics[method].false_positives += 1
                elif is_positive:
                    # Method missed a true positive
                    metrics[method].false_negatives += 1

        # Update result with calculated metrics
        self.result.method_metrics = metrics

        return metrics

    def get_method_comparison(self) -> list[dict]:
        """Get a comparison table of all methods.

        Returns:
            List of dictionaries with method comparison data
        """
        if not self.result.method_metrics:
            self.calculate_metrics()

        comparison = []
        for method, m in sorted(
            self.result.method_metrics.items(),
            key=lambda x: x[1].f1_score,
            reverse=True,
        ):
            comparison.append({
                "method": method.value,
                "true_positives": m.true_positives,
                "false_positives": m.false_positives,
                "false_negatives": m.false_negatives,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1_score": round(m.f1_score, 4),
                "total_detections": m.total_detections,
                "fps": round(m.fps, 2),
            })

        return comparison

    def get_event_type_breakdown(self) -> dict[EventType, dict]:
        """Break down results by event type.

        Returns:
            Dictionary mapping event types to their statistics
        """
        breakdown: dict[EventType, dict] = defaultdict(
            lambda: {"total": 0, "true_positive": 0, "false_positive": 0, "uncertain": 0}
        )

        for event in self.result.candidate_events:
            for event_type in event.event_types_detected:
                breakdown[event_type]["total"] += 1

                label = event.ground_truth_label
                if label == VLMLabel.TRUE_POSITIVE:
                    breakdown[event_type]["true_positive"] += 1
                elif label == VLMLabel.FALSE_POSITIVE:
                    breakdown[event_type]["false_positive"] += 1
                elif label == VLMLabel.UNCERTAIN:
                    breakdown[event_type]["uncertain"] += 1

        return dict(breakdown)

    def get_camera_breakdown(self) -> dict[str, dict]:
        """Break down results by camera.

        Returns:
            Dictionary mapping camera IDs to their statistics
        """
        breakdown: dict[str, dict] = defaultdict(
            lambda: {"videos": 0, "events": 0, "true_positive": 0, "false_positive": 0}
        )

        # Count videos per camera
        for video in self.result.videos_processed:
            breakdown[video.camera_id]["videos"] += 1

        # Count events per camera
        for event in self.result.candidate_events:
            camera_id = event.video.camera_id
            breakdown[camera_id]["events"] += 1

            label = event.ground_truth_label
            if label == VLMLabel.TRUE_POSITIVE:
                breakdown[camera_id]["true_positive"] += 1
            elif label == VLMLabel.FALSE_POSITIVE:
                breakdown[camera_id]["false_positive"] += 1

        return dict(breakdown)

    def get_detection_overlap_matrix(self) -> dict[str, dict[str, int]]:
        """Calculate overlap between detection methods.

        Returns:
            Matrix showing how often methods agree
        """
        all_methods = set()
        for event in self.result.candidate_events:
            all_methods.update(event.methods_that_detected)

        methods_list = sorted(all_methods, key=lambda m: m.value)

        matrix: dict[str, dict[str, int]] = {}
        for m1 in methods_list:
            matrix[m1.value] = {}
            for m2 in methods_list:
                matrix[m1.value][m2.value] = 0

        # Count co-occurrences
        for event in self.result.candidate_events:
            methods = event.methods_that_detected
            for m1 in methods:
                for m2 in methods:
                    matrix[m1.value][m2.value] += 1

        return matrix

    def get_summary(self) -> dict:
        """Get a summary of the benchmark results.

        Returns:
            Summary dictionary
        """
        if not self.result.method_metrics:
            self.calculate_metrics()

        # Count label distribution
        labels = [e.ground_truth_label for e in self.result.candidate_events]
        label_counts = {
            "true_positive": sum(1 for l in labels if l == VLMLabel.TRUE_POSITIVE),
            "false_positive": sum(1 for l in labels if l == VLMLabel.FALSE_POSITIVE),
            "uncertain": sum(1 for l in labels if l == VLMLabel.UNCERTAIN),
            "error": sum(1 for l in labels if l == VLMLabel.ERROR),
            "unlabeled": sum(1 for l in labels if l is None),
        }

        # Find best method by F1
        best_method = None
        best_f1 = 0.0
        for method, m in self.result.method_metrics.items():
            if m.f1_score > best_f1:
                best_f1 = m.f1_score
                best_method = method

        return {
            "run_id": self.result.run_id,
            "duration_seconds": self.result.duration_seconds,
            "total_videos": self.result.total_videos,
            "total_events": self.result.total_events,
            "label_distribution": label_counts,
            "best_method": best_method.value if best_method else None,
            "best_f1_score": round(best_f1, 4),
            "methods_tested": len(self.result.method_metrics),
        }

    def get_false_positive_analysis(self) -> list[dict]:
        """Analyze false positive events.

        Returns:
            List of false positive event details
        """
        false_positives = []

        for event in self.result.candidate_events:
            if event.ground_truth_label != VLMLabel.FALSE_POSITIVE:
                continue

            false_positives.append({
                "video": str(event.video.path),
                "frame": event.frame_number,
                "timestamp": event.timestamp_seconds,
                "methods": [m.value for m in event.methods_that_detected],
                "event_types": [t.value for t in event.event_types_detected],
                "vlm_response": event.vlm_response,
                "vlm_detected": event.vlm_detected_objects,
            })

        return false_positives

    def get_method_speed_comparison(self) -> list[dict]:
        """Compare processing speeds of all methods.

        Returns:
            List of speed comparison data
        """
        speeds = []

        for method, m in self.result.method_metrics.items():
            speeds.append({
                "method": method.value,
                "frames_processed": m.total_frames_processed,
                "processing_time_seconds": round(m.processing_time_seconds, 2),
                "fps": round(m.fps, 2),
            })

        return sorted(speeds, key=lambda x: x["fps"], reverse=True)


def analyze_results(result: BenchmarkResult) -> BenchmarkAnalyzer:
    """Convenience function to create and run analyzer.

    Args:
        result: Benchmark result to analyze

    Returns:
        Initialized analyzer with calculated metrics
    """
    analyzer = BenchmarkAnalyzer(result)
    analyzer.calculate_metrics()
    return analyzer
