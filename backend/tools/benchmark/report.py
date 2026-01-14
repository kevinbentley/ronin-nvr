"""Report generation for benchmark results."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from .analyzer import BenchmarkAnalyzer
from .models import BenchmarkResult

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Benchmark Report - {run_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{ color: #333; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; }}
        tr:hover {{ background: #f5f5f5; }}
        .metric {{
            display: inline-block;
            padding: 15px 25px;
            margin: 5px;
            background: #e9ecef;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .best {{ background: #d4edda; }}
        .good {{ color: #28a745; }}
        .bad {{ color: #dc3545; }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #2196F3);
        }}
    </style>
</head>
<body>
    <h1>Detection Benchmark Report</h1>
    <p>Generated: {generated_at}</p>

    <div class="card">
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{total_videos}</div>
                <div class="metric-label">Videos Processed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_events}</div>
                <div class="metric-label">Events Detected</div>
            </div>
            <div class="metric">
                <div class="metric-value">{true_positives}</div>
                <div class="metric-label">True Positives</div>
            </div>
            <div class="metric">
                <div class="metric-value">{false_positives}</div>
                <div class="metric-label">False Positives</div>
            </div>
            <div class="metric">
                <div class="metric-value">{best_method}</div>
                <div class="metric-label">Best Method</div>
            </div>
            <div class="metric">
                <div class="metric-value">{best_f1:.3f}</div>
                <div class="metric-label">Best F1 Score</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Method Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>TP</th>
                    <th>FP</th>
                    <th>FN</th>
                    <th>FPS</th>
                </tr>
            </thead>
            <tbody>
                {method_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Event Type Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Event Type</th>
                    <th>Total</th>
                    <th>True Positive</th>
                    <th>False Positive</th>
                    <th>Uncertain</th>
                </tr>
            </thead>
            <tbody>
                {event_type_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Camera Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Camera</th>
                    <th>Videos</th>
                    <th>Events</th>
                    <th>True Positives</th>
                    <th>False Positives</th>
                </tr>
            </thead>
            <tbody>
                {camera_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Processing Speed</h2>
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Frames Processed</th>
                    <th>Processing Time (s)</th>
                    <th>FPS</th>
                </tr>
            </thead>
            <tbody>
                {speed_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Configuration</h2>
        <pre>{config_json}</pre>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """Generates reports from benchmark results."""

    def __init__(self, result: BenchmarkResult, output_dir: Path):
        """Initialize report generator.

        Args:
            result: Benchmark result to report on
            output_dir: Directory for output files
        """
        self.result = result
        self.output_dir = output_dir
        self.analyzer = BenchmarkAnalyzer(result)
        self.analyzer.calculate_metrics()

    def generate_all(self) -> dict[str, Path]:
        """Generate all report formats.

        Returns:
            Dictionary mapping format names to output paths
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}
        outputs["json"] = self.generate_json()
        outputs["csv"] = self.generate_csv()
        outputs["html"] = self.generate_html()

        return outputs

    def generate_json(self) -> Path:
        """Generate JSON report.

        Returns:
            Path to generated JSON file
        """
        output_path = self.output_dir / f"benchmark_{self.result.run_id}.json"

        report_data = {
            "summary": self.analyzer.get_summary(),
            "method_comparison": self.analyzer.get_method_comparison(),
            "event_type_breakdown": {
                k.value: v for k, v in self.analyzer.get_event_type_breakdown().items()
            },
            "camera_breakdown": self.analyzer.get_camera_breakdown(),
            "speed_comparison": self.analyzer.get_method_speed_comparison(),
            "overlap_matrix": self.analyzer.get_detection_overlap_matrix(),
            "full_result": self.result.to_dict(),
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"JSON report saved: {output_path}")
        return output_path

    def generate_csv(self) -> Path:
        """Generate CSV report with method comparison.

        Returns:
            Path to generated CSV file
        """
        output_path = self.output_dir / f"benchmark_{self.result.run_id}.csv"

        comparison = self.analyzer.get_method_comparison()

        if comparison:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
                writer.writeheader()
                writer.writerows(comparison)

        logger.info(f"CSV report saved: {output_path}")
        return output_path

    def generate_html(self) -> Path:
        """Generate HTML report.

        Returns:
            Path to generated HTML file
        """
        output_path = self.output_dir / f"benchmark_{self.result.run_id}.html"

        summary = self.analyzer.get_summary()
        comparison = self.analyzer.get_method_comparison()
        event_breakdown = self.analyzer.get_event_type_breakdown()
        camera_breakdown = self.analyzer.get_camera_breakdown()
        speed_comparison = self.analyzer.get_method_speed_comparison()

        # Build method rows
        method_rows = []
        best_f1 = summary.get("best_f1_score", 0)
        for m in comparison:
            is_best = abs(m["f1_score"] - best_f1) < 0.001
            row_class = "best" if is_best else ""
            method_rows.append(
                f'<tr class="{row_class}">'
                f'<td>{m["method"]}</td>'
                f'<td>{m["precision"]:.3f}</td>'
                f'<td>{m["recall"]:.3f}</td>'
                f'<td><strong>{m["f1_score"]:.3f}</strong></td>'
                f'<td class="good">{m["true_positives"]}</td>'
                f'<td class="bad">{m["false_positives"]}</td>'
                f'<td>{m["false_negatives"]}</td>'
                f'<td>{m["fps"]:.1f}</td>'
                f"</tr>"
            )

        # Build event type rows
        event_type_rows = []
        for event_type, stats in event_breakdown.items():
            event_type_rows.append(
                f"<tr>"
                f"<td>{event_type.value}</td>"
                f'<td>{stats["total"]}</td>'
                f'<td class="good">{stats["true_positive"]}</td>'
                f'<td class="bad">{stats["false_positive"]}</td>'
                f'<td>{stats["uncertain"]}</td>'
                f"</tr>"
            )

        # Build camera rows
        camera_rows = []
        for camera, stats in sorted(camera_breakdown.items()):
            camera_rows.append(
                f"<tr>"
                f"<td>{camera}</td>"
                f'<td>{stats["videos"]}</td>'
                f'<td>{stats["events"]}</td>'
                f'<td class="good">{stats["true_positive"]}</td>'
                f'<td class="bad">{stats["false_positive"]}</td>'
                f"</tr>"
            )

        # Build speed rows
        speed_rows = []
        for s in speed_comparison:
            speed_rows.append(
                f"<tr>"
                f'<td>{s["method"]}</td>'
                f'<td>{s["frames_processed"]}</td>'
                f'<td>{s["processing_time_seconds"]:.1f}</td>'
                f'<td><strong>{s["fps"]:.1f}</strong></td>'
                f"</tr>"
            )

        # Format label distribution
        label_dist = summary.get("label_distribution", {})

        html = HTML_TEMPLATE.format(
            run_id=self.result.run_id,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_videos=summary.get("total_videos", 0),
            total_events=summary.get("total_events", 0),
            true_positives=label_dist.get("true_positive", 0),
            false_positives=label_dist.get("false_positive", 0),
            best_method=summary.get("best_method", "N/A"),
            best_f1=summary.get("best_f1_score", 0),
            method_rows="\n".join(method_rows),
            event_type_rows="\n".join(event_type_rows),
            camera_rows="\n".join(camera_rows),
            speed_rows="\n".join(speed_rows),
            config_json=json.dumps(self.result.config_snapshot, indent=2),
        )

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"HTML report saved: {output_path}")
        return output_path


def generate_reports(
    result: BenchmarkResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Convenience function to generate all reports.

    Args:
        result: Benchmark result
        output_dir: Output directory

    Returns:
        Dictionary of output file paths
    """
    generator = ReportGenerator(result, output_dir)
    return generator.generate_all()
