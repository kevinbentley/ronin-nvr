"""CLI for running detection benchmarks."""

import argparse
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .analyzer import analyze_results
from .checkpoint import CheckpointManager
from .config import BenchmarkConfig
from .models import BenchmarkResult, DetectionMethod
from .orchestrator import DetectionOrchestrator
from .report import generate_reports
from .video_selector import get_video_metadata, select_videos, select_videos_by_camera
from .vlm_labeler import VLMLabeler, test_vlm_connection

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging for the benchmark."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run detection benchmark comparing multiple methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark with default settings
  python -m tools.benchmark.cli

  # Run with specific number of videos
  python -m tools.benchmark.cli --videos 20

  # Resume from checkpoint
  python -m tools.benchmark.cli --resume abc123

  # Skip VLM labeling (faster, for testing)
  python -m tools.benchmark.cli --skip-vlm

  # Test only specific methods
  python -m tools.benchmark.cli --methods yolov8n yolo11l
        """,
    )

    parser.add_argument(
        "--videos",
        type=int,
        default=50,
        help="Number of videos to process (default: 50)",
    )

    parser.add_argument(
        "--storage",
        type=Path,
        default=Path("/opt3/ronin/storage"),
        help="Path to video storage directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmark_results"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint with given run ID",
    )

    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM labeling (for faster testing)",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[m.value for m in DetectionMethod],
        help="Specific detection methods to test",
    )

    parser.add_argument(
        "--by-camera",
        action="store_true",
        help="Select videos evenly distributed across cameras",
    )

    parser.add_argument(
        "--videos-per-camera",
        type=int,
        default=10,
        help="Videos per camera when using --by-camera (default: 10)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)",
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Frame sampling rate per second (default: 1.0)",
    )

    parser.add_argument(
        "--vlm-endpoint",
        type=str,
        default="http://192.168.1.125:9001/v1/chat/completions",
        help="VLM API endpoint URL",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select videos but don't process them",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write detailed log to file (tail -f to monitor)",
    )

    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> int:
    """Run the detection benchmark.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Set up log file path
    log_file = args.log_file
    if log_file is None:
        log_file = args.output / "benchmark.log"

    setup_logging(args.verbose, log_file)
    logger.info(f"Logging to: {log_file} (tail -f to monitor)")

    # Build configuration
    config = BenchmarkConfig(
        storage_root=args.storage,
        output_dir=args.output,
        target_total_videos=args.videos,
        yolo_confidence=args.confidence,
        sample_fps=args.sample_fps,
        vlm_endpoint=args.vlm_endpoint,
    )

    # Filter methods if specified
    if args.methods:
        config.enabled_methods = [DetectionMethod(m) for m in args.methods]

    # Determine run ID
    if args.resume:
        run_id = args.resume
        logger.info(f"Resuming benchmark: {run_id}")
    else:
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting new benchmark: {run_id}")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(config, run_id)

    # Try to load existing checkpoint
    checkpoint_data = None
    processed_paths: set[str] = set()

    if args.resume:
        checkpoint_data = checkpoint_mgr.load()
        if checkpoint_data:
            result, processed_paths = checkpoint_data
            logger.info(f"Loaded checkpoint: {len(processed_paths)} videos already processed")
        else:
            logger.warning("No checkpoint found, starting fresh")

    # Initialize or resume result
    if checkpoint_data:
        result = checkpoint_data[0]
    else:
        result = BenchmarkResult(
            run_id=run_id,
            start_time=datetime.now(),
            config_snapshot=config.to_dict(),
        )

    # Select videos
    logger.info("Selecting videos...")
    if args.by_camera:
        videos = select_videos_by_camera(
            config,
            videos_per_camera=args.videos_per_camera,
            exclude_paths=processed_paths,
        )
    else:
        videos = select_videos(config, exclude_paths=processed_paths)

    if not videos:
        logger.error("No videos found matching criteria")
        return 1

    logger.info(f"Selected {len(videos)} videos for processing")

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - selected videos:")
        for v in videos[:10]:
            logger.info(f"  {v.camera_id}/{v.date_str}/{v.time_str} ({v.file_size_mb:.1f}MB)")
        if len(videos) > 10:
            logger.info(f"  ... and {len(videos) - 10} more")
        return 0

    # Test VLM connection
    if not args.skip_vlm:
        logger.info("Testing VLM connection...")
        if not test_vlm_connection(config):
            logger.error("VLM endpoint not reachable. Use --skip-vlm to skip labeling.")
            return 1

    # Initialize orchestrator
    orchestrator = DetectionOrchestrator(config)

    # Initialize VLM labeler
    vlm_labeler = VLMLabeler(config) if not args.skip_vlm else None

    # Process videos
    total = len(videos)
    for i, video in enumerate(videos):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing video {i+1}/{total}: {video.path.name}")
        logger.info(f"{'='*60}")

        try:
            # Get video metadata
            video = get_video_metadata(video)

            # Run detectors
            events = orchestrator.process_video(video)

            # Extract frames for VLM labeling
            if events and vlm_labeler:
                frame_dir = config.frame_cache_dir / run_id
                for event in events:
                    orchestrator.extract_event_frame(event, frame_dir)

                # Label with VLM
                vlm_labeler.label_events(events)

            # Update result
            result.videos_processed.append(video)
            result.candidate_events.extend(events)
            processed_paths.add(str(video.path))

            # Save checkpoint periodically
            if (i + 1) % config.checkpoint_interval == 0:
                checkpoint_mgr.save(result, processed_paths)

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Saving checkpoint...")
            checkpoint_mgr.save(result, processed_paths)
            return 130

        except Exception as e:
            logger.error(f"Error processing {video.path}: {e}")
            continue

    # Mark completion
    result.end_time = datetime.now()

    # Update processing stats from orchestrator
    stats = orchestrator.get_processing_stats()
    for method, stat in stats.items():
        if method in result.method_metrics:
            result.method_metrics[method].total_frames_processed = stat["frames_processed"]
            result.method_metrics[method].processing_time_seconds = stat["processing_time_seconds"]

    # Analyze results
    logger.info("\nAnalyzing results...")
    analyzer = analyze_results(result)

    # Print summary
    summary = analyzer.get_summary()
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*60)
    logger.info(f"Videos processed: {summary['total_videos']}")
    logger.info(f"Events detected: {summary['total_events']}")
    logger.info(f"Duration: {summary['duration_seconds']:.1f}s")
    logger.info(f"Best method: {summary['best_method']} (F1: {summary['best_f1_score']:.3f})")

    # Print method comparison
    comparison = analyzer.get_method_comparison()
    logger.info("\nMethod Comparison:")
    logger.info(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPS':<8}")
    logger.info("-" * 53)
    for m in comparison:
        logger.info(
            f"{m['method']:<15} {m['precision']:<10.3f} {m['recall']:<10.3f} "
            f"{m['f1_score']:<10.3f} {m['fps']:<8.1f}"
        )

    # Generate reports
    logger.info("\nGenerating reports...")
    reports = generate_reports(result, config.output_dir)
    logger.info(f"Reports saved to: {config.output_dir}")
    for fmt, path in reports.items():
        logger.info(f"  {fmt}: {path.name}")

    # Clean up checkpoint on success
    checkpoint_mgr.delete()

    logger.info("\nBenchmark complete!")
    return 0


def main() -> None:
    """Entry point for CLI."""
    args = parse_args()
    sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
