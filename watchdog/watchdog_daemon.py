#!/usr/bin/env python3
"""LLM-powered system watchdog daemon.

This daemon monitors system resources and processes, using an LLM to analyze
anomalies and decide on remediation actions (stopping Docker containers,
killing runaway processes).

Usage:
    python watchdog_daemon.py [--dry-run] [--once] [--interval SECONDS]

Environment variables:
    WATCHDOG_LLM_ENDPOINT   LLM API endpoint (default: http://192.168.1.125:9001/v1)
    WATCHDOG_LLM_MODEL      Model name (default: default)
    WATCHDOG_CHECK_INTERVAL Check interval in seconds (default: 60)
    WATCHDOG_DRY_RUN        Dry run mode (default: false)
    WATCHDOG_LOG_LEVEL      Log level (default: INFO)
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from config import WatchdogConfig
from collectors import SystemCollector, SystemState
from llm_client import LLMClient, AnalysisResult
from actions import ActionExecutor, ActionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("watchdog")


class WatchdogDaemon:
    """Main watchdog daemon class."""

    def __init__(self, config: WatchdogConfig) -> None:
        """Initialize the watchdog daemon."""
        self.config = config
        self.collector = SystemCollector()
        self.llm_client = LLMClient(config)
        self.executor = ActionExecutor(config)

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Statistics
        self.checks_performed = 0
        self.actions_taken = 0
        self.llm_failures = 0
        self.start_time: Optional[datetime] = None

    async def start(self) -> None:
        """Start the watchdog daemon."""
        if self._running:
            return

        self._running = True
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("RoninNVR System Watchdog Starting")
        logger.info("=" * 60)
        logger.info(f"  LLM Endpoint: {self.config.llm_endpoint}")
        logger.info(f"  Check Interval: {self.config.check_interval}s")
        logger.info(f"  Dry Run: {self.config.dry_run}")
        logger.info("=" * 60)

        # Test LLM connection
        if await self.llm_client.test_connection():
            logger.info("LLM endpoint is reachable")
        else:
            logger.warning("LLM endpoint is not reachable - will retry on each check")

        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the watchdog daemon."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("=" * 60)
        logger.info("Watchdog Daemon Stopped")
        logger.info(f"  Total checks: {self.checks_performed}")
        logger.info(f"  Actions taken: {self.actions_taken}")
        logger.info(f"  LLM failures: {self.llm_failures}")
        if self.start_time:
            runtime = datetime.now() - self.start_time
            logger.info(f"  Runtime: {runtime}")
        logger.info("=" * 60)

    async def _run_loop(self) -> None:
        """Main watchdog loop."""
        while self._running:
            try:
                await self._check()
            except Exception as e:
                logger.exception(f"Error during watchdog check: {e}")

            # Sleep until next check
            if self._running:
                try:
                    await asyncio.sleep(self.config.check_interval)
                except asyncio.CancelledError:
                    break

    async def _check(self) -> None:
        """Perform a single watchdog check."""
        self.checks_performed += 1
        logger.debug(f"Starting check #{self.checks_performed}")

        # Collect system state
        state = self.collector.collect()

        # Log summary
        logger.info(
            f"Check #{self.checks_performed}: "
            f"Memory {state.memory_percent:.1f}%, "
            f"CPU {state.cpu_percent:.1f}%, "
            f"FFmpeg processes: {len(state.ffmpeg_processes)}, "
            f"Containers: {len(state.docker_containers)}"
        )

        # Quick check for critical memory - act without LLM if needed
        if state.memory_percent >= self.config.memory_critical_percent:
            logger.warning(
                f"CRITICAL: Memory at {state.memory_percent:.1f}% - "
                f"performing emergency analysis"
            )

        # Analyze with LLM
        analysis = await self.llm_client.analyze(state)

        if analysis is None:
            self.llm_failures += 1
            logger.warning("LLM analysis failed - falling back to rule-based check")
            analysis = self._rule_based_analysis(state)

        # Log analysis
        self._log_analysis(analysis)

        # Execute actions
        if analysis and analysis.actions:
            for action in analysis.actions:
                logger.warning(f"Executing action: {action.action_type} on {action.target}")
                result = self.executor.execute(action)
                self._log_action_result(result)

                if result.success:
                    self.actions_taken += 1

    def _rule_based_analysis(self, state: SystemState) -> AnalysisResult:
        """Simple rule-based analysis as LLM fallback."""
        from llm_client import AnalysisResult, WatchdogAction

        actions = []
        severity = "normal"
        analysis_parts = []

        # Check memory
        if state.memory_percent >= self.config.memory_critical_percent:
            severity = "critical"
            analysis_parts.append(f"Critical memory usage: {state.memory_percent:.1f}%")

            # Stop transcode workers if memory is critical
            for container in state.docker_containers:
                if "transcode" in container.name.lower():
                    actions.append(WatchdogAction(
                        action_type="stop_container",
                        target="transcode-worker",
                        reason="Critical memory usage - emergency stop",
                    ))
                    break

        elif state.memory_percent >= self.config.memory_warning_percent:
            severity = "warning"
            analysis_parts.append(f"High memory usage: {state.memory_percent:.1f}%")

        # Check FFmpeg process count
        ffmpeg_count = len(state.ffmpeg_processes)
        if ffmpeg_count > self.config.max_ffmpeg_processes:
            if severity == "normal":
                severity = "warning"
            analysis_parts.append(f"Too many ffmpeg processes: {ffmpeg_count}")

            # Stop transcode workers if too many ffmpeg
            actions.append(WatchdogAction(
                action_type="stop_container",
                target="transcode-worker",
                reason=f"Too many ffmpeg processes ({ffmpeg_count})",
            ))

        if not analysis_parts:
            analysis_parts.append("System operating normally (rule-based check)")

        return AnalysisResult(
            analysis=" | ".join(analysis_parts),
            severity=severity,
            actions=actions,
            raw_response="rule-based",
            tokens_used=0,
        )

    def _log_analysis(self, analysis: Optional[AnalysisResult]) -> None:
        """Log the analysis result."""
        if analysis is None:
            return

        level = {
            "normal": logging.DEBUG,
            "warning": logging.WARNING,
            "critical": logging.ERROR,
        }.get(analysis.severity, logging.INFO)

        logger.log(level, f"Analysis [{analysis.severity.upper()}]: {analysis.analysis}")

        if analysis.actions:
            logger.warning(f"Recommended actions: {len(analysis.actions)}")
            for action in analysis.actions:
                logger.warning(f"  - {action.action_type}: {action.target} ({action.reason})")

    def _log_action_result(self, result: ActionResult) -> None:
        """Log the result of executing an action."""
        if result.success:
            prefix = "[DRY RUN] " if result.dry_run else ""
            logger.info(f"{prefix}Action succeeded: {result.message}")
        else:
            logger.error(f"Action failed: {result.message}")

    async def run_once(self) -> None:
        """Run a single check (useful for testing)."""
        self.start_time = datetime.now()
        logger.info("Running single watchdog check")
        await self._check()


async def main_async(config: WatchdogConfig, run_once: bool = False) -> None:
    """Async main entry point."""
    daemon = WatchdogDaemon(config)

    if run_once:
        await daemon.run_once()
        return

    # Set up shutdown handling
    shutdown_event = asyncio.Event()
    loop = asyncio.get_event_loop()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Start daemon
    await daemon.start()

    # Wait for shutdown
    await shutdown_event.wait()

    # Stop daemon
    await daemon.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-powered system watchdog daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without executing them",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single check and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Check interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="LLM API endpoint URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Load config from environment
    config = WatchdogConfig.from_env()

    # Override with CLI arguments
    if args.dry_run:
        config.dry_run = True
    if args.interval:
        config.check_interval = args.interval
    if args.endpoint:
        config.llm_endpoint = args.endpoint
    if args.model:
        config.llm_model = args.model
    if args.verbose:
        config.log_level = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)

    # Run
    try:
        asyncio.run(main_async(config, run_once=args.once))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
