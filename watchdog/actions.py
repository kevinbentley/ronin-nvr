"""Action executor for the watchdog daemon."""

import logging
import os
import signal
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config import WatchdogConfig
from llm_client import WatchdogAction

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of executing an action."""

    success: bool
    action: WatchdogAction
    message: str
    dry_run: bool = False


@dataclass
class ActionHistory:
    """Tracks action history for rate limiting."""

    # Map of action key -> list of timestamps
    actions: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, action_key: str) -> None:
        """Record an action execution."""
        self.actions[action_key].append(time.time())

    def count_recent(self, action_key: str, window_seconds: float) -> int:
        """Count actions in the recent time window."""
        cutoff = time.time() - window_seconds
        self.actions[action_key] = [t for t in self.actions[action_key] if t > cutoff]
        return len(self.actions[action_key])

    def last_execution(self, action_key: str) -> Optional[float]:
        """Get timestamp of last execution, or None if never executed."""
        times = self.actions.get(action_key, [])
        return max(times) if times else None


class ActionExecutor:
    """Executes remediation actions."""

    def __init__(self, config: WatchdogConfig) -> None:
        """Initialize the action executor."""
        self.config = config
        self.history = ActionHistory()
        self.protected_processes = set(config.protected_processes)

    def execute(self, action: WatchdogAction) -> ActionResult:
        """Execute a single action."""
        # Build action key for rate limiting
        action_key = f"{action.action_type}:{action.target}"

        # Check rate limiting
        if not self._check_rate_limit(action_key):
            return ActionResult(
                success=False,
                action=action,
                message="Rate limit exceeded - too many recent actions",
                dry_run=self.config.dry_run,
            )

        # Check cooldown
        if not self._check_cooldown(action_key):
            return ActionResult(
                success=False,
                action=action,
                message="Cooldown active - action was executed recently",
                dry_run=self.config.dry_run,
            )

        # Execute based on action type
        if action.action_type == "stop_container":
            result = self._stop_container(action)
        elif action.action_type == "restart_container":
            result = self._restart_container(action)
        elif action.action_type == "kill_process":
            result = self._kill_process(action)
        else:
            result = ActionResult(
                success=False,
                action=action,
                message=f"Unknown action type: {action.action_type}",
                dry_run=self.config.dry_run,
            )

        # Record successful actions
        if result.success:
            self.history.record(action_key)

        return result

    def _check_rate_limit(self, action_key: str) -> bool:
        """Check if we're within rate limits."""
        # Count all actions in the last 10 minutes
        total_recent = sum(
            self.history.count_recent(key, 600)
            for key in self.history.actions
        )
        return total_recent < self.config.max_actions_per_10min

    def _check_cooldown(self, action_key: str) -> bool:
        """Check if cooldown has elapsed for this specific action."""
        last = self.history.last_execution(action_key)
        if last is None:
            return True
        elapsed = time.time() - last
        return elapsed >= self.config.action_cooldown_seconds

    def _stop_container(self, action: WatchdogAction) -> ActionResult:
        """Stop a Docker container using docker compose."""
        container_pattern = str(action.target)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would stop container matching: {container_pattern}")
            return ActionResult(
                success=True,
                action=action,
                message=f"[DRY RUN] Would stop container: {container_pattern}",
                dry_run=True,
            )

        try:
            # Use docker compose stop with the service name
            # Extract service name from container pattern
            service_name = self._container_to_service(container_pattern)

            logger.warning(f"Stopping container service: {service_name} (reason: {action.reason})")

            result = subprocess.run(
                ["docker", "compose", "stop", service_name],
                cwd=self.config.docker_compose_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return ActionResult(
                    success=True,
                    action=action,
                    message=f"Successfully stopped service: {service_name}",
                )
            else:
                return ActionResult(
                    success=False,
                    action=action,
                    message=f"Failed to stop service: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False,
                action=action,
                message="Docker compose stop timed out",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                message=f"Error stopping container: {e}",
            )

    def _restart_container(self, action: WatchdogAction) -> ActionResult:
        """Restart a Docker container using docker compose."""
        container_pattern = str(action.target)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would restart container matching: {container_pattern}")
            return ActionResult(
                success=True,
                action=action,
                message=f"[DRY RUN] Would restart container: {container_pattern}",
                dry_run=True,
            )

        try:
            service_name = self._container_to_service(container_pattern)

            logger.warning(f"Restarting container service: {service_name} (reason: {action.reason})")

            result = subprocess.run(
                ["docker", "compose", "restart", service_name],
                cwd=self.config.docker_compose_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return ActionResult(
                    success=True,
                    action=action,
                    message=f"Successfully restarted service: {service_name}",
                )
            else:
                return ActionResult(
                    success=False,
                    action=action,
                    message=f"Failed to restart service: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False,
                action=action,
                message="Docker compose restart timed out",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                message=f"Error restarting container: {e}",
            )

    def _kill_process(self, action: WatchdogAction) -> ActionResult:
        """Kill a process by PID."""
        try:
            pid = int(action.target)
        except (ValueError, TypeError):
            return ActionResult(
                success=False,
                action=action,
                message=f"Invalid PID: {action.target}",
            )

        # Check if process is protected
        if self._is_protected_process(pid):
            return ActionResult(
                success=False,
                action=action,
                message=f"Process {pid} is protected and cannot be killed",
            )

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would kill process {pid}")
            return ActionResult(
                success=True,
                action=action,
                message=f"[DRY RUN] Would kill process: {pid}",
                dry_run=True,
            )

        try:
            logger.warning(f"Killing process {pid} (reason: {action.reason})")

            # First try SIGTERM
            os.kill(pid, signal.SIGTERM)

            # Wait a moment and check if it's still running
            time.sleep(2)

            try:
                os.kill(pid, 0)  # Check if process exists
                # Still running, use SIGKILL
                logger.warning(f"Process {pid} didn't respond to SIGTERM, using SIGKILL")
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass  # Process already terminated

            return ActionResult(
                success=True,
                action=action,
                message=f"Successfully killed process: {pid}",
            )

        except ProcessLookupError:
            return ActionResult(
                success=False,
                action=action,
                message=f"Process {pid} not found",
            )
        except PermissionError:
            return ActionResult(
                success=False,
                action=action,
                message=f"Permission denied killing process {pid}",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action=action,
                message=f"Error killing process: {e}",
            )

    def _is_protected_process(self, pid: int) -> bool:
        """Check if a process is protected from being killed."""
        try:
            import psutil
            proc = psutil.Process(pid)
            name = proc.name().lower()

            for protected in self.protected_processes:
                if protected.lower() in name:
                    return True

            return False
        except Exception:
            return False  # If we can't check, assume not protected

    def _container_to_service(self, container_name: str) -> str:
        """Convert container name/pattern to docker compose service name."""
        # Common patterns:
        # ronin-nvr-transcode-worker-1 -> transcode-worker
        # ronin-backend -> backend
        # ronin-live-detection -> live-detection

        name = container_name.lower()

        # Remove common prefixes
        for prefix in ["ronin-nvr-", "ronin-"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Remove replica suffixes (-1, -2, etc.)
        import re
        name = re.sub(r'-\d+$', '', name)

        # Map known container patterns to service names
        service_map = {
            "transcode-worker": "transcode-worker",
            "live-detection": "live-detection",
            "backend": "backend",
            "frontend": "frontend",
            "stream-manager": "stream-manager",
            "postgres": "postgres",
            "ml-worker": "ml-worker",
        }

        # Try exact match first
        if name in service_map:
            return service_map[name]

        # Try partial match
        for pattern, service in service_map.items():
            if pattern in name:
                return service

        # Fallback to the cleaned name
        return name
