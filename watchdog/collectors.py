"""System metrics collectors for the watchdog daemon."""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a running process."""

    pid: int
    name: str
    cmdline: str
    memory_mb: float
    cpu_percent: float
    create_time: datetime
    username: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pid": self.pid,
            "name": self.name,
            "cmdline": self.cmdline[:100] + "..." if len(self.cmdline) > 100 else self.cmdline,
            "memory_mb": round(self.memory_mb, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "runtime_minutes": round((datetime.now() - self.create_time).total_seconds() / 60, 1),
            "username": self.username,
        }


@dataclass
class ContainerInfo:
    """Information about a Docker container."""

    name: str
    status: str
    image: str
    cpu_percent: float
    memory_mb: float
    pids: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status,
            "image": self.image,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
            "pids": self.pids,
        }


@dataclass
class SystemState:
    """Complete system state snapshot."""

    timestamp: datetime
    memory_percent: float
    memory_available_gb: float
    memory_total_gb: float
    cpu_percent: float
    load_average: tuple[float, float, float]
    ffmpeg_processes: list[ProcessInfo] = field(default_factory=list)
    docker_containers: list[ContainerInfo] = field(default_factory=list)
    top_memory_processes: list[ProcessInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memory": {
                "percent": round(self.memory_percent, 1),
                "available_gb": round(self.memory_available_gb, 1),
                "total_gb": round(self.memory_total_gb, 1),
            },
            "cpu_percent": round(self.cpu_percent, 1),
            "load_average": {
                "1m": round(self.load_average[0], 2),
                "5m": round(self.load_average[1], 2),
                "15m": round(self.load_average[2], 2),
            },
            "ffmpeg_processes": [p.to_dict() for p in self.ffmpeg_processes],
            "docker_containers": [c.to_dict() for c in self.docker_containers],
            "top_memory_processes": [p.to_dict() for p in self.top_memory_processes],
        }

    def format_for_llm(self) -> str:
        """Format system state for LLM prompt."""
        lines = [
            "## Current System State",
            f"- Memory: {self.memory_percent:.1f}% used ({self.memory_available_gb:.1f} GB available of {self.memory_total_gb:.1f} GB total)",
            f"- CPU: {self.cpu_percent:.1f}% used",
            f"- Load Average: {self.load_average[0]:.2f} (1m), {self.load_average[1]:.2f} (5m), {self.load_average[2]:.2f} (15m)",
            "",
            f"## FFmpeg Processes ({len(self.ffmpeg_processes)} running)",
        ]

        if self.ffmpeg_processes:
            lines.append("| PID | Memory (MB) | CPU % | Runtime (min) | Command |")
            lines.append("|-----|-------------|-------|---------------|---------|")
            for p in self.ffmpeg_processes:
                runtime = (datetime.now() - p.create_time).total_seconds() / 60
                cmd_short = p.cmdline[:50] + "..." if len(p.cmdline) > 50 else p.cmdline
                lines.append(f"| {p.pid} | {p.memory_mb:.0f} | {p.cpu_percent:.1f} | {runtime:.1f} | {cmd_short} |")
        else:
            lines.append("No ffmpeg processes running.")

        lines.extend([
            "",
            f"## Top Memory Consumers (top 10)",
            "| PID | Name | Memory (MB) | CPU % |",
            "|-----|------|-------------|-------|",
        ])
        for p in self.top_memory_processes[:10]:
            lines.append(f"| {p.pid} | {p.name} | {p.memory_mb:.0f} | {p.cpu_percent:.1f} |")

        lines.extend([
            "",
            f"## Docker Containers ({len(self.docker_containers)} running)",
            "| Name | Status | Memory (MB) | CPU % | PIDs |",
            "|------|--------|-------------|-------|------|",
        ])
        for c in self.docker_containers:
            lines.append(f"| {c.name} | {c.status} | {c.memory_mb:.0f} | {c.cpu_percent:.1f} | {c.pids} |")

        return "\n".join(lines)


class SystemCollector:
    """Collects system metrics."""

    def __init__(self) -> None:
        """Initialize the collector."""
        # Pre-fetch CPU percent to initialize (first call always returns 0)
        psutil.cpu_percent(interval=None)

    def collect(self) -> SystemState:
        """Collect current system state."""
        # Memory
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_available_gb = mem.available / (1024 ** 3)
        memory_total_gb = mem.total / (1024 ** 3)

        # CPU and load
        cpu_percent = psutil.cpu_percent(interval=None)
        load_average = psutil.getloadavg()

        # Processes
        ffmpeg_processes = self._get_ffmpeg_processes()
        top_memory_processes = self._get_top_memory_processes()

        # Docker containers
        docker_containers = self._get_docker_containers()

        return SystemState(
            timestamp=datetime.now(),
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            memory_total_gb=memory_total_gb,
            cpu_percent=cpu_percent,
            load_average=load_average,
            ffmpeg_processes=ffmpeg_processes,
            docker_containers=docker_containers,
            top_memory_processes=top_memory_processes,
        )

    def _get_ffmpeg_processes(self) -> list[ProcessInfo]:
        """Get all running ffmpeg processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent', 'create_time', 'username']):
            try:
                if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                    processes.append(self._process_to_info(proc))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return sorted(processes, key=lambda p: p.memory_mb, reverse=True)

    def _get_top_memory_processes(self, limit: int = 15) -> list[ProcessInfo]:
        """Get top memory-consuming processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent', 'create_time', 'username']):
            try:
                info = self._process_to_info(proc)
                if info.memory_mb > 10:  # Only include processes using > 10 MB
                    processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return sorted(processes, key=lambda p: p.memory_mb, reverse=True)[:limit]

    def _process_to_info(self, proc: psutil.Process) -> ProcessInfo:
        """Convert psutil Process to ProcessInfo."""
        info = proc.info
        cmdline = " ".join(info.get('cmdline') or []) or info.get('name', 'unknown')
        memory_mb = (info.get('memory_info') or proc.memory_info()).rss / (1024 ** 2)
        create_time = datetime.fromtimestamp(info.get('create_time', 0))

        return ProcessInfo(
            pid=info['pid'],
            name=info.get('name', 'unknown'),
            cmdline=cmdline,
            memory_mb=memory_mb,
            cpu_percent=info.get('cpu_percent', 0) or 0,
            create_time=create_time,
            username=info.get('username', 'unknown'),
        )

    def _get_docker_containers(self) -> list[ContainerInfo]:
        """Get running Docker container stats."""
        containers = []
        try:
            # Use docker stats --no-stream for a single snapshot
            result = subprocess.run(
                [
                    "docker", "stats", "--no-stream",
                    "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.PIDs}}"
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        name = parts[0]
                        cpu_str = parts[1].replace("%", "")
                        mem_str = parts[2].split("/")[0].strip()
                        pids_str = parts[3]

                        # Parse CPU
                        try:
                            cpu_percent = float(cpu_str)
                        except ValueError:
                            cpu_percent = 0.0

                        # Parse memory (e.g., "1.5GiB" or "500MiB")
                        memory_mb = self._parse_memory_string(mem_str)

                        # Parse PIDs
                        try:
                            pids = int(pids_str)
                        except ValueError:
                            pids = 0

                        containers.append(ContainerInfo(
                            name=name,
                            status="running",
                            image="",
                            cpu_percent=cpu_percent,
                            memory_mb=memory_mb,
                            pids=pids,
                        ))
        except subprocess.TimeoutExpired:
            logger.warning("Docker stats timed out")
        except FileNotFoundError:
            logger.warning("Docker command not found")
        except Exception as e:
            logger.warning(f"Error getting docker stats: {e}")

        return containers

    def _parse_memory_string(self, mem_str: str) -> float:
        """Parse Docker memory string like '1.5GiB' or '500MiB' to MB."""
        mem_str = mem_str.strip().upper()
        try:
            if "GIB" in mem_str or "GB" in mem_str:
                return float(mem_str.replace("GIB", "").replace("GB", "").strip()) * 1024
            elif "MIB" in mem_str or "MB" in mem_str:
                return float(mem_str.replace("MIB", "").replace("MB", "").strip())
            elif "KIB" in mem_str or "KB" in mem_str:
                return float(mem_str.replace("KIB", "").replace("KB", "").strip()) / 1024
            else:
                return float(mem_str)
        except ValueError:
            return 0.0
