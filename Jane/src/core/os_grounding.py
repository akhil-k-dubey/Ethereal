"""OS Grounding and System Perception Engine for Project Jane.

Provides Jane with dynamic, real-time embodiment and awareness of the host system:
1. Hardware & System Resource Telemetry:
   - CPU topology, core count, load averages (1m, 5m, 15m).
   - Host RAM utilization (MemTotal, MemFree, MemAvailable from /proc/meminfo).
   - VRAM & GPU status (torch.cuda device properties + non-blocking nvidia-smi telemetry).
   - Host filesystem capacity and disk usage (os.statvfs).
   - Host OS kernel release, machine architecture, and runtime platform.
2. Workspace & Git Repository Grounding:
   - Identifies active workspace path and git repository status.
   - Current git branch, latest commit hash, and summary.
   - Staged, modified, and untracked file counts (git status --porcelain).
   - Recently modified project files.
3. Process & Task Perception:
   - Scans active developer processes (e.g., python, pytest, code, bash, gcc).
   - Tracks top resource-consuming tasks without brittle regex.
4. Desktop & Display Environment:
   - Detects display servers (X11 / Wayland), terminal types, and desktop environments.
5. Unified Grounded Manifest:
   - Aggregates all environmental telemetry into structured dictionaries and
     first-person grounded narrative summaries for SelfModel and Global Cognitive Workspace.

Zero regular expressions; zero brittle string heuristics.
"""

import os
import time
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from src.utils.logger import get_logger

logger = get_logger("os_grounding")


@dataclass
class HardwareTelemetry:
    """Live hardware utilization and platform properties."""
    platform: str
    os_release: str
    architecture: str
    cpu_count: int
    load_averages: Tuple[float, float, float]
    total_ram_mb: float
    available_ram_mb: float
    ram_usage_percent: float
    disk_total_gb: float
    disk_free_gb: float
    disk_used_percent: float
    gpu_available: bool
    gpu_device_name: Optional[str] = None
    vram_total_mb: float = 0.0
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    gpu_temperature_c: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class WorkspaceTelemetry:
    """Ground truth state of the active developer project workspace."""
    workspace_path: str
    is_git_repository: bool
    current_branch: Optional[str] = None
    latest_commit_hash: Optional[str] = None
    latest_commit_message: Optional[str] = None
    modified_files_count: int = 0
    untracked_files_count: int = 0
    staged_files_count: int = 0
    recent_modified_files: List[str] = field(default_factory=list)


@dataclass
class ProcessTelemetry:
    """Active process telemetry on the host operating system."""
    pid: int
    cpu_percent: float
    memory_percent: float
    command: str


@dataclass
class DesktopTelemetry:
    """Desktop environment and terminal session telemetry."""
    has_display: bool
    display_server: str  # "X11", "Wayland", "Headless"
    desktop_environment: Optional[str] = None
    terminal_type: Optional[str] = None
    user_name: str = "akhil"


@dataclass
class OSStateManifest:
    """Unified operating system perception manifest."""
    timestamp: float
    hardware: HardwareTelemetry
    workspace: WorkspaceTelemetry
    active_processes: List[ProcessTelemetry]
    desktop: DesktopTelemetry
    grounded_narrative: str


class OSGroundingEngine:
    """
    Host System Perception and Environmental Grounding Engine.
    Provides real-time telemetry of hardware, workspace, git, and processes.
    """

    _instance: Optional["OSGroundingEngine"] = None

    @classmethod
    def get_instance(cls, workspace_root: Optional[str] = None) -> "OSGroundingEngine":
        if cls._instance is None:
            cls._instance = cls(workspace_root=workspace_root)
        return cls._instance

    def __init__(self, workspace_root: Optional[str] = None, cache_ttl_seconds: float = 1.0):
        if workspace_root:
            self.workspace_root = os.path.abspath(workspace_root)
        else:
            # Detect root from current working directory or Jane codebase
            cwd = os.getcwd()
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_root = os.path.abspath(os.path.join(current_file_dir, "../.."))
            if os.path.exists(os.path.join(cwd, "src")):
                self.workspace_root = os.path.abspath(cwd)
            elif os.path.exists(os.path.join(candidate_root, "src")):
                self.workspace_root = candidate_root
            else:
                self.workspace_root = os.path.abspath(cwd)

        self.cache_ttl = cache_ttl_seconds
        self._cached_manifest: Optional[OSStateManifest] = None
        self._last_manifest_time: float = 0.0

    def get_hardware_telemetry(self) -> HardwareTelemetry:
        """Collects live CPU, RAM, disk, and GPU hardware telemetry."""
        # 1. Platform & CPU
        plat = platform.system()
        release = platform.release()
        arch = platform.machine()
        cpu_count = os.cpu_count() or 1

        try:
            load_avgs = os.getloadavg()
        except (AttributeError, OSError):
            load_avgs = (0.0, 0.0, 0.0)

        # 2. Host RAM via /proc/meminfo or fallback
        total_ram_mb = 0.0
        avail_ram_mb = 0.0
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    mem_data: Dict[str, float] = {}
                    for line in f:
                        parts = line.strip().split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val_str = parts[1].strip().split()[0]
                            if val_str.isdigit():
                                mem_data[key] = float(val_str)
                    total_kb = mem_data.get("MemTotal", 0.0)
                    avail_kb = mem_data.get("MemAvailable", mem_data.get("MemFree", 0.0))
                    total_ram_mb = round(total_kb / 1024.0, 2)
                    avail_ram_mb = round(avail_kb / 1024.0, 2)
            except Exception as e:
                logger.debug("Failed parsing /proc/meminfo: %s", e)

        if total_ram_mb <= 0.0:
            total_ram_mb = 16384.0
            avail_ram_mb = 8192.0

        used_ram = total_ram_mb - avail_ram_mb
        ram_percent = round((used_ram / total_ram_mb) * 100.0, 1) if total_ram_mb > 0 else 0.0

        # 3. Disk space
        disk_total_gb = 0.0
        disk_free_gb = 0.0
        disk_percent = 0.0
        try:
            stat = os.statvfs(self.workspace_root if os.path.exists(self.workspace_root) else "/")
            total_b = stat.f_blocks * stat.f_frsize
            free_b = stat.f_bavail * stat.f_frsize
            disk_total_gb = round(total_b / (1024.0 ** 3), 2)
            disk_free_gb = round(free_b / (1024.0 ** 3), 2)
            if disk_total_gb > 0:
                disk_percent = round(((disk_total_gb - disk_free_gb) / disk_total_gb) * 100.0, 1)
        except Exception as e:
            logger.debug("Failed statvfs disk query: %s", e)

        # 4. GPU / CUDA
        gpu_avail = torch.cuda.is_available()
        gpu_name: Optional[str] = None
        vram_total_mb = 0.0
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        gpu_temp: Optional[float] = None
        gpu_util: Optional[float] = None

        if gpu_avail:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_total_mb = round(props.total_memory / (1024.0 ** 2), 2)
                vram_allocated_mb = round(torch.cuda.memory_allocated(0) / (1024.0 ** 2), 2)
                vram_reserved_mb = round(torch.cuda.memory_reserved(0) / (1024.0 ** 2), 2)
            except Exception as e:
                logger.debug("Torch CUDA properties query error: %s", e)

        # Non-blocking nvidia-smi check for temperature and utilization
        if gpu_avail or os.path.exists("/usr/bin/nvidia-smi"):
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    parts = proc.stdout.strip().split(",")
                    if len(parts) >= 2:
                        temp_str = parts[0].strip()
                        util_str = parts[1].strip()
                        if temp_str.replace(".", "").isdigit():
                            gpu_temp = float(temp_str)
                        if util_str.replace(".", "").isdigit():
                            gpu_util = float(util_str)
            except Exception:
                pass

        return HardwareTelemetry(
            platform=plat,
            os_release=release,
            architecture=arch,
            cpu_count=cpu_count,
            load_averages=(round(load_avgs[0], 2), round(load_avgs[1], 2), round(load_avgs[2], 2)),
            total_ram_mb=total_ram_mb,
            available_ram_mb=avail_ram_mb,
            ram_usage_percent=ram_percent,
            disk_total_gb=disk_total_gb,
            disk_free_gb=disk_free_gb,
            disk_used_percent=disk_percent,
            gpu_available=gpu_avail,
            gpu_device_name=gpu_name,
            vram_total_mb=vram_total_mb,
            vram_allocated_mb=vram_allocated_mb,
            vram_reserved_mb=vram_reserved_mb,
            gpu_temperature_c=gpu_temp,
            gpu_utilization_percent=gpu_util
        )

    def get_workspace_telemetry(self, target_dir: Optional[str] = None) -> WorkspaceTelemetry:
        """Inspects active project workspace and git status."""
        ws_dir = os.path.abspath(target_dir or self.workspace_root)
        is_git = False
        branch: Optional[str] = None
        commit_hash: Optional[str] = None
        commit_msg: Optional[str] = None
        modified_cnt = 0
        untracked_cnt = 0
        staged_cnt = 0

        # Check git worktree
        try:
            rev_proc = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=ws_dir,
                capture_output=True,
                text=True,
                timeout=1.0
            )
            is_git = (rev_proc.returncode == 0 and "true" in rev_proc.stdout.lower())
        except Exception:
            is_git = False

        if is_git:
            # 1. Branch
            try:
                b_proc = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=ws_dir,
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                if b_proc.returncode == 0:
                    branch = b_proc.stdout.strip()
            except Exception:
                pass

            # 2. Latest Commit
            try:
                log_proc = subprocess.run(
                    ["git", "log", "-1", "--format=%h|%s"],
                    cwd=ws_dir,
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                if log_proc.returncode == 0 and "|" in log_proc.stdout:
                    c_parts = log_proc.stdout.strip().split("|", 1)
                    commit_hash = c_parts[0].strip()
                    commit_msg = c_parts[1].strip()
            except Exception:
                pass

            # 3. Status
            try:
                status_proc = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=ws_dir,
                    capture_output=True,
                    text=True,
                    timeout=1.5
                )
                if status_proc.returncode == 0:
                    for line in status_proc.stdout.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("??"):
                            untracked_cnt += 1
                        elif line[0] in ("M", "A", "D", "R", "C"):
                            staged_cnt += 1
                        elif len(line) >= 2 and line[1] in ("M", "D"):
                            modified_cnt += 1
            except Exception:
                pass

        # 4. Recent modified files (last 2 hours)
        recent_files: List[str] = []
        now = time.time()
        two_hours_ago = now - 7200.0
        try:
            for root, dirs, files in os.walk(ws_dir):
                # Skip hidden directories like .git, .venv, __pycache__
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                for f in files:
                    if f.startswith("."):
                        continue
                    full_p = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(full_p)
                        if mtime >= two_hours_ago:
                            rel_p = os.path.relpath(full_p, ws_dir)
                            recent_files.append(rel_p)
                            if len(recent_files) >= 15:
                                break
                    except (OSError, PermissionError):
                        continue
                if len(recent_files) >= 15:
                    break
        except Exception as e:
            logger.debug("Failed scanning recent modified files: %s", e)

        return WorkspaceTelemetry(
            workspace_path=ws_dir,
            is_git_repository=is_git,
            current_branch=branch,
            latest_commit_hash=commit_hash,
            latest_commit_message=commit_msg,
            modified_files_count=modified_cnt,
            untracked_files_count=untracked_cnt,
            staged_files_count=staged_cnt,
            recent_modified_files=recent_files
        )

    def get_process_telemetry(self, max_processes: int = 15) -> List[ProcessTelemetry]:
        """Captures active developer and system processes."""
        processes: List[ProcessTelemetry] = []
        try:
            proc = subprocess.run(
                ["ps", "-eo", "pid,%cpu,%mem,comm", "--sort=-%cpu"],
                capture_output=True,
                text=True,
                timeout=1.5
            )
            if proc.returncode == 0:
                lines = proc.stdout.strip().splitlines()
                # Skip header
                for line in lines[1:]:
                    parts = line.strip().split(None, 3)
                    if len(parts) >= 4:
                        pid_str, cpu_str, mem_str, comm = parts[0], parts[1], parts[2], parts[3]
                        if pid_str.isdigit():
                            pid = int(pid_str)
                            try:
                                cpu_val = float(cpu_str)
                                mem_val = float(mem_str)
                            except ValueError:
                                continue

                            # Focus on active processes or developer tools
                            comm_clean = comm.strip().lower()
                            is_dev = any(dev in comm_clean for dev in ("python", "pytest", "code", "bash", "gcc", "clang", "node", "git", "cargo"))
                            if cpu_val > 0.0 or mem_val > 0.5 or is_dev:
                                processes.append(ProcessTelemetry(
                                    pid=pid,
                                    cpu_percent=cpu_val,
                                    memory_percent=mem_val,
                                    command=comm.strip()
                                ))
                                if len(processes) >= max_processes:
                                    break
        except Exception as e:
            logger.debug("Failed ps process telemetry query: %s", e)

        return processes

    def get_desktop_telemetry(self) -> DesktopTelemetry:
        """Inspects desktop environment and display servers."""
        disp = os.environ.get("DISPLAY")
        wayland = os.environ.get("WAYLAND_DISPLAY")
        desktop_env = os.environ.get("XDG_CURRENT_DESKTOP") or os.environ.get("DESKTOP_SESSION")
        term = os.environ.get("TERM") or os.environ.get("COLORTERM")
        user = os.environ.get("USER") or os.environ.get("LOGNAME", "akhil")

        has_display = bool(disp or wayland)
        display_server = "Wayland" if wayland else ("X11" if disp else "Headless")

        return DesktopTelemetry(
            has_display=has_display,
            display_server=display_server,
            desktop_environment=desktop_env,
            terminal_type=term,
            user_name=user
        )

    def get_unified_os_manifest(self, force_refresh: bool = False) -> OSStateManifest:
        """Returns unified OS perception manifest with short caching."""
        now = time.time()
        if not force_refresh and self._cached_manifest is not None:
            if (now - self._last_manifest_time) < self.cache_ttl:
                return self._cached_manifest

        hw = self.get_hardware_telemetry()
        ws = self.get_workspace_telemetry()
        procs = self.get_process_telemetry()
        dt = self.get_desktop_telemetry()

        # Build grounded natural narrative
        gpu_part = (
            f"GPU: {hw.gpu_device_name} (VRAM {hw.vram_allocated_mb:.1f}/{hw.vram_total_mb:.1f} MB)"
            if hw.gpu_available else "GPU: CPU-only mode"
        )
        git_part = (
            f"Git: on branch '{ws.current_branch}' ({ws.modified_files_count} modified, {ws.untracked_files_count} untracked)"
            if ws.is_git_repository else "Workspace: standalone directory"
        )
        procs_str = ", ".join(p.command for p in procs[:4]) if procs else "none"

        narrative = (
            f"Embodied on {hw.platform} {hw.architecture} (Kernel {hw.os_release}). "
            f"CPU Load: {hw.load_averages[0]:.2f} across {hw.cpu_count} cores | "
            f"Host RAM: {hw.total_ram_mb - hw.available_ram_mb:.0f}/{hw.total_ram_mb:.0f} MB ({hw.ram_usage_percent}%). "
            f"{gpu_part}. "
            f"Active Workspace: '{ws.workspace_path}' | {git_part}. "
            f"Display: {dt.display_server} session. Active developer tasks: [{procs_str}]."
        )

        manifest = OSStateManifest(
            timestamp=now,
            hardware=hw,
            workspace=ws,
            active_processes=procs,
            desktop=dt,
            grounded_narrative=narrative
        )

        self._cached_manifest = manifest
        self._last_manifest_time = now
        return manifest

    def get_grounded_summary(self) -> str:
        """Returns dynamic natural language summary of current OS state."""
        return self.get_unified_os_manifest().grounded_narrative
