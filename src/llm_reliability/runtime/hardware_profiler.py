"""Hardware profiling — detect CPU, memory, GPU, and runtime statistics."""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CPUInfo:
    count: int = 0
    model: str = ""
    usage_percent: float = 0.0


@dataclass
class MemoryInfo:
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    percent: float = 0.0


@dataclass
class GPUInfo:
    available: bool = False
    count: int = 0
    models: list[str] = field(default_factory=list)
    vram_total_gb: list[float] = field(default_factory=list)
    vram_free_gb: list[float] = field(default_factory=list)
    cuda_version: str | None = None
    rocm_version: str | None = None
    metal_supported: bool = False


@dataclass
class RuntimeStatistics:
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0


@dataclass
class HardwareProfile:
    platform: str = field(default_factory=platform.system)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    python_version: str = field(default_factory=lambda: platform.python_version())
    hostname: str = field(default_factory=lambda: platform.node())


class HardwareProfiler:
    """Probes hardware capabilities with optional GPU detection."""

    @staticmethod
    def profile(gpu_enabled: bool = True) -> HardwareProfile:
        profile = HardwareProfile()
        HardwareProfiler._detect_cpu(profile)
        HardwareProfiler._detect_memory(profile)
        if gpu_enabled:
            HardwareProfiler._detect_gpu(profile)
        return profile

    @staticmethod
    def estimate_model_memory(parameter_count_b: float) -> dict[str, float]:
        return {
            "fp32_gb": parameter_count_b * 4.0,
            "fp16_gb": parameter_count_b * 2.0,
            "int8_gb": parameter_count_b * 1.0,
            "int4_gb": parameter_count_b * 0.5,
        }

    @staticmethod
    def can_run_model(
        parameter_count_b: float,
        precision: str = "fp16",
        gpu_enabled: bool = True,
    ) -> tuple[bool, str]:
        profile = HardwareProfiler.profile(gpu_enabled=gpu_enabled)
        memory_map = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}
        bytes_per_param = memory_map.get(precision, 2.0)
        required = parameter_count_b * bytes_per_param
        available = 0.0
        if profile.gpu.available:
            available = max(profile.gpu.vram_total_gb) if profile.gpu.vram_total_gb else 0.0
        if available < required and profile.memory.available_gb > required * 1.5:
            return (
                True,
                f"CPU fallback ({profile.memory.available_gb:.1f}GB RAM >= {required:.1f}GB required)",
            )
        if available >= required:
            return True, f"GPU ({available:.1f}GB VRAM >= {required:.1f}GB required)"
        return False, f"Insufficient memory: need {required:.1f}GB, have {available:.1f}GB"

    @staticmethod
    def _detect_cpu(profile: HardwareProfile) -> None:
        profile.cpu.count = os.cpu_count() or 1
        try:
            if platform.system() == "Linux":
                import subprocess

                result = subprocess.run(
                    ["cat", "/proc/cpuinfo"], capture_output=True, text=True, timeout=5.0
                )
                for line in result.stdout.split("\n"):
                    if "model name" in line:
                        profile.cpu.model = line.split(":")[-1].strip()
                        break
            elif platform.system() == "Darwin":
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0:
                    profile.cpu.model = result.stdout.strip()
            elif platform.system() == "Windows":
                import subprocess

                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=5.0
                )
                lines = [line.strip() for line in result.stdout.split("\n") if line.strip()]
                if len(lines) > 1:
                    profile.cpu.model = lines[1]
        except Exception:
            pass

    @staticmethod
    def _detect_memory(profile: HardwareProfile) -> None:
        try:
            import psutil

            mem = psutil.virtual_memory()
            profile.memory.total_gb = mem.total / (1024**3)
            profile.memory.available_gb = mem.available / (1024**3)
            profile.memory.used_gb = mem.used / (1024**3)
            profile.memory.percent = mem.percent
        except ImportError:
            try:
                if platform.system() == "Linux":
                    import subprocess

                    result = subprocess.run(
                        ["free", "-b"], capture_output=True, text=True, timeout=5.0
                    )
                    for line in result.stdout.split("\n"):
                        if line.startswith("Mem:"):
                            parts = line.split()
                            if len(parts) > 1:
                                profile.memory.total_gb = float(parts[1]) / (1024**3)
                                if len(parts) > 2:
                                    profile.memory.used_gb = float(parts[2]) / (1024**3)
                                if len(parts) > 3:
                                    profile.memory.available_gb = float(parts[3]) / (1024**3)
            except Exception:
                pass

    @staticmethod
    def _detect_gpu(profile: HardwareProfile) -> None:
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                for line in lines:
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        profile.gpu.models.append(parts[0])
                        try:
                            profile.gpu.vram_total_gb.append(float(parts[1]) / 1024.0)
                        except ValueError:
                            profile.gpu.vram_total_gb.append(0.0)
                if profile.gpu.models:
                    profile.gpu.available = True
                    profile.gpu.count = len(profile.gpu.models)
                    HardwareProfiler._detect_cuda_version(profile)
                return
        except Exception:
            pass
        if platform.system() == "Darwin":
            profile.gpu.metal_supported = True
            profile.gpu.available = True
            try:
                import subprocess

                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "Chipset Model" in line:
                            profile.gpu.models.append(line.split(":")[-1].strip())
                    profile.gpu.count = len(profile.gpu.models)
            except Exception:
                pass

    @staticmethod
    def _detect_cuda_version(profile: HardwareProfile) -> None:
        try:
            import subprocess

            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10.0)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "CUDA Version" in line:
                        profile.gpu.cuda_version = line.split(":")[-1].strip()
                        break
        except Exception:
            pass


class RuntimeTimer:
    """Measures execution time of callables and accumulates statistics."""

    def __init__(self) -> None:
        self.stats = RuntimeStatistics()

    def measure(self, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            result = fn()
            return result
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._record(elapsed)

    def measure_iteration(self, fn: Callable[[], Any], iterations: int = 1) -> list[float]:
        latencies: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        for ms in latencies:
            self._record(ms)
        return latencies

    def _record(self, ms: float) -> None:
        self.stats.execution_count += 1
        self.stats.total_time_ms += ms
        self.stats.avg_time_ms = self.stats.total_time_ms / self.stats.execution_count
        if self.stats.min_time_ms == 0 or ms < self.stats.min_time_ms:
            self.stats.min_time_ms = ms
        if ms > self.stats.max_time_ms:
            self.stats.max_time_ms = ms

    def reset(self) -> None:
        self.stats = RuntimeStatistics()
