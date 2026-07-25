"""
Hardware Profile Abstraction Layer for LLM Reliability Ranking Framework.

Provides automatic detection of local hardware specs, registration of target
hardware profiles (CPU, RAM, GPU, OS), and serialization for reproducible execution manifests.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any

from llm_reliability.utils.serialization import SerializableModel

logger = logging.getLogger(__name__)


class HardwareProfile(SerializableModel):
    """Hardware profile specification representing machine resources."""

    profile_id: str
    os_name: str
    os_version: str
    cpu_architecture: str
    cpu_cores_logical: int
    cpu_cores_physical: int | None = None
    cpu_frequency_mhz: float | None = None
    ram_total_gb: float
    ram_available_gb: float | None = None
    gpu_name: str | None = None
    gpu_count: int = 0
    vram_total_gb: float = 0.0
    gpu_driver: str | None = None
    cuda_version: str | None = None
    python_version: str = ""
    ollama_version: str | None = None
    node_type: str = "local"  # "local", "cloud", "edge"
    profile_name: str = ""
    metadata: dict[str, Any] = {}


class HardwareRegistry:
    """Registry managing target hardware execution profiles."""

    _profiles: dict[str, HardwareProfile] = {}

    @classmethod
    def register(cls, profile: HardwareProfile) -> None:
        cls._profiles[profile.profile_id] = profile

    @classmethod
    def get(cls, profile_id: str) -> HardwareProfile:
        if profile_id not in cls._profiles:
            raise KeyError(f"Hardware profile '{profile_id}' not found in registry.")
        return cls._profiles[profile_id]

    @classmethod
    def list_profiles(cls) -> list[str]:
        return sorted(list(cls._profiles.keys()))


def detect_hardware_profile(profile_id: str = "Local_System") -> HardwareProfile:
    """Detect current local system hardware specs dynamically."""
    os_name = platform.system()
    os_version = platform.platform()
    arch = platform.machine()
    python_ver = platform.python_version()

    logical_cores = os.cpu_count() or 0
    physical_cores = 0
    ram_gb = 0.0
    ram_avail = None
    cpu_freq = None
    try:
        import psutil

        logical_cores = psutil.cpu_count(logical=True) or 1
        physical_cores = psutil.cpu_count(logical=False) or 0
        mem = psutil.virtual_memory()
        ram_gb = round(mem.total / (1024**3), 2)
        ram_avail = round(mem.available / (1024**3), 2)
        freq = psutil.cpu_freq()
        if freq is not None:
            cpu_freq = round(freq.current, 2)
    except ImportError:
        logger.warning("psutil not available; hardware detection limited.")

    gpu_name = None
    gpu_count = 0
    vram_gb = 0.0
    cuda_ver = None
    gpu_driver = None

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)
            cuda_ver = torch.version.cuda
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_getDriverVersion"):
            gpu_driver = str(torch._C._cuda_getDriverVersion())
    except Exception:
        pass

    ollama_ver = None
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        output = (result.stdout or result.stderr or "").strip()
        if output:
            for line in output.splitlines():
                line = line.strip()
                if line and "Warning" not in line and "could not connect" not in line:
                    ollama_ver = line
                    break
    except Exception:
        pass

    profile = HardwareProfile(
        profile_id=profile_id,
        os_name=os_name,
        os_version=os_version,
        cpu_architecture=arch,
        cpu_cores_logical=logical_cores,
        cpu_cores_physical=physical_cores,
        cpu_frequency_mhz=cpu_freq,
        ram_total_gb=ram_gb,
        ram_available_gb=ram_avail,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        vram_total_gb=vram_gb,
        gpu_driver=gpu_driver,
        cuda_version=cuda_ver,
        python_version=python_ver,
        ollama_version=ollama_ver,
        node_type="local",
    )
    HardwareRegistry.register(profile)
    return profile


def _register_named_profiles() -> None:
    profiles = [
        HardwareProfile(
            profile_id="Laptop-8GB",
            profile_name="Laptop-8GB",
            os_name="Linux",
            os_version="Ubuntu 22.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            cpu_cores_physical=4,
            ram_total_gb=8.0,
            ram_available_gb=4.5,
            node_type="local",
            metadata={"category": "entry-level"},
        ),
        HardwareProfile(
            profile_id="Laptop-16GB",
            profile_name="Laptop-16GB",
            os_name="Linux",
            os_version="Ubuntu 22.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=8,
            cpu_cores_physical=4,
            ram_total_gb=16.0,
            ram_available_gb=10.0,
            node_type="local",
            metadata={"category": "mid-range"},
        ),
        HardwareProfile(
            profile_id="Desktop-32GB",
            profile_name="Desktop-32GB",
            os_name="Linux",
            os_version="Ubuntu 24.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=16,
            cpu_cores_physical=8,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_name="NVIDIA RTX 4060",
            gpu_count=1,
            vram_total_gb=8.0,
            cuda_version="12.4",
            node_type="local",
            metadata={"category": "desktop"},
        ),
        HardwareProfile(
            profile_id="CPU-Only",
            profile_name="CPU-Only",
            os_name="Linux",
            os_version="Ubuntu 22.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=8,
            cpu_cores_physical=4,
            ram_total_gb=32.0,
            ram_available_gb=28.0,
            gpu_name=None,
            gpu_count=0,
            vram_total_gb=0.0,
            node_type="local",
            metadata={"category": "cpu-only"},
        ),
        HardwareProfile(
            profile_id="RTX3060",
            profile_name="RTX3060",
            os_name="Linux",
            os_version="Ubuntu 24.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=12,
            cpu_cores_physical=6,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_name="NVIDIA GeForce RTX 3060",
            gpu_count=1,
            vram_total_gb=12.0,
            cuda_version="12.4",
            node_type="local",
            metadata={"category": "gpu-entry"},
        ),
        HardwareProfile(
            profile_id="RTX4060",
            profile_name="RTX4060",
            os_name="Linux",
            os_version="Ubuntu 24.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=16,
            cpu_cores_physical=8,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_name="NVIDIA GeForce RTX 4060",
            gpu_count=1,
            vram_total_gb=8.0,
            cuda_version="12.4",
            node_type="local",
            metadata={"category": "gpu-mainstream"},
        ),
        HardwareProfile(
            profile_id="RTX4090",
            profile_name="RTX4090",
            os_name="Linux",
            os_version="Ubuntu 24.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=24,
            cpu_cores_physical=12,
            ram_total_gb=64.0,
            ram_available_gb=56.0,
            gpu_name="NVIDIA GeForce RTX 4090",
            gpu_count=1,
            vram_total_gb=24.0,
            cuda_version="12.4",
            node_type="local",
            metadata={"category": "gpu-high-end"},
        ),
        HardwareProfile(
            profile_id="Apple-Silicon",
            profile_name="Apple-Silicon",
            os_name="Darwin",
            os_version="macOS Sonoma 14.5",
            cpu_architecture="arm64",
            cpu_cores_logical=12,
            cpu_cores_physical=12,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_name="Apple M3 Pro",
            gpu_count=1,
            vram_total_gb=32.0,
            node_type="local",
            metadata={"category": "apple-silicon"},
        ),
        HardwareProfile(
            profile_id="Cloud_NVIDIA_A100_80GB",
            profile_name="A100-80GB",
            os_name="Linux",
            os_version="Ubuntu 22.04 LTS",
            cpu_architecture="x86_64",
            cpu_cores_logical=64,
            cpu_cores_physical=32,
            ram_total_gb=256.0,
            ram_available_gb=240.0,
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_count=8,
            vram_total_gb=640.0,
            cuda_version="12.2",
            node_type="cloud",
            metadata={"category": "datacenter"},
        ),
    ]
    for p in profiles:
        HardwareRegistry.register(p)


_register_named_profiles()
