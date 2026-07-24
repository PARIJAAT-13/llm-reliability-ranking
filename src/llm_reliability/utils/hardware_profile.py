"""
Hardware Profile Abstraction Layer for LLM Reliability Ranking Framework.

Provides automatic detection of local hardware specs, registration of target
hardware profiles (CPU, RAM, GPU, OS), and serialization for reproducible execution manifests.
"""

from __future__ import annotations

import logging
import platform
import psutil
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
    ram_total_gb: float
    gpu_name: str | None = None
    gpu_count: int = 0
    vram_total_gb: float = 0.0
    cuda_version: str | None = None
    node_type: str = "local"  # "local", "cloud", "edge"
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
    logical_cores = psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False)
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

    gpu_name = None
    gpu_count = 0
    vram_gb = 0.0
    cuda_ver = None

    # Optional torch CUDA check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            cuda_ver = torch.version.cuda
    except Exception:
        pass

    profile = HardwareProfile(
        profile_id=profile_id,
        os_name=os_name,
        os_version=os_version,
        cpu_architecture=arch,
        cpu_cores_logical=logical_cores,
        cpu_cores_physical=physical_cores,
        ram_total_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        vram_total_gb=vram_gb,
        cuda_version=cuda_ver,
        node_type="local",
    )
    HardwareRegistry.register(profile)
    return profile


# Register standard reference hardware profiles for reproducibility
HardwareRegistry.register(
    HardwareProfile(
        profile_id="Local_x86_CPU_RAM16GB",
        os_name="Windows",
        os_version="10.0.26200",
        cpu_architecture="AMD64",
        cpu_cores_logical=16,
        cpu_cores_physical=8,
        ram_total_gb=15.34,
        gpu_name="Integrated/Host GPU",
        gpu_count=1,
        vram_total_gb=4.0,
        node_type="local",
    )
)

HardwareRegistry.register(
    HardwareProfile(
        profile_id="Cloud_NVIDIA_A100_80GB",
        os_name="Linux",
        os_version="Ubuntu 22.04 LTS",
        cpu_architecture="x86_64",
        cpu_cores_logical=64,
        cpu_cores_physical=32,
        ram_total_gb=256.0,
        gpu_name="NVIDIA A100-SXM4-80GB",
        gpu_count=8,
        vram_total_gb=640.0,
        cuda_version="12.2",
        node_type="cloud",
    )
)

HardwareRegistry.register(
    HardwareProfile(
        profile_id="Edge_Apple_M3_32GB",
        os_name="Darwin",
        os_version="macOS Sonoma 14.5",
        cpu_architecture="arm64",
        cpu_cores_logical=12,
        cpu_cores_physical=12,
        ram_total_gb=32.0,
        gpu_name="Apple M3 Max GPU",
        gpu_count=1,
        vram_total_gb=32.0,
        node_type="edge",
    )
)
