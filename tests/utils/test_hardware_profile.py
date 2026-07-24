"""Tests for hardware profile abstraction."""

import pytest
from pydantic import ValidationError

from llm_reliability.utils.hardware_profile import (
    HardwareProfile,
    HardwareRegistry,
    detect_hardware_profile,
)


class TestHardwareProfile:
    def test_minimal_creation(self) -> None:
        profile = HardwareProfile(
            profile_id="test",
            os_name="Linux",
            os_version="Ubuntu",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=16.0,
        )
        assert profile.profile_id == "test"
        assert profile.ram_total_gb == 16.0
        assert profile.gpu_name is None
        assert profile.gpu_count == 0
        assert profile.node_type == "local"

    def test_full_creation(self) -> None:
        profile = HardwareProfile(
            profile_id="full-test",
            os_name="Windows",
            os_version="10.0",
            cpu_architecture="AMD64",
            cpu_cores_logical=16,
            cpu_cores_physical=8,
            ram_total_gb=32.0,
            gpu_name="NVIDIA RTX 4090",
            gpu_count=1,
            vram_total_gb=24.0,
            cuda_version="12.1",
            node_type="local",
        )
        assert profile.gpu_name == "NVIDIA RTX 4090"
        assert profile.vram_total_gb == 24.0

    def test_immutable(self) -> None:
        profile = HardwareProfile(
            profile_id="test",
            os_name="Linux",
            os_version="Ubuntu",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=16.0,
        )
        with pytest.raises(ValidationError):
            profile.profile_id = "changed"

    def test_serialization_round_trip(self) -> None:
        profile = HardwareProfile(
            profile_id="test",
            os_name="Linux",
            os_version="Ubuntu",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=16.0,
        )
        json_str = profile.canonical_json()
        restored = HardwareProfile.from_canonical_json(json_str)
        assert profile == restored
        assert profile.sha256() == restored.sha256()


class TestHardwareRegistry:
    def setup_method(self) -> None:
        HardwareRegistry._profiles.clear()

    def test_register_and_get(self) -> None:
        profile = HardwareProfile(
            profile_id="test-profile",
            os_name="Linux",
            os_version="Ubuntu",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=16.0,
        )
        HardwareRegistry.register(profile)
        retrieved = HardwareRegistry.get("test-profile")
        assert retrieved == profile

    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            HardwareRegistry.get("nonexistent")

    def test_list_profiles_empty(self) -> None:
        assert HardwareRegistry.list_profiles() == []

    def test_list_profiles_sorted(self) -> None:
        for pid in ["z-profile", "a-profile", "m-profile"]:
            HardwareRegistry.register(
                HardwareProfile(
                    profile_id=pid,
                    os_name="Linux",
                    os_version="Ubuntu",
                    cpu_architecture="x86_64",
                    cpu_cores_logical=4,
                    ram_total_gb=16.0,
                )
            )
        profiles = HardwareRegistry.list_profiles()
        assert profiles == ["a-profile", "m-profile", "z-profile"]

    def test_duplicate_overwrites(self) -> None:
        p1 = HardwareProfile(
            profile_id="dup", os_name="Linux", os_version="A",
            cpu_architecture="x86_64", cpu_cores_logical=4, ram_total_gb=16.0,
        )
        p2 = HardwareProfile(
            profile_id="dup", os_name="Windows", os_version="B",
            cpu_architecture="x86_64", cpu_cores_logical=8, ram_total_gb=32.0,
        )
        HardwareRegistry.register(p1)
        HardwareRegistry.register(p2)
        assert HardwareRegistry.get("dup").os_name == "Windows"


class TestDetectHardwareProfile:
    def test_detection_returns_profile(self) -> None:
        profile = detect_hardware_profile(profile_id="test-detect")
        assert profile.profile_id == "test-detect"
        assert profile.os_name != ""
        assert profile.cpu_cores_logical > 0
        assert profile.ram_total_gb > 0

    def test_detection_registers_profile(self) -> None:
        HardwareRegistry._profiles.clear()
        profile = detect_hardware_profile(profile_id="detect-test")
        retrieved = HardwareRegistry.get("detect-test")
        assert retrieved == profile
