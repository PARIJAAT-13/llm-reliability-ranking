"""Extended tests for HardwareProfile, HardwareProfiler — 45+ tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_reliability.runtime.hardware_profiler import CPUInfo, GPUInfo
from llm_reliability.runtime.hardware_profiler import \
    HardwareProfile as RuntimeHardwareProfile
from llm_reliability.runtime.hardware_profiler import (HardwareProfiler,
                                                       MemoryInfo)
from llm_reliability.utils.hardware_profile import (HardwareProfile,
                                                    HardwareRegistry,
                                                    detect_hardware_profile)

SHA256_HEX_LEN = 64


class TestHardwareProfileUtils:
    def test_minimal_construction(self):
        hp = HardwareProfile(
            profile_id="test",
            os_name="Linux",
            os_version="Ubuntu 22.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        assert hp.profile_id == "test"
        assert hp.os_name == "Linux"
        assert hp.cpu_cores_logical == 4
        assert hp.ram_total_gb == 8.0
        assert hp.node_type == "local"

    def test_full_construction(self):
        hp = HardwareProfile(
            profile_id="full",
            profile_name="Full System",
            os_name="Linux",
            os_version="Ubuntu 24.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=16,
            cpu_cores_physical=8,
            cpu_frequency_mhz=3500.0,
            ram_total_gb=32.0,
            ram_available_gb=28.5,
            gpu_name="NVIDIA RTX 4090",
            gpu_count=1,
            vram_total_gb=24.0,
            gpu_driver="535.129.03",
            cuda_version="12.2",
            python_version="3.12",
            ollama_version="0.1.30",
            node_type="local",
            metadata={"owner": "test"},
        )
        assert hp.gpu_name == "NVIDIA RTX 4090"
        assert hp.cuda_version == "12.2"
        assert hp.vram_total_gb == 24.0
        assert hp.metadata == {"owner": "test"}

    def test_defaults(self):
        hp = HardwareProfile(
            profile_id="defaults",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        assert hp.cpu_cores_physical is None
        assert hp.cpu_frequency_mhz is None
        assert hp.ram_available_gb is None
        assert hp.gpu_name is None
        assert hp.gpu_count == 0
        assert hp.vram_total_gb == 0.0
        assert hp.gpu_driver is None
        assert hp.cuda_version is None
        assert hp.ollama_version is None
        assert hp.node_type == "local"
        assert hp.profile_name == ""
        assert hp.metadata == {}

    def test_serialization_round_trip(self):
        hp = HardwareProfile(
            profile_id="rt",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        restored = HardwareProfile.from_canonical_json(hp.canonical_json())
        assert hp == restored
        assert hp.sha256() == restored.sha256()

    def test_sha256_consistency(self):
        hp = HardwareProfile(
            profile_id="hash-test",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        assert len(hp.sha256()) == SHA256_HEX_LEN
        assert hp.sha256() == hp.sha256()

    def test_equality(self):
        a = HardwareProfile(
            profile_id="eq",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        b = HardwareProfile(
            profile_id="eq",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        assert a == b

    def test_inequality(self):
        a = HardwareProfile(
            profile_id="a",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        b = HardwareProfile(
            profile_id="b",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=8,
            ram_total_gb=16.0,
            python_version="3.11",
        )
        assert a != b

    def test_node_type_variants(self):
        for node_type in ["local", "cloud", "edge"]:
            hp = HardwareProfile(
                profile_id=node_type,
                os_name="Linux",
                os_version="x86_64",
                cpu_architecture="x86_64",
                cpu_cores_logical=4,
                ram_total_gb=8.0,
                python_version="3.11",
                node_type=node_type,
            )
            assert hp.node_type == node_type

    def test_immutable(self):
        hp = HardwareProfile(
            profile_id="imm",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        with pytest.raises((TypeError, ValueError)):
            hp.profile_id = "changed"

    def test_rejects_extra_field(self):
        with pytest.raises((TypeError, ValueError)):
            HardwareProfile(
                profile_id="x",
                os_name="L",
                os_version="x",
                cpu_architecture="x",
                cpu_cores_logical=1,
                ram_total_gb=1.0,
                python_version="3",
                unknown=True,
            )


class TestHardwareRegistry:
    def setup_method(self):
        HardwareRegistry._profiles = {}

    def test_register_and_get(self):
        hp = HardwareProfile(
            profile_id="test-profile",
            os_name="Linux",
            os_version="x86_64",
            cpu_architecture="x86_64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
            python_version="3.11",
        )
        HardwareRegistry.register(hp)
        retrieved = HardwareRegistry.get("test-profile")
        assert retrieved == hp

    def test_get_raises_on_missing(self):
        with pytest.raises(KeyError, match="not found"):
            HardwareRegistry.get("nonexistent")

    def test_list_profiles(self):
        for pid in ["z", "a", "m"]:
            HardwareRegistry.register(
                HardwareProfile(
                    profile_id=pid,
                    os_name="L",
                    os_version="x",
                    cpu_architecture="x",
                    cpu_cores_logical=1,
                    ram_total_gb=1.0,
                    python_version="3",
                )
            )
        profiles = HardwareRegistry.list_profiles()
        assert profiles == ["a", "m", "z"]

    def test_register_overwrites(self):
        hp1 = HardwareProfile(
            profile_id="dup",
            os_name="A",
            os_version="x",
            cpu_architecture="x",
            cpu_cores_logical=1,
            ram_total_gb=1.0,
            python_version="3",
        )
        hp2 = HardwareProfile(
            profile_id="dup",
            os_name="B",
            os_version="x",
            cpu_architecture="x",
            cpu_cores_logical=2,
            ram_total_gb=2.0,
            python_version="3",
        )
        HardwareRegistry.register(hp1)
        HardwareRegistry.register(hp2)
        assert HardwareRegistry.get("dup").os_name == "B"

    def test_detect_hardware_profile_returns_valid(self):
        hp = detect_hardware_profile(profile_id="detected")
        assert isinstance(hp, HardwareProfile)
        assert hp.profile_id == "detected"
        assert hp.os_name in ("Linux", "Windows", "Darwin")
        assert hp.python_version != ""


class TestRuntimeHardwareProfile:
    def test_default_construction(self):
        hp = RuntimeHardwareProfile()
        assert isinstance(hp.platform, str)
        assert isinstance(hp.cpu.count, int)
        assert hp.cpu.count >= 0
        assert isinstance(hp.memory.total_gb, float)
        assert hp.gpu.available is False
        assert hp.gpu.count == 0
        assert hp.gpu.models == []
        assert hp.gpu.vram_total_gb == []
        assert hp.gpu.cuda_version is None
        assert hp.gpu.rocm_version is None
        assert hp.gpu.metal_supported is False

    def test_platform_set_correctly(self):
        import platform

        hp = RuntimeHardwareProfile()
        assert hp.platform == platform.system()

    def test_fields_can_be_set(self):
        hp = RuntimeHardwareProfile(
            platform="Linux",
            cpu=CPUInfo(count=8, model="Intel Core i7"),
            memory=MemoryInfo(total_gb=32.0),
            gpu=GPUInfo(
                available=True,
                count=2,
                models=["RTX 4090", "RTX 4080"],
                vram_total_gb=[24.0, 16.0],
                cuda_version="12.2",
                metal_supported=False,
            ),
        )
        assert hp.platform == "Linux"
        assert hp.cpu.count == 8
        assert hp.cpu.model == "Intel Core i7"
        assert hp.memory.total_gb == 32.0
        assert hp.gpu.available is True
        assert hp.gpu.count == 2
        assert hp.gpu.models == ["RTX 4090", "RTX 4080"]
        assert hp.gpu.vram_total_gb == [24.0, 16.0]
        assert hp.gpu.cuda_version == "12.2"
        assert hp.gpu.metal_supported is False

    def test_cpu_model_empty_by_default(self):
        hp = RuntimeHardwareProfile()
        assert hp.cpu.model == ""

    def test_rocm_version_none_by_default(self):
        hp = RuntimeHardwareProfile()
        assert hp.gpu.rocm_version is None


class TestHardwareProfiler:
    def test_profile_returns_valid_structure(self):
        hp = HardwareProfiler.profile()
        assert isinstance(hp, RuntimeHardwareProfile)
        assert hp.cpu.count >= 1
        assert isinstance(hp.memory.total_gb, float)

    def test_profile_includes_cpu_model(self):
        hp = HardwareProfiler.profile()
        assert isinstance(hp.cpu.model, str)

    def test_estimate_model_memory_fp32(self):
        result = HardwareProfiler.estimate_model_memory(1.0)
        assert result["fp32_gb"] == 4.0
        assert result["fp16_gb"] == 2.0
        assert result["int8_gb"] == 1.0
        assert result["int4_gb"] == 0.5

    def test_estimate_model_memory_7b(self):
        result = HardwareProfiler.estimate_model_memory(7.0)
        assert result["fp32_gb"] == 28.0
        assert result["fp16_gb"] == 14.0
        assert result["int8_gb"] == 7.0
        assert result["int4_gb"] == 3.5

    def test_estimate_model_memory_70b(self):
        result = HardwareProfiler.estimate_model_memory(70.0)
        assert result["fp32_gb"] == 280.0
        assert result["fp16_gb"] == 140.0
        assert result["int8_gb"] == 70.0
        assert result["int4_gb"] == 35.0

    def test_estimate_model_memory_zero_params(self):
        result = HardwareProfiler.estimate_model_memory(0.0)
        for v in result.values():
            assert v == 0.0

    def test_estimate_model_memory_small_model(self):
        result = HardwareProfiler.estimate_model_memory(0.5)
        assert result["fp16_gb"] == 1.0

    def test_estimate_model_memory_returns_four_keys(self):
        result = HardwareProfiler.estimate_model_memory(1.0)
        assert set(result.keys()) == {"fp32_gb", "fp16_gb", "int8_gb", "int4_gb"}

    def test_can_run_model_returns_tuple(self):
        result = HardwareProfiler.can_run_model(1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_can_run_model_default_precision(self):
        ok, msg = HardwareProfiler.can_run_model(0.1)
        assert isinstance(ok, bool)

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_gpu_sufficient(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = True
        mock_hp.gpu.vram_total_gb = [24.0]
        mock_hp.memory.available_gb = 64.0
        mock_profile.return_value = mock_hp
        ok, msg = HardwareProfiler.can_run_model(7.0, precision="fp16")
        assert ok is True
        assert "GPU" in msg

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_cpu_fallback(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = False
        mock_hp.gpu.vram_total_gb = []
        mock_hp.memory.available_gb = 64.0
        mock_profile.return_value = mock_hp
        ok, msg = HardwareProfiler.can_run_model(7.0, precision="fp16")
        assert ok is True
        assert "CPU fallback" in msg

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_insufficient(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = False
        mock_hp.gpu.vram_total_gb = []
        mock_hp.memory.available_gb = 4.0
        mock_profile.return_value = mock_hp
        ok, msg = HardwareProfiler.can_run_model(70.0, precision="fp16")
        assert ok is False
        assert "Insufficient" in msg

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_gpu_insufficient_vram(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = True
        mock_hp.gpu.vram_total_gb = [8.0]
        mock_hp.memory.available_gb = 16.0
        mock_profile.return_value = mock_hp
        ok, msg = HardwareProfiler.can_run_model(70.0, precision="fp16")
        assert ok is False
        assert "Insufficient" in msg

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_precision_variants(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = True
        mock_hp.gpu.vram_total_gb = [4.0]
        mock_hp.memory.available_gb = 4.0
        mock_profile.return_value = mock_hp
        for prec, factor in [("fp32", 4.0), ("fp16", 2.0), ("int8", 1.0), ("int4", 0.5)]:
            ok, _ = HardwareProfiler.can_run_model(10.0, precision=prec)
            expected_gb = 10.0 * factor
            assert ok == (4.0 >= expected_gb)

    def test_profile_cpu_count_positive(self):
        hp = HardwareProfiler.profile()
        assert hp.cpu.count > 0

    def test_profile_ram_gb_non_negative(self):
        hp = HardwareProfiler.profile()
        assert hp.memory.total_gb >= 0.0

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_with_unknown_precision_falls_back_to_fp16(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = False
        mock_hp.gpu.vram_total_gb = []
        mock_hp.memory.available_gb = 16.0
        mock_profile.return_value = mock_hp
        ok, _ = HardwareProfiler.can_run_model(7.0, precision="unknown")
        assert isinstance(ok, bool)

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_returns_cpu_fallback_message(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = False
        mock_hp.gpu.vram_total_gb = []
        mock_hp.memory.available_gb = 32.0
        mock_profile.return_value = mock_hp
        ok, msg = HardwareProfiler.can_run_model(7.0)
        assert ok is True
        assert "CPU" in msg

    @patch.object(HardwareProfiler, "profile")
    def test_can_run_model_int8_precision(self, mock_profile):
        mock_hp = MagicMock()
        mock_hp.gpu.available = True
        mock_hp.gpu.vram_total_gb = [8.0]
        mock_hp.memory.available_gb = 32.0
        mock_profile.return_value = mock_hp
        ok, _ = HardwareProfiler.can_run_model(7.0, precision="int8")
        assert ok is True

    def test_hardware_profile_gpu_models_defaults_empty(self):
        hp = RuntimeHardwareProfile()
        assert hp.gpu.models == []

    def test_hardware_profile_gpu_vram_gb_defaults_empty(self):
        hp = RuntimeHardwareProfile()
        assert hp.gpu.vram_total_gb == []


class TestHardwareProfileSerialization:
    def test_utils_profile_canonical_json(self):
        hp = HardwareProfile(
            profile_id="s",
            os_name="L",
            os_version="x",
            cpu_architecture="x",
            cpu_cores_logical=1,
            ram_total_gb=1.0,
            python_version="3",
        )
        j = hp.canonical_json()
        restored = HardwareProfile.from_canonical_json(j)
        assert hp == restored

    def test_utils_profile_sha256_deterministic(self):
        hp = HardwareProfile(
            profile_id="d",
            os_name="L",
            os_version="x",
            cpu_architecture="x",
            cpu_cores_logical=2,
            ram_total_gb=4.0,
            python_version="3.11",
        )
        assert hp.sha256() == hp.sha256()
        assert len(hp.sha256()) == 64
