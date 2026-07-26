"""Extended tests for RuntimeMetadata, RuntimeCapabilities — 40+ tests."""

from __future__ import annotations

import copy

import pytest

from llm_reliability.runtime.metadata import RuntimeCapabilities, RuntimeMetadata

SHA256_HEX_LEN = 64


class TestRuntimeMetadataConstruction:
    def test_minimal_construction(self):
        meta = RuntimeMetadata(runtime_name="test")
        assert meta.runtime_name == "test"
        assert meta.runtime_version is None
        assert meta.backend is None
        assert meta.api_version is None
        assert meta.execution_mode is None
        assert meta.quantization is None
        assert meta.context_length is None
        assert meta.gpu_acceleration is False
        assert meta.gpu_devices == []
        assert meta.thread_count is None
        assert meta.batch_size is None
        assert meta.max_concurrent_requests is None
        assert meta.inference_parameters == {}
        assert isinstance(meta.capabilities, RuntimeCapabilities)
        assert meta.metadata == {}

    def test_full_construction(self):
        caps = RuntimeCapabilities(
            model_loading=True,
            token_counting=True,
            streaming=True,
            gpu_acceleration=True,
        )
        meta = RuntimeMetadata(
            runtime_name="vllm",
            runtime_version="0.6.0",
            backend="vllm",
            api_version="v1",
            execution_mode="remote",
            quantization="fp16",
            context_length=131072,
            gpu_acceleration=True,
            gpu_devices=["NVIDIA A100", "NVIDIA A100"],
            thread_count=16,
            batch_size=32,
            max_concurrent_requests=64,
            inference_parameters={"temperature": 0.7, "top_p": 0.9},
            capabilities=caps,
            metadata={"deployed_by": "ci"},
        )
        assert meta.runtime_name == "vllm"
        assert meta.runtime_version == "0.6.0"
        assert meta.backend == "vllm"
        assert meta.api_version == "v1"
        assert meta.execution_mode == "remote"
        assert meta.quantization == "fp16"
        assert meta.context_length == 131072
        assert meta.gpu_acceleration is True
        assert meta.gpu_devices == ["NVIDIA A100", "NVIDIA A100"]
        assert meta.thread_count == 16
        assert meta.batch_size == 32
        assert meta.max_concurrent_requests == 64
        assert meta.inference_parameters == {"temperature": 0.7, "top_p": 0.9}
        assert meta.capabilities.token_counting is True
        assert meta.capabilities.streaming is True
        assert meta.metadata == {"deployed_by": "ci"}

    def test_construction_with_only_required(self):
        RuntimeMetadata(runtime_name="minimal")

    def test_construction_with_inference_params_empty(self):
        meta = RuntimeMetadata(runtime_name="t", inference_parameters={})
        assert meta.inference_parameters == {}

    def test_construction_with_metadata_nested(self):
        meta = RuntimeMetadata(
            runtime_name="t",
            metadata={"env": {"region": "us-east-1", "zone": "a"}},
        )
        assert meta.metadata["env"]["region"] == "us-east-1"

    def test_construction_with_gpu_devices_various(self):
        meta = RuntimeMetadata(runtime_name="t", gpu_devices=["A"])
        assert meta.gpu_devices == ["A"]

    def test_construction_rejects_extra_field(self):
        with pytest.raises((TypeError, ValueError)):
            RuntimeMetadata(runtime_name="t", unknown_field="x")

    def test_construction_accepts_all_runtime_version_formats(self):
        for ver in ["1.0.0", "latest", "v0.1", "", None]:
            meta = RuntimeMetadata(runtime_name="t", runtime_version=ver)
            assert meta.runtime_version == ver

    def test_context_length_zero(self):
        meta = RuntimeMetadata(runtime_name="t", context_length=0)
        assert meta.context_length == 0

    def test_context_length_large(self):
        meta = RuntimeMetadata(runtime_name="t", context_length=1_000_000)
        assert meta.context_length == 1_000_000

    def test_batch_size_none(self):
        meta = RuntimeMetadata(runtime_name="t", batch_size=None)
        assert meta.batch_size is None

    def test_thread_count_zero(self):
        meta = RuntimeMetadata(runtime_name="t", thread_count=0)
        assert meta.thread_count == 0

    def test_max_concurrent_requests(self):
        meta = RuntimeMetadata(runtime_name="t", max_concurrent_requests=128)
        assert meta.max_concurrent_requests == 128

    def test_quantization_various_formats(self):
        for q in ["q4_k_m", "q8_0", "fp16", "int8", "awq", "gptq"]:
            meta = RuntimeMetadata(runtime_name="t", quantization=q)
            assert meta.quantization == q

    def test_execution_mode_various(self):
        for mode in ["local", "remote", "hybrid", "edge"]:
            meta = RuntimeMetadata(runtime_name="t", execution_mode=mode)
            assert meta.execution_mode == mode

    def test_gpu_acceleration_flag(self):
        meta = RuntimeMetadata(runtime_name="t", gpu_acceleration=True)
        assert meta.gpu_acceleration is True

    def test_all_fields_explicit_none(self):
        meta = RuntimeMetadata(
            runtime_name="t",
            runtime_version=None,
            backend=None,
            api_version=None,
            execution_mode=None,
            quantization=None,
            context_length=None,
            thread_count=None,
            batch_size=None,
            max_concurrent_requests=None,
        )
        assert meta.runtime_version is None
        assert meta.backend is None
        assert meta.api_version is None


class TestRuntimeCapabilities:
    def test_all_defaults_false(self):
        caps = RuntimeCapabilities()
        for field_name in RuntimeCapabilities.model_fields:
            assert getattr(caps, field_name) is False

    def test_selective_true(self):
        caps = RuntimeCapabilities(
            model_loading=True,
            health_check=True,
            streaming=True,
            quantization=True,
        )
        assert caps.model_loading is True
        assert caps.health_check is True
        assert caps.streaming is True
        assert caps.quantization is True
        assert caps.token_counting is False
        assert caps.batch_inference is False
        assert caps.multi_gpu is False

    def test_all_true(self):
        kwargs = {f: True for f in RuntimeCapabilities.model_fields}
        caps = RuntimeCapabilities(**kwargs)
        for field_name in RuntimeCapabilities.model_fields:
            assert getattr(caps, field_name) is True

    def test_mixed_values(self):
        caps = RuntimeCapabilities(
            model_loading=True,
            model_unloading=False,
            health_check=True,
            token_counting=False,
            latency_measurement=True,
            memory_measurement=False,
        )
        assert caps.model_loading is True
        assert caps.model_unloading is False
        assert caps.health_check is True
        assert caps.token_counting is False
        assert caps.latency_measurement is True
        assert caps.memory_measurement is False

    def test_capabilities_immutable(self):
        caps = RuntimeCapabilities(model_loading=True)
        with pytest.raises((TypeError, ValueError)):
            caps.model_loading = False

    def test_capabilities_equality(self):
        a = RuntimeCapabilities(model_loading=True, streaming=True)
        b = RuntimeCapabilities(model_loading=True, streaming=True)
        assert a == b

    def test_capabilities_inequality(self):
        a = RuntimeCapabilities(model_loading=True)
        b = RuntimeCapabilities(model_loading=False)
        assert a != b


class TestRuntimeMetadataSerialization:
    def test_canonical_json_round_trip(self):
        meta = RuntimeMetadata(
            runtime_name="test",
            runtime_version="1.0",
            gpu_acceleration=True,
            inference_parameters={"temp": 0.5},
        )
        json_str = meta.canonical_json()
        restored = RuntimeMetadata.from_canonical_json(json_str)
        assert meta == restored

    def test_sha256_consistency(self):
        meta = RuntimeMetadata(runtime_name="test", runtime_version="1.0")
        h1 = meta.sha256()
        h2 = meta.sha256()
        assert h1 == h2
        assert len(h1) == SHA256_HEX_LEN

    def test_sha256_differs_on_field_change(self):
        a = RuntimeMetadata(runtime_name="test", runtime_version="1.0")
        b = RuntimeMetadata(runtime_name="test", runtime_version="2.0")
        assert a.sha256() != b.sha256()

    def test_serialization_excludes_none(self):
        meta = RuntimeMetadata(runtime_name="t", runtime_version=None)
        d = meta.canonical_dict()
        assert "runtime_version" not in d

    def test_serialization_includes_false_bools(self):
        meta = RuntimeMetadata(runtime_name="t", gpu_acceleration=False)
        d = meta.canonical_dict()
        assert d.get("gpu_acceleration") is False

    def test_serialization_includes_capabilities(self):
        caps = RuntimeCapabilities(streaming=True)
        meta = RuntimeMetadata(runtime_name="t", capabilities=caps)
        d = meta.canonical_dict()
        assert d["capabilities"]["streaming"] is True

    def test_canonical_dict_structure(self):
        meta = RuntimeMetadata(runtime_name="t", runtime_version="1.0")
        d = meta.canonical_dict()
        assert isinstance(d, dict)
        assert d["runtime_name"] == "t"

    def test_round_trip_with_capabilities(self):
        caps = RuntimeCapabilities(model_loading=True, streaming=True)
        meta = RuntimeMetadata(runtime_name="t", capabilities=caps)
        json_str = meta.canonical_json()
        restored = RuntimeMetadata.from_canonical_json(json_str)
        assert meta == restored
        assert restored.capabilities.model_loading is True

    def test_round_trip_with_nested_metadata(self):
        meta = RuntimeMetadata(runtime_name="t", metadata={"nested": {"key": "val"}})
        restored = RuntimeMetadata.from_canonical_json(meta.canonical_json())
        assert meta == restored


class TestRuntimeMetadataEquality:
    def test_equality_same_values(self):
        a = RuntimeMetadata(runtime_name="t", runtime_version="1.0")
        b = RuntimeMetadata(runtime_name="t", runtime_version="1.0")
        assert a == b

    def test_inequality_different_name(self):
        a = RuntimeMetadata(runtime_name="a")
        b = RuntimeMetadata(runtime_name="b")
        assert a != b

    def test_inequality_different_version(self):
        a = RuntimeMetadata(runtime_name="t", runtime_version="1.0")
        b = RuntimeMetadata(runtime_name="t", runtime_version="2.0")
        assert a != b

    def test_inequality_different_capabilities(self):
        a = RuntimeMetadata(runtime_name="t", capabilities=RuntimeCapabilities(streaming=True))
        b = RuntimeMetadata(runtime_name="t", capabilities=RuntimeCapabilities(streaming=False))
        assert a != b

    def test_equality_with_capabilities(self):
        caps = RuntimeCapabilities(streaming=True)
        a = RuntimeMetadata(runtime_name="t", capabilities=caps)
        b = RuntimeMetadata(runtime_name="t", capabilities=caps)
        assert a == b

    def test_copy_is_equal(self):
        meta = RuntimeMetadata(runtime_name="t", runtime_version="1.0")
        copied = copy.deepcopy(meta)
        assert meta == copied

    def test_canonical_json_preserves_gpu_devices(self):
        meta = RuntimeMetadata(runtime_name="t", gpu_devices=["A100", "V100"])
        restored = RuntimeMetadata.from_canonical_json(meta.canonical_json())
        assert restored.gpu_devices == ["A100", "V100"]
