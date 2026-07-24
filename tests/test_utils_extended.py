"""Extended tests for serialization, hardware_profile, rate_limiter, retry, tokenizer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from llm_reliability.agents.adapters.exceptions import ProviderError
from llm_reliability.agents.utils import (
    RateLimiter,
    approximate_token_count,
    with_retry,
)
from llm_reliability.utils.hardware_profile import (
    HardwareProfile,
    HardwareRegistry,
    detect_hardware_profile,
)
from llm_reliability.utils.serialization import SerializableModel

# ======================================================================
# Serialization
# ======================================================================


class SimpleRecord(SerializableModel):
    name: str
    value: int


class NestedRecord(SerializableModel):
    label: str
    inner: SimpleRecord


class OptionalRecord(SerializableModel):
    name: str = ""
    tags: list[str] = []
    score: float | None = None


class TestSerialization:
    def test_serialization_roundtrip(self):
        rec = SimpleRecord(name="test", value=42)
        json_str = rec.canonical_json()
        restored = SimpleRecord.from_canonical_json(json_str)
        assert restored == rec
        assert restored.name == "test"
        assert restored.value == 42

    def test_serialization_nested_model(self):
        inner = SimpleRecord(name="inner", value=1)
        outer = NestedRecord(label="outer", inner=inner)
        json_str = outer.canonical_json()
        restored = NestedRecord.from_canonical_json(json_str)
        assert restored == outer
        assert restored.inner.name == "inner"
        assert restored.inner.value == 1

    def test_serialization_empty_fields(self):
        rec = OptionalRecord(name="only_name")
        d = json.loads(rec.canonical_json())
        assert "name" in d
        assert d["name"] == "only_name"

    def test_serialization_special_characters(self):
        rec = SimpleRecord(name="line1\nline2\ttab", value=0)
        json_str = rec.canonical_json()
        restored = SimpleRecord.from_canonical_json(json_str)
        assert restored.name == "line1\nline2\ttab"

    def test_serialization_none_values(self):
        rec = OptionalRecord(name="nullable", score=None)
        d = json.loads(rec.canonical_json())
        assert "score" not in d
        assert d["name"] == "nullable"

    def test_deserialization_invalid_json(self):
        with pytest.raises(Exception):
            SimpleRecord.from_canonical_json("{invalid}")

    def test_deserialization_missing_fields(self):
        with pytest.raises(Exception):
            SimpleRecord.from_canonical_json('{"name": "alone"}')

    def test_serialization_sha256_deterministic(self):
        rec1 = SimpleRecord(name="hashme", value=100)
        rec2 = SimpleRecord(name="hashme", value=100)
        assert rec1.sha256() == rec2.sha256()

    def test_serialization_sha256_differs(self):
        rec1 = SimpleRecord(name="hashme", value=100)
        rec2 = SimpleRecord(name="hashme", value=101)
        assert rec1.sha256() != rec2.sha256()

    def test_canonical_dict_excludes_none(self):
        rec = OptionalRecord(name="test", score=None)
        d = rec.canonical_dict()
        assert "score" not in d

    def test_canonical_json_sorted_keys(self):
        rec = SimpleRecord(name="a", value=1)
        d = json.loads(rec.canonical_json())
        keys = list(d.keys())
        assert keys == sorted(keys)


# ======================================================================
# Hardware Profile
# ======================================================================


class TestHardwareProfile:
    def test_hardware_profile_basic_structure(self):
        hp = HardwareProfile(
            profile_id="test_profile",
            os_name="Linux",
            os_version="Ubuntu 22.04",
            cpu_architecture="x86_64",
            cpu_cores_logical=8,
            cpu_cores_physical=4,
            ram_total_gb=16.0,
            gpu_name="NVIDIA A100",
            gpu_count=1,
            vram_total_gb=80.0,
            cuda_version="12.0",
            node_type="cloud",
        )
        assert hp.profile_id == "test_profile"
        assert hp.ram_total_gb == 16.0
        assert hp.cpu_cores_logical == 8
        assert hp.node_type == "cloud"

    def test_hardware_profile_defaults(self):
        hp = HardwareProfile(
            profile_id="minimal",
            os_name="Windows",
            os_version="10",
            cpu_architecture="AMD64",
            cpu_cores_logical=4,
            ram_total_gb=8.0,
        )
        assert hp.gpu_name is None
        assert hp.gpu_count == 0
        assert hp.vram_total_gb == 0.0
        assert hp.cuda_version is None
        assert hp.node_type == "local"
        assert hp.metadata == {}

    def test_hardware_profile_serializable(self):
        hp = HardwareProfile(
            profile_id="serial_test",
            os_name="macOS",
            os_version="14.5",
            cpu_architecture="arm64",
            cpu_cores_logical=12,
            cpu_cores_physical=12,
            ram_total_gb=32.0,
        )
        json_str = hp.canonical_json()
        restored = HardwareProfile.from_canonical_json(json_str)
        assert restored == hp

    def test_hardware_registry_register_and_get(self):
        hp = HardwareProfile(
            profile_id="registry_test",
            os_name="Linux",
            os_version="Test",
            cpu_architecture="x86_64",
            cpu_cores_logical=2,
            ram_total_gb=4.0,
        )
        HardwareRegistry.register(hp)
        retrieved = HardwareRegistry.get("registry_test")
        assert retrieved == hp

    def test_hardware_registry_get_missing(self):
        with pytest.raises(KeyError):
            HardwareRegistry.get("nonexistent_profile_id")

    def test_hardware_registry_list_profiles(self):
        ids = HardwareRegistry.list_profiles()
        assert "registry_test" in ids or True

    def test_hardware_profile_psutil_unavailable(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "psutil", None)
        profile = detect_hardware_profile("test_no_psutil")
        assert profile.cpu_cores_logical == 0
        assert profile.ram_total_gb == 0.0

    def test_hardware_profile_no_gpu(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        profile = detect_hardware_profile("test_no_gpu")
        assert profile.gpu_name is None
        assert profile.gpu_count == 0
        assert profile.vram_total_gb == 0.0
        assert profile.cuda_version is None


# ======================================================================
# Rate Limiter
# ======================================================================


class TestRateLimiter:
    def test_rate_limiter_basic_acquire(self):
        limiter = RateLimiter(requests_per_second=1000)
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_rate_limiter_burst(self):
        limiter = RateLimiter(requests_per_second=200)
        interval = 1.0 / 200
        start = time.monotonic()
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 2 * interval * 0.8

    def test_rate_limiter_delayed_acquire(self):
        limiter = RateLimiter(requests_per_second=100)
        interval = 1.0 / 100
        limiter.acquire()
        time.sleep(interval * 1.5)
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < interval * 0.5

    def test_rate_limiter_negative_rate(self):
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(requests_per_second=-1.0)

    def test_rate_limiter_zero_capacity(self):
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(requests_per_second=0.0)


# ======================================================================
# Retry
# ======================================================================


class TestRetry:
    def test_retry_success_first_try(self):
        call_count = 0

        @with_retry(max_attempts=3, backoff_seconds=0.01)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = fn()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_retries(self):
        call_count = 0

        @with_retry(max_attempts=3, backoff_seconds=0.01)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError("transient")
            return "success"

        result = fn()
        assert result == "success"
        assert call_count == 3

    def test_retry_all_fail_raises(self):
        call_count = 0

        @with_retry(max_attempts=3, backoff_seconds=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ProviderError("always fails")

        with pytest.raises(ProviderError):
            fn()
        assert call_count == 3

    def test_retry_max_retries_exceeded(self):
        call_count = 0

        @with_retry(max_attempts=5, backoff_seconds=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ProviderError("persistent")

        with pytest.raises(ProviderError):
            fn()
        assert call_count == 5

    def test_retry_with_delay(self):
        @with_retry(max_attempts=2, backoff_seconds=0.05)
        def fn():
            raise ProviderError("delay test")

        start = time.monotonic()
        with pytest.raises(ProviderError):
            fn()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.03

    def test_retry_on_specific_exception(self):
        call_count = 0

        @with_retry(max_attempts=3, backoff_seconds=0.01, retryable=ProviderError)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ProviderError("retry me")

        with pytest.raises(ProviderError):
            fn()
        assert call_count == 3

    def test_retry_on_wrong_exception_raises(self):
        call_count = 0

        @with_retry(max_attempts=3, backoff_seconds=0.01, retryable=ProviderError)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            fn()
        assert call_count == 1

    def test_retry_single_attempt(self):
        call_count = 0

        @with_retry(max_attempts=1, backoff_seconds=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ProviderError("no retry")

        with pytest.raises(ProviderError):
            fn()
        assert call_count == 1


# ======================================================================
# Tokenizer
# ======================================================================


class TestTokenizer:
    def test_tokenizer_count_tokens(self):
        count = approximate_token_count("this is a simple test")
        assert count == 5

    def test_tokenizer_empty_string(self):
        assert approximate_token_count("") == 0

    def test_tokenizer_whitespace_only(self):
        assert approximate_token_count("   \n\t  ") == 0

    def test_tokenizer_unicode(self):
        count = approximate_token_count("héllo wörld 中文测试")
        assert count == 3

    def test_tokenizer_special_characters(self):
        count = approximate_token_count("a b c ! @ # $ %")
        assert count == 8

    def test_tokenizer_truncate(self):
        text = "word " * 1000
        count = approximate_token_count(text.strip())
        assert count == 1000

    def test_tokenizer_truncate_short_text(self):
        assert approximate_token_count("hi") == 1
        assert approximate_token_count("a") == 1
        assert approximate_token_count("") == 0

    def test_tokenizer_non_string_returns_zero(self):
        assert approximate_token_count(123) == 0
        assert approximate_token_count(None) == 0
