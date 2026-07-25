"""
Runtime metadata models — standardized metadata for all inference runtimes.

``RuntimeMetadata`` captures every property of an inference runtime that
may affect reproducibility: version, backend, quantization, hardware
acceleration, and inference parameters.  The metadata is automatically
collected by the runtime adapter and embedded in every experiment artifact.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.utils.serialization import SerializableModel


class RuntimeCapabilities(SerializableModel):
    """Describes which optional capabilities a runtime adapter supports."""

    model_loading: bool = False
    model_unloading: bool = False
    health_check: bool = False
    token_counting: bool = False
    latency_measurement: bool = False
    memory_measurement: bool = False
    version_detection: bool = False
    batch_inference: bool = False
    streaming: bool = False
    quantization: bool = False
    gpu_acceleration: bool = False
    multi_gpu: bool = False
    context_length_config: bool = False
    thread_config: bool = False
    custom_parameters: bool = False


class RuntimeMetadata(SerializableModel):
    """Standardized metadata captured from an inference runtime.

    Fields are populated by the runtime adapter's ``runtime_metadata()``
    method.  Unsupported fields remain ``None`` (or their default).
    """

    runtime_name: str
    runtime_version: str | None = None
    backend: str | None = None
    api_version: str | None = None
    execution_mode: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    gpu_acceleration: bool = False
    gpu_devices: list[str] = []
    thread_count: int | None = None
    batch_size: int | None = None
    max_concurrent_requests: int | None = None
    inference_parameters: dict[str, Any] = {}
    capabilities: RuntimeCapabilities = RuntimeCapabilities()
    metadata: dict[str, Any] = {}
