"""Shared test fixtures."""

from llm_reliability.configs import Configuration

CONFIG_HASH = "a" * 64
TIMESTAMP = "2026-01-01T00:00:00+00:00"


def make_configuration(**overrides: object) -> Configuration:
    """Return a valid pilot configuration with optional field overrides."""
    defaults = {
        "experiment_name": "pilot",
        "benchmark": "agentboard",
        "agent": "mock_agent",
        "llm": "gpt-4",
        "prompt_version": "v1",
        "dataset_version": "1.0",
        "seed": 42,
        "repetitions": 5,
        "perturbations": ("typo",),
        "fault_injection": False,
    }
    defaults.update(overrides)
    return Configuration(**defaults)
