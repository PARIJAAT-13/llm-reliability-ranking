"""Tests for shared serialization utilities."""

from llm_reliability.utils.serialization import SerializableModel


class _SampleModel(SerializableModel):
    alpha: int
    beta: str = "default"


def test_canonical_json_sorts_keys() -> None:
    model = _SampleModel(alpha=1, beta="x")
    assert model.canonical_json() == '{"alpha":1,"beta":"x"}'


def test_round_trip() -> None:
    model = _SampleModel(alpha=42)
    restored = _SampleModel.from_canonical_json(model.canonical_json())
    assert model == restored
    assert model.sha256() == restored.sha256()
