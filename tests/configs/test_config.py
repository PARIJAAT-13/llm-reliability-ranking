"""Tests for Configuration (Artifact 1)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_reliability.configs import CONFIG_VERSION, Configuration
from tests.conftest import make_configuration


def test_hash_is_deterministic() -> None:
    assert make_configuration().sha256() == make_configuration().sha256()


def test_json_is_deterministic() -> None:
    assert make_configuration().canonical_json() == make_configuration().canonical_json()


def test_round_trip_via_model_dump() -> None:
    original = make_configuration()
    restored = Configuration.model_validate(original.model_dump())
    assert original == restored


def test_round_trip_via_canonical_json() -> None:
    original = make_configuration()
    restored = Configuration.from_canonical_json(original.canonical_json())
    assert original == restored
    assert original.sha256() == restored.sha256()


def test_immutable() -> None:
    cfg = make_configuration()
    with pytest.raises(ValidationError):
        cfg.seed = 99  # type: ignore[misc]


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Configuration(
            experiment_name="pilot",
            benchmark="agentboard",
            agent="mock_agent",
            llm="gpt-4",
            prompt_version="v1",
            dataset_version="1.0",
            seed=42,
            repetitions=5,
            unknown_field=True,  # type: ignore[call-arg]
        )


def test_rejects_invalid_seed() -> None:
    with pytest.raises(ValidationError):
        make_configuration(seed=-1)


def test_rejects_invalid_repetitions() -> None:
    with pytest.raises(ValidationError):
        make_configuration(repetitions=0)


def test_rejects_invalid_version_format() -> None:
    with pytest.raises(ValidationError):
        make_configuration(version="not-semver")


def test_default_version() -> None:
    cfg = make_configuration()
    assert cfg.version == CONFIG_VERSION


def test_equality() -> None:
    assert make_configuration() == make_configuration()
    assert make_configuration(seed=1) != make_configuration(seed=2)


def test_file_round_trip(tmp_path: Path) -> None:
    original = make_configuration()
    file_path = tmp_path / "config.json"
    original.write_file(file_path)
    restored = Configuration.from_file(file_path)
    assert original == restored


def test_perturbations_are_tuple() -> None:
    cfg = make_configuration(perturbations=["typo", "paraphrase"])
    assert cfg.perturbations == ("typo", "paraphrase")
