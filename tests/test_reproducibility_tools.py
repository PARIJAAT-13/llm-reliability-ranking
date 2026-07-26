"""Tests for reproducibility tools: dataset verification, manifest, and archive validation."""

import json
import pathlib
import zipfile
from typing import Any

import pytest

from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.reproducibility.dataset_verifier import DatasetVerifier
from llm_reliability.reproducibility.environment import EnvironmentCapture
from llm_reliability.reproducibility.manifest import ManifestGenerator
from llm_reliability.reproducibility.validate import ArchiveValidator
from tests.ranking_test_helpers import create_mock_metric
from tests.statistics_test_helpers import create_mock_ranking

# =========================================================================
# DatasetVerifier tests
# =========================================================================


def test_dataset_verifier_verify_exists(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text('{"key": "value"}', encoding="utf-8")

    verifier = DatasetVerifier()
    result = verifier.verify(path)

    assert result["exists"] is True
    assert result["is_directory"] is False
    assert result["sha256"] is not None
    assert result["error"] is None


def test_dataset_verifier_verify_missing(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "nonexistent.json"

    verifier = DatasetVerifier()
    result = verifier.verify(path)

    assert result["exists"] is False
    assert result["sha256"] is None
    assert result["error"] is not None
    assert "does not exist" in result["error"]


def test_dataset_verifier_verify_checksum_match(tmp_path: pathlib.Path) -> None:
    content = '{"key": "value"}'
    path = tmp_path / "dataset.json"
    path.write_text(content, encoding="utf-8")

    verifier = DatasetVerifier()
    result = verifier.verify(path)

    assert verifier.verify_checksum(path, result["sha256"]) is True


def test_dataset_verifier_verify_checksum_mismatch(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text("original content", encoding="utf-8")

    verifier = DatasetVerifier()

    assert verifier.verify_checksum(path, "f" * 64) is False


def test_dataset_verifier_verify_directory(tmp_path: pathlib.Path) -> None:
    sub = tmp_path / "subdir"
    sub.mkdir(parents=True)
    (sub / "a.txt").write_text("aaa", encoding="utf-8")
    (sub / "b.txt").write_text("bbb", encoding="utf-8")

    verifier = DatasetVerifier()
    result = verifier.verify(sub)

    assert result["exists"] is True
    assert result["is_directory"] is True
    assert result["sha256"] is not None


# =========================================================================
# Manifest tests
# =========================================================================


def test_manifest_generation(tmp_path: pathlib.Path) -> None:
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9, benchmark="mock"),
    ]
    summary = ExperimentSummary(
        experiment_id="manifest-test-001",
        experiment_name="Manifest Generation Test",
        metrics=metrics,
        config_snapshot={"base_seed": 42},
    )

    env = EnvironmentCapture.capture()
    gen = ManifestGenerator()
    manifest = gen.build(summary, environment=env)

    assert manifest.experiment_id == "manifest-test-001"
    assert manifest.experiment_name == "Manifest Generation Test"
    assert manifest.config_hash != ""
    assert len(manifest.record_hashes.metrics) == 1

    dest = tmp_path / "manifest.json"
    gen.save(manifest, dest)
    assert dest.exists()

    reloaded = gen.load(dest)
    assert reloaded.experiment_id == manifest.experiment_id
    assert reloaded.config_hash == manifest.config_hash


def test_manifest_contains_required_fields(tmp_path: pathlib.Path) -> None:
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9, benchmark="mock"),
    ]
    summary = ExperimentSummary(
        experiment_id="manifest-fields-test",
        experiment_name="Field Coverage Test",
        metrics=metrics,
        config_snapshot={"seed": 7},
    )

    env = EnvironmentCapture.capture()
    gen = ManifestGenerator()
    manifest = gen.build(summary, environment=env)

    assert manifest.manifest_version == "1.0"
    assert manifest.python_version != ""
    assert isinstance(manifest.seeds, list)
    assert manifest.record_hashes.executions == []
    assert manifest.record_hashes.evaluations == []
    assert len(manifest.record_hashes.metrics) == 1
    assert manifest.environment_file == "environment.json"

    dest = tmp_path / "manifest.json"
    gen.save(manifest, dest)

    with open(dest, encoding="utf-8") as f:
        data = json.load(f)

    for field in ("experiment_id", "experiment_name", "created_at", "config_hash", "record_hashes"):
        assert field in data, f"Missing required field: {field}"


# =========================================================================
# ArchiveValidator tests
# =========================================================================


def _create_valid_archive(tmp_path: pathlib.Path) -> pathlib.Path:
    """Helper to build a minimal valid experiment ZIP archive."""
    archive_dir = tmp_path / "archive_contents"
    archive_dir.mkdir()

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "experiment_id": "val-test-001",
        "experiment_name": "Validation Test",
        "created_at": "2026-07-25T00:00:00+00:00",
        "git_commit": None,
        "python_version": "3.11.0",
        "seeds": [42],
        "config_hash": "a" * 64,
        "record_hashes": {"executions": [], "evaluations": [], "metrics": [], "rankings": []},
        "environment_file": "environment.json",
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (archive_dir / "environment.json").write_text('{"python_version": "3.11.0"}', encoding="utf-8")
    (archive_dir / "CITATION.cff").write_text("cff-version: 1.2.0", encoding="utf-8")
    (archive_dir / "CHECKLIST.md").write_text("# Checklist", encoding="utf-8")
    (archive_dir / "README.md").write_text("# Readme", encoding="utf-8")

    zip_path = tmp_path / "experiment.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in archive_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(archive_dir))

    return zip_path


def test_archive_validation_valid(tmp_path: pathlib.Path) -> None:
    zip_path = _create_valid_archive(tmp_path)

    validator = ArchiveValidator()
    report = validator.validate(zip_path)

    assert report["valid"] is True
    assert report["manifest_valid"] is True
    assert report["checksums_match"] is True
    assert report["required_files_present"] is True
    assert report["missing_files"] == []
    assert report["corrupted_files"] == []


def test_archive_validation_missing_files(tmp_path: pathlib.Path) -> None:
    archive_dir = tmp_path / "archive_contents"
    archive_dir.mkdir()

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "experiment_id": "missing-test",
        "experiment_name": "Missing Test",
        "created_at": "2026-07-25T00:00:00+00:00",
        "git_commit": None,
        "python_version": "3.11.0",
        "seeds": [42],
        "config_hash": "b" * 64,
        "record_hashes": {"executions": [], "evaluations": [], "metrics": [], "rankings": []},
        "environment_file": "environment.json",
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (archive_dir / "environment.json").write_text("{}", encoding="utf-8")
    (archive_dir / "CITATION.cff").write_text("", encoding="utf-8")
    (archive_dir / "README.md").write_text("", encoding="utf-8")

    zip_path = tmp_path / "partial_experiment.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in archive_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(archive_dir))

    validator = ArchiveValidator()
    report = validator.validate(zip_path)

    assert report["valid"] is False
    assert "CHECKLIST.md" in report["missing_files"]
    assert report["required_files_present"] is False


def test_archive_validation_corrupted(tmp_path: pathlib.Path) -> None:
    archive_dir = tmp_path / "archive_contents"
    archive_dir.mkdir()

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "experiment_id": "corrupt-test",
        "experiment_name": "Corruption Test",
        "created_at": "2026-07-25T00:00:00+00:00",
        "git_commit": None,
        "python_version": "3.11.0",
        "seeds": [42],
        "config_hash": "c" * 64,
        "record_hashes": {"executions": [], "evaluations": [], "metrics": [], "rankings": []},
        "environment_file": "environment.json",
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (archive_dir / "environment.json").write_text('{"python_version": "3.11.0"}', encoding="utf-8")
    (archive_dir / "CITATION.cff").write_text("cff-version: 1.2.0", encoding="utf-8")
    (archive_dir / "CHECKLIST.md").write_text("# Checklist", encoding="utf-8")
    (archive_dir / "README.md").write_text("# Readme", encoding="utf-8")

    zip_path = tmp_path / "corrupted_experiment.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in archive_dir.rglob("*"):
            if file_path.is_file():
                if file_path.name == "manifest.json":
                    zf.writestr("manifest.json", "{corrupted json")
                else:
                    zf.write(file_path, file_path.relative_to(archive_dir))

    validator = ArchiveValidator()
    report = validator.validate(zip_path)

    assert report["valid"] is False
    assert report["manifest_valid"] is False
    assert report["required_files_present"] is True
