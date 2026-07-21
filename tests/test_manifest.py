"""Tests for reproducibility manifest generation."""

import tempfile
import pathlib
import pytest
import json

from llm_reliability.reproducibility.manifest import ManifestGenerator
from llm_reliability.reproducibility.environment import EnvironmentCapture
from llm_reliability.reporting.summary import ExperimentSummary
from tests.ranking_test_helpers import create_mock_metric


def test_manifest_generation():
    # Setup mock summary
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9, benchmark="mock-bench"),
    ]
    summary = ExperimentSummary(
        experiment_id="test-exp-123",
        experiment_name="Reproducibility Test",
        metrics=metrics,
        config_snapshot={"base_seed": 42},
    )
    
    # Capture environment
    env = EnvironmentCapture.capture()
    
    # Generate manifest
    gen = ManifestGenerator()
    manifest = gen.build(summary, environment=env)
    
    assert manifest.experiment_id == "test-exp-123"
    assert manifest.experiment_name == "Reproducibility Test"
    assert manifest.config_hash != ""
    assert len(manifest.record_hashes.metrics) == 1
    
    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = pathlib.Path(tmpdir) / "manifest.json"
        saved_path = gen.save(manifest, dest_path)
        assert saved_path.exists()
        
        # Verify JSON
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["experiment_id"] == "test-exp-123"
            assert "config_hash" in data
            
        reloaded = gen.load(saved_path)
        assert reloaded.experiment_id == manifest.experiment_id
        assert reloaded.config_hash == manifest.config_hash
