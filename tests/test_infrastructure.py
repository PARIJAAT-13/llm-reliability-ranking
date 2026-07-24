"""Tests for Docker and release automation infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_dockerfile_exists():
    assert Path("Dockerfile").exists()


def test_dockerfile_readable():
    content = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python" in content
    assert "ENTRYPOINT" in content


def test_docker_compose_exists():
    assert Path("docker-compose.yml").exists()


def test_docker_compose_readable():
    content = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "services" in content


def test_release_workflow_exists():
    path = Path(".github/workflows/release.yml")
    assert path.exists()


def test_release_workflow_readable():
    content = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "on:" in content
    assert "push:" in content
    assert "tags:" in content
    assert "pytest" in content
