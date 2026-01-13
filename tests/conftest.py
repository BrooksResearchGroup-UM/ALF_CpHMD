"""Pytest configuration and fixtures for cphmd tests."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def toppar_dir(project_root: Path) -> Path:
    """Return the toppar directory."""
    return project_root / "toppar"


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "padding": 10.0,
        "salt_concentration": 0.15,
        "temperature": 298.15,
        "crystal_type": "OCTAHEDRAL",
    }
