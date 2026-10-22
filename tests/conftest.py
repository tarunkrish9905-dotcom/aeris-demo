"""
Shared pytest fixtures and configuration for the AERIS test suite.

This module is automatically loaded by pytest before test execution.
Fixtures defined here are available to all tests without explicit import.
"""

import pathlib

import pytest

# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root() -> pathlib.Path:
    """Return the absolute path to the AERIS project root directory."""
    # conftest.py lives at tests/conftest.py → parent is project root
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    """Return the absolute path to the test fixtures directory.

    Synthetic datasets and other test data files will be stored here
    once Milestone 1 is implemented.
    """
    return pathlib.Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def csv_fixtures_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Return the absolute path to the CSV test fixtures directory."""
    return fixtures_dir / "csv"
