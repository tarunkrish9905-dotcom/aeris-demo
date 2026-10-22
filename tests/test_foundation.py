"""
Phase 0 foundation smoke tests.

Validates that the AERIS package is importable, versioned, and that test
infrastructure fixtures resolve correctly. Does NOT test any domain logic or
future milestone functionality.
"""

import pathlib

import aeris
import aeris.core


def test_package_version() -> None:
    """Verify AERIS package metadata and version string."""
    assert hasattr(aeris, "__version__")
    assert aeris.__version__ == "0.1.0"


def test_core_package_importable() -> None:
    """Verify AERIS core package is importable."""
    assert aeris.core is not None
    assert aeris.core.__doc__ is not None


def test_project_root_fixture(project_root: pathlib.Path) -> None:
    """Verify project_root fixture points to directory containing pyproject.toml."""
    assert project_root.is_dir()
    assert (project_root / "pyproject.toml").is_file()


def test_fixtures_dir_fixture(fixtures_dir: pathlib.Path, csv_fixtures_dir: pathlib.Path) -> None:
    """Verify fixture paths resolve to expected structure relative to tests."""
    assert fixtures_dir.name == "fixtures"
    assert csv_fixtures_dir == fixtures_dir / "csv"
