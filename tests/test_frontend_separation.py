"""
Architectural Boundary Tests: Core Isolation from Frontend and Web Frameworks.

Verifies that AERIS core analytical modules remain strictly decoupled from
Dash, FastAPI, Starlette, and frontend presentation layers.
"""

from __future__ import annotations

import ast
import pathlib

PROHIBITED_CORE_IMPORTS = {
    "dash",
    "dash_bootstrap_components",
    "fastapi",
    "starlette",
    "a2wsgi",
    "flask",
    "werkzeug",
    "aeris.frontend",
}


def test_core_modules_do_not_import_frontend_or_web_frameworks() -> None:
    """Scan all Python source files in src/aeris/core/ to verify strict architectural isolation."""
    core_dir = pathlib.Path(__file__).parents[1] / "src" / "aeris" / "core"
    assert core_dir.exists() and core_dir.is_dir()

    violations: list[str] = []

    for py_file in core_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception as exc:
            violations.append(f"Failed to parse {py_file}: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prohibited in PROHIBITED_CORE_IMPORTS:
                        if alias.name == prohibited or alias.name.startswith(f"{prohibited}."):
                            violations.append(
                                f"{py_file.name}:{node.lineno} imports '{alias.name}'"
                            )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for prohibited in PROHIBITED_CORE_IMPORTS:
                    if node.module == prohibited or node.module.startswith(f"{prohibited}."):
                        violations.append(
                            f"{py_file.name}:{node.lineno} imports from '{node.module}'"
                        )

    assert not violations, "Architectural boundary violations found in core:\n" + "\n".join(
        violations
    )
