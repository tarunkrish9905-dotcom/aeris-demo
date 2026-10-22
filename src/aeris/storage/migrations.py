"""
AERIS Schema Migration Engine.

Applies versioned schema migrations sequentially to the SQLite metadata repository.
Ensures transaction safety and schema version tracking.
"""

from __future__ import annotations

import datetime
import pathlib
import sqlite3

from aeris.storage.exceptions import DatabaseMigrationError

CURRENT_TARGET_VERSION = 1


def get_current_schema_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied schema version, or 0 if uninitialized."""
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        if cursor.fetchone() is None:
            return 0

        cursor = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error as exc:
        raise DatabaseMigrationError(f"Failed to query schema version: {exc}") from exc


def apply_migrations(conn: sqlite3.Connection) -> int:
    """
    Apply any unapplied migrations to the target database connection.

    Returns
    -------
    int
        The final schema version after migration.
    """
    current_version = get_current_schema_version(conn)

    if current_version >= CURRENT_TARGET_VERSION:
        return current_version

    # Migration 1: schema_v1.sql
    if current_version < 1:
        _apply_migration_v1(conn)

    return get_current_schema_version(conn)


def _apply_migration_v1(conn: sqlite3.Connection) -> None:
    """Apply schema migration v1."""
    sql_path = pathlib.Path(__file__).parent / "schema_v1.sql"
    if not sql_path.exists():
        raise DatabaseMigrationError(f"Migration script missing: {sql_path}")

    try:
        sql_content = sql_path.read_text(encoding="utf-8")
        with conn:
            conn.executescript(sql_content)
            now_iso = datetime.datetime.now(datetime.UTC).isoformat()
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
                (1, now_iso, "Initial AERIS schema v1: projects, datasets, runs, jobs, reports"),
            )
    except sqlite3.Error as exc:
        raise DatabaseMigrationError(f"Failed executing schema_v1.sql: {exc}") from exc
