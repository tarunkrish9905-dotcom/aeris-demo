"""
AERIS Database Connection and Transaction Management.

Provides connection factory, foreign-key enforcement, WAL mode configuration,
and transaction context management using the Python standard library sqlite3.
"""

from __future__ import annotations

import contextlib
import pathlib
import sqlite3
from collections.abc import Generator

from aeris.storage.exceptions import DatabaseError


def get_db_connection(
    db_path: str | pathlib.Path,
    wal_mode: bool = True,
    timeout: float = 30.0,
) -> sqlite3.Connection:
    """
    Open and configure a SQLite connection.

    Parameters
    ----------
    db_path : str | pathlib.Path
        Path to the SQLite database file, or ':memory:'.
    wal_mode : bool
        Whether to enable Write-Ahead Logging (WAL) mode.
    timeout : float
        Timeout in seconds to wait for database locks.

    Returns
    -------
    sqlite3.Connection
        Configured SQLite connection with foreign keys enabled and Row factory set.
    """
    db_str = str(db_path)
    try:
        if db_str != ":memory:":
            pathlib.Path(db_str).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            db_str,
            timeout=timeout,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row

        # Mandatory: Enforce foreign key constraints on every connection
        conn.execute("PRAGMA foreign_keys = ON;")

        # Enable WAL journal mode for high-concurrency read operations
        if wal_mode and db_str != ":memory:":
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("PRAGMA journal_mode = WAL;")

        return conn
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to open database connection at '{db_path}': {exc}") from exc


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for atomic database transactions.

    Commits on clean exit, rolls back on exception.
    """
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise DatabaseError(f"Transaction rolled back due to error: {exc}") from exc
