"""Additive SQLite schema migrations.

Design (deliberately minimal for a local, single-user application):

* ``PRAGMA user_version`` stores the current schema version.
* A **fresh** database gets ``Base.metadata.create_all`` and jumps straight to
  ``SCHEMA_VERSION``.
* An **existing** database is only ever evolved *additively* — missing columns
  are added with ``ALTER TABLE ... ADD COLUMN`` (idempotent, checked through
  ``PRAGMA table_info``).  Existing rows are never deleted or rewritten.
* A database whose ``user_version`` is **newer** than this code understands is
  rejected with a clear :class:`~core.errors.StorageError` instead of being
  silently modified.

Every migration carries a unique identifier (see ``MIGRATIONS``) so the applied
history can be logged and reasoned about.

Only SQLite ``ADD COLUMN``-style changes are automated here.  Column type
changes, drops or renames require a purpose-built migration (table rebuild)
and must be added explicitly — never by silently recreating the database.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from core.errors import StorageError

logger = logging.getLogger(__name__)

#: Target schema version of this codebase.
SCHEMA_VERSION = 2

#: ``{table: {column: column-DDL}}`` — additive column definitions per
#: migration.  Types are SQLite type names used verbatim in ``ADD COLUMN``.
_MIGRATION_COLUMNS: Dict[int, Dict[str, Dict[str, str]]] = {
    2: {
        "analyses": {
            "config_snapshot": "JSON",
            "context": "JSON",
            "report": "JSON",
            "error": "TEXT",
            "partial_failures": "JSON",
        },
        "dataset_reports": {"payload": "JSON"},
        "prompt_reports": {"payload": "JSON"},
        "hyperparameter_reports": {"payload": "JSON"},
        "model_reports": {"payload": "JSON"},
        "predictions": {"payload": "JSON"},
        "cost_estimates": {"payload": "JSON"},
        "recommendations": {"payload": "JSON"},
        "experiments": {
            "status": "TEXT NOT NULL DEFAULT 'estimated'",
            "fine_tuning_method": "TEXT",
            "hardware": "JSON",
            "estimates": "JSON",
            "actual_training_time": "REAL",
            "analysis_id": "TEXT REFERENCES analyses(analysis_id)",
            "error": "TEXT",
        },
    },
}

#: Unique (id, version) identifiers, ordered ascending.
MIGRATIONS: List[Tuple[str, int]] = [
    ("0001_initial_schema", 1),
    ("0002_analysis_run_and_experiment_fields", 2),
]


def _run_migrations(sync_conn) -> int:
    """Run pending migrations on *sync_conn*; return the final version.

    Executed via ``conn.run_sync(...)`` from ``storage.database``.
    Import of ``storage.models`` is deferred to avoid an import cycle.
    """
    # Registers every ORM model on Base.metadata (also avoids circular import).
    import storage.models  # noqa: F401
    from storage.database import Base

    current = sync_conn.exec_driver_sql("PRAGMA user_version").scalar() or 0

    if current > SCHEMA_VERSION:
        raise StorageError(
            f"Database schema version {current} is newer than the supported "
            f"version {SCHEMA_VERSION}. The database was created by a newer "
            "version of the LLM Training Agent backend; refusing to modify it. "
            "Upgrade the backend or point DATABASE_URL at a different file.",
            error_code="STORAGE_SCHEMA_MISMATCH",
        )

    inspector = inspect(sync_conn)
    known_tables = set(inspector.get_table_names())

    if current == 0 and "projects" not in known_tables:
        # Fresh database: create everything at the latest version.
        Base.metadata.create_all(bind=sync_conn)
        sync_conn.exec_driver_sql(f"PRAGMA user_version = {SCHEMA_VERSION}")
        logger.info("Created fresh database schema at version %s", SCHEMA_VERSION)
        return SCHEMA_VERSION

    if current == 0:
        # Pre-migration database created by the originally shipped schema.
        current = 1

    # Create any table that is missing entirely (additive, checkfirst —
    # existing tables and their data are never touched).
    Base.metadata.create_all(bind=sync_conn)
    inspector = inspect(sync_conn)  # refresh: pick up any just-created tables
    known_tables = set(inspector.get_table_names())

    for migration_id, version in MIGRATIONS:
        if version <= current:
            continue
        columns_by_table = _MIGRATION_COLUMNS.get(version, {})
        # Add missing columns to existing tables.
        for table_name, columns in columns_by_table.items():
            if table_name not in known_tables:
                continue
            existing = {col["name"] for col in inspector.get_columns(table_name)}
            for column_name, ddl in columns.items():
                if column_name in existing:
                    continue
                try:
                    sync_conn.exec_driver_sql(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"
                    )
                except SQLAlchemyError as error:
                    raise StorageError(
                        f"Migration {migration_id} failed on table '{table_name}' "
                        f"while adding column '{column_name}': {error}",
                        error_code="STORAGE_SCHEMA_MISMATCH",
                    ) from error
        sync_conn.exec_driver_sql(f"PRAGMA user_version = {version}")
        logger.info("Applied schema migration %s (version %s)", migration_id, version)
        current = version

    return current


__all__ = ["SCHEMA_VERSION", "MIGRATIONS", "_run_migrations"]
