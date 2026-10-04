"""Database configuration, connection management and schema initialization.

Storage facts (see ``docs/persistence.md`` for the full audit):

* Engine: SQLAlchemy 2 async + ``aiosqlite`` over a **single local SQLite
  file** — agent-wide state, deliberately *not* stored inside user projects.
* The default URL (``sqlite+aiosqlite:///./llm_training_agent.db``) is
  anchored to the backend package directory, so the database location does
  not depend on the process working directory.  ``DATABASE_URL`` (env/.env)
  overrides it, including for absolute paths elsewhere.
* ``PRAGMA journal_mode=WAL`` + ``busy_timeout`` make concurrent reads/writes
  from the single uvicorn worker safe for the expected local usage.
* Schema creation/migration happens lazily on first use (and eagerly during
  app startup) via ``storage.migrations._run_migrations``; failures raise
  :class:`~core.errors.StorageError` with an actionable message — the app
  never fabricates empty results when the database cannot be read.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from core.config import get_settings
from core.errors import StorageError
from core.redaction import scrub_error

logger = logging.getLogger(__name__)

#: Repository root of the backend package — the stable anchor for the default
#: (relative) database file, independent of the process working directory.
BACKEND_ROOT = Path(__file__).resolve().parent.parent

_SQLITE_URL_PREFIX = "sqlite+aiosqlite:///"


class Base(DeclarativeBase):
    """Base class for ORM models."""
    pass


def resolve_database_url(database_url: Optional[str] = None) -> str:
    """Return an absolute, ready-to-use database URL.

    Relative SQLite paths are anchored to :data:`BACKEND_ROOT` so the same
    database file is used whether the backend is started from the repo root,
    the backend folder, or by the VS Code extension (whose ``cwd`` is the
    backend directory).  In-memory URLs and non-SQLite URLs are unchanged.
    """
    url = database_url or get_settings().database_url
    if not url.startswith(_SQLITE_URL_PREFIX):
        return url

    raw_path = url[len(_SQLITE_URL_PREFIX):]
    if not raw_path or raw_path == ":memory:":
        return url

    path = Path(raw_path)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    # Forward slashes are valid for sqlite3 on Windows and avoid URL escaping
    # problems with spaces/backslashes in the path.
    return _SQLITE_URL_PREFIX + str(path).replace("\\", "/")


def _sqlite_file_path(database_url: str) -> Optional[Path]:
    """Extract the on-disk file path from *database_url* (None if not a file)."""
    if not database_url.startswith(_SQLITE_URL_PREFIX):
        return None
    raw_path = database_url[len(_SQLITE_URL_PREFIX):]
    if not raw_path or raw_path == ":memory:":
        return None
    return Path(raw_path)


def storage_error_from(error: BaseException, database_url: str = "") -> StorageError:
    """Translate a low-level failure into an actionable :class:`StorageError`.

    Classifies corruption / locking / permission problems so the API can
    return a precise, credential-scrubbed error instead of a generic 500 or,
    worse, fabricated empty data.
    """
    if isinstance(error, StorageError):
        return error

    detail = scrub_error(error)
    lower = str(error).lower()
    path_text = _sqlite_file_path(database_url)

    if "not a database" in lower or "malformed" in lower or "encrypted" in lower:
        return StorageError(
            f"The database file{' at ' + str(path_text) if path_text else ''} is "
            f"corrupted and cannot be read ({detail}). Move or delete the file "
            "to reinitialize storage — previously stored experiments and "
            "analysis history cannot be recovered from a corrupted file.",
            error_code="STORAGE_CORRUPT",
        )
    if "locked" in lower or "busy" in lower:
        return StorageError(
            f"The database is locked by another process ({detail}). Retry in a "
            "few seconds; if the problem persists, close other backend instances.",
            error_code="STORAGE_LOCKED",
        )
    if "permission" in lower or "unable to open database file" in lower or "readonly" in lower:
        return StorageError(
            f"The database file cannot be opened for writing"
            f"{' at ' + str(path_text) if path_text else ''} ({detail}). Check "
            "file/directory permissions, or set DATABASE_URL to a writable location.",
            error_code="STORAGE_PERMISSION",
        )
    return StorageError(
        f"Storage operation failed ({detail}). History endpoints may be "
        "unavailable until the problem is resolved.",
        error_code="STORAGE_UNAVAILABLE",
    )


class Database:
    """Database connection manager (one engine per process)."""

    def __init__(self, database_url: str):
        self.database_url = resolve_database_url(database_url)
        self.engine: AsyncEngine = self._create_engine(self.database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @staticmethod
    def _create_engine(database_url: str) -> AsyncEngine:
        # Make sure the directory for file-based databases exists *before*
        # SQLite tries to create the file (fresh clone / first run safety).
        file_path = _sqlite_file_path(database_url)
        if file_path is not None:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise storage_error_from(error, database_url) from error

        # NullPool: every checkout opens/closes a dedicated sqlite connection.
        # For this local single-user app that removes all cross-event-loop
        # pooling hazards (backend restarts, tests) at negligible cost.
        engine = create_async_engine(
            database_url,
            echo=False,
            poolclass=NullPool,
            connect_args={"timeout": 30},
        )

        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # pragma: no cover
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

        return engine

    async def create_tables(self) -> None:
        """Create/migrate all tables (kept for backwards compatibility)."""
        await self.init_schema()

    async def init_schema(self) -> None:
        """Create/migrate the schema. Idempotent; raises StorageError on failure."""
        async with self._init_lock:
            if self._initialized:
                return
            try:
                # Local import: migrations import storage.models which imports
                # storage.database (Base) — deferred to avoid an import cycle.
                from storage.migrations import _run_migrations

                async with self.engine.begin() as conn:
                    await conn.run_sync(_run_migrations)
            except Exception as error:
                raise storage_error_from(error, self.database_url) from error
            self._initialized = True

    async def ensure_initialized(self) -> None:
        """Initialize the schema on first use (no-op afterwards)."""
        await self.init_schema()

    async def dispose(self) -> None:
        """Dispose engine."""
        try:
            await self.engine.dispose()
        finally:
            self._initialized = False


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get database instance (lazily created from settings)."""
    global _db
    if _db is None:
        _db = Database(get_settings().database_url)
    return _db


def configure_database(database_url: str) -> Database:
    """Point the global database at *database_url* (used by tests)."""
    global _db
    _db = Database(database_url)
    return _db


def reset_database() -> None:
    """Drop the global database instance without touching the file."""
    global _db
    _db = None


def get_session_factory() -> async_sessionmaker:
    """Get session factory."""
    return get_database().session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a session with guaranteed cleanup.

    Guarantees the schema exists (creating/migrating the database on first
    use), and rolls the transaction back if the body raises, so a failed
    operation never leaves an open transaction behind.
    """
    database = get_database()
    await database.ensure_initialized()
    async with database.session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


__all__ = [
    "Base",
    "BACKEND_ROOT",
    "Database",
    "configure_database",
    "get_database",
    "get_session",
    "get_session_factory",
    "reset_database",
    "resolve_database_url",
    "storage_error_from",
]