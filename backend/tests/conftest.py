"""Pytest configuration and fixtures."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from storage.database import Base
import storage.models  # noqa: F401  # register ORM models on Base.metadata
from models.schemas import ProjectContext
from api.routes import router
from core.security import ALLOWED_ROOTS_ENV, ALLOWED_ROOTS_FILE_ENV


def pytest_configure(config):
    config.AsyncClient = AsyncClient
    config.ASGITransport = ASGITransport
    # Tests use temporary workspaces outside the repository checkout.
    # Tests include synthetic Windows project identities that are not real
    # directories; filesystem routes still require an existing authorized root.
    # Assign (do not setdefault): a stray value inherited from the developer's
    # shell would otherwise silently make every filesystem test fail with 403.
    os.environ[ALLOWED_ROOTS_ENV] = (
        str(Path(tempfile.gettempdir()).resolve()) + os.pathsep + "C:\\"
    )
    os.environ.pop(ALLOWED_ROOTS_FILE_ENV, None)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session():
    """In-memory SQLite session for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
async def client(db_session):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_project_context():
    return ProjectContext(
        project_name="test_project",
        project_path="/tmp/test_project",
        detected_framework="huggingface",
        dataset_paths=["data/train.json"],
        prompt_templates=["prompts/default.txt"],
    )
