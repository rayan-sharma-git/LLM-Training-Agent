"""FastAPI application entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.errors import StorageError
from core.logging import setup_logging
from storage.database import get_database
from api.routes import router as api_router
from api.websocket import router as ws_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    setup_logging()
    logger.info("LLM Training Agent Backend starting")

    # Eager schema initialization: fail fast (with a useful log line) if the
    # database cannot be created/migrated.  The app still starts so non-DB
    # endpoints keep working; DB endpoints then return explicit 503 errors.
    try:
        database = get_database()
        await database.init_schema()
        logger.info("Storage ready: %s", database.database_url)
    except StorageError as error:
        logger.error(
            "Storage initialization failed (%s): %s",
            error.error_code,
            error.message,
        )

    yield

    logger.info("LLM Training Agent Backend shutting down")
    try:
        await get_database().dispose()
    except Exception as error:  # pragma: no cover - best-effort shutdown
        logger.warning("Storage dispose failed: %s", error)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered engineering assistant for LLM fine-tuning",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("main:app", host=settings.host, port=settings.port, workers=settings.workers, reload=True)