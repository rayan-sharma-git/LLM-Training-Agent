"""WebSocket handler for real-time analysis progress."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

# The analysis pipeline itself is imported (not re-implemented) so the
# WebSocket and REST transports can never drift apart.
from api.routes import _execute_analysis_pipeline
from core.config import get_active_model, get_active_provider
from core.errors import AnalysisError, StorageError
from core.security import resolve_workspace_root
from core.redaction import scrub, scrub_error
from chat.store import save_pipeline_snapshot
from storage.analysis_store import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    complete_run,
    fail_run,
    start_run,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        # list.remove() raises ValueError when the socket is already gone, which
        # would mask the real error inside a `finally` block. Remove defensively
        # and tolerate concurrent disconnects.
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def send_event(self, event_type: str, data: Dict[str, Any]):
        """Send event to all connected clients.

        A socket that fails to receive is dropped from the registry so a dead
        connection can never be retried (or leak memory) on later events.
        """
        event = {"type": event_type, "timestamp": datetime.utcnow().isoformat(), "data": data}
        # Iterate over a snapshot: a failing send below mutates the registry.
        for connection in list(self.active_connections):
            try:
                await connection.send_json(event)
            except Exception as e:
                logger.warning(f"Dropping WebSocket connection that failed to receive: {scrub_error(e)}")
                self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/ws/analysis")
async def analysis_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis progress.

    The socket is always removed from the registry, whatever ends the loop:
    a normal disconnect, malformed client input, or an unexpected error.
    """
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                await manager.send_event(
                    "error", {"error": "The message was not valid JSON."}
                )
                continue
            if not isinstance(data, dict):
                await manager.send_event(
                    "error", {"error": "The message must be a JSON object."}
                )
                continue
            if data.get("action") == "analyze":
                project_path = data.get("projectPath")
                if not isinstance(project_path, str) or not project_path.strip():
                    await manager.send_event(
                        "error", {"error": "projectPath is required to start an analysis."}
                    )
                    continue
                await run_analysis(project_path, websocket)
    except WebSocketDisconnect:
        logger.debug("Analysis WebSocket disconnected.")
    except Exception as e:
        # scrub_error keeps credentials and oversized payloads out of the log.
        logger.error(f"WebSocket error: {scrub_error(e)}")
    finally:
        # Guarantees the connection cannot leak on any exit path.
        manager.disconnect(websocket)


async def run_analysis(project_path: str, websocket: WebSocket):
    """Run analysis, stream progress and persist the run.

    Follows the same lifecycle as the REST endpoint (running →
    completed/partial/failed), so WebSocket runs appear in history too and a
    crash never leaves a run silently marked successful.
    """
    # resolve_workspace_root raises HTTPException for a path outside the
    # authorized roots. That must become a clean "failed" event, not an
    # unhandled error that also leaves a run stuck in "running" state.
    try:
        project_path = str(resolve_workspace_root(project_path))
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        message = str(detail.get("message") or "The requested project is outside the authorized workspace.")
        await manager.send_event("failed", {"error": scrub(message), "analysisId": None})
        return

    analysis_id: Optional[str] = None
    failures: List[Dict[str, Any]] = []

    if project_path:
        try:
            analysis_id = await start_run(
                project_path,
                project_name=Path(project_path).name,
                config_snapshot={
                    "provider": get_active_provider(),
                    "model": get_active_model(),
                    "transport": "websocket",
                },
            )
        except StorageError as error:
            logger.error(f"Could not record analysis start: {error.message}")

    try:
        await manager.send_event(
            "analysis_started",
            {"projectPath": project_path, "analysisId": analysis_id},
        )

        async def progress(stage: str, detail: str) -> None:
            """Stream each pipeline stage to the client as it happens."""
            await manager.send_event("analyzer_progress", {"stage": stage, "status": detail})

        # Reuse the REST pipeline verbatim — the WebSocket transport must not
        # re-implement the analyzer sequence, or the two would drift apart.
        pipeline = await _execute_analysis_pipeline(project_path, progress=progress)

        context = pipeline["context"]
        report = pipeline["report"]
        results = pipeline["results"]
        recommendations = pipeline["recommendations"]
        prediction_result = pipeline["prediction"]
        complete_analysis = pipeline["complete_analysis"]
        failures = pipeline["failures"]

        save_pipeline_snapshot(
            project_path,
            context=context,
            report=report,
            recommendations=recommendations,
            analyzer_results=results,
            prediction=prediction_result,
            cost=pipeline["cost"],
            gpu_time_estimate=pipeline["gpu_time_estimate"],
            hardware_detection=pipeline["hardware_result"],
        )

        critical_failures = [f for f in failures if f.get("critical")]
        run_status = STATUS_PARTIAL if critical_failures else STATUS_COMPLETED
        if analysis_id:
            try:
                await complete_run(
                    analysis_id,
                    context=context,
                    report=report,
                    results=results,
                    recommendations=recommendations,
                    partial_failures=failures,
                    health_score=getattr(context, "project_health_score", None),
                    readiness_score=getattr(context, "training_readiness_score", None),
                    status=run_status,
                )
            except StorageError as error:
                logger.error(f"Could not persist analysis results: {error.message}")

        report_payload = report.model_dump(mode="json") if hasattr(report, "model_dump") else report
        await manager.send_event("report_ready", {"report": report_payload})
        if complete_analysis is not None:
            await manager.send_event(
                "analysis_ready", {"analysis": complete_analysis.model_dump(mode="json")}
            )
        await manager.send_event(
            "completed",
            {
                "status": "Analysis complete",
                "analysisId": analysis_id,
                "analysisStatus": run_status,
                "partialFailures": failures,
            },
        )
    except AnalysisError as e:
        await _fail_ws_run(analysis_id, e)
        await manager.send_event("failed", {"error": scrub(e.message), "analysisId": analysis_id})
    except Exception as e:
        logger.error(f"Analysis failed: {scrub_error(e)}")
        await _fail_ws_run(analysis_id, e)
        await manager.send_event("failed", {"error": scrub_error(e), "analysisId": analysis_id})


async def _fail_ws_run(analysis_id: Optional[str], error: Any) -> None:
    """Best-effort: mark the run failed so history never shows it as running."""
    if not analysis_id:
        return
    try:
        await fail_run(analysis_id, error)
    except StorageError as persist_error:
        logger.error(f"Could not mark run failed: {persist_error.message}")