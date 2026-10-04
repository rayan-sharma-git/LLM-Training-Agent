"""Experiment tracking service (SQLite-backed).

Contract:
* Records are **project-scoped** (identity = normalized absolute path), so
  experiments of one project can never be listed under another project.
* **Estimated vs actual are never mixed**: ``estimates`` holds what the agent
  predicted *before* training (e.g. GPU-time estimate, prediction, cost),
  while ``metrics`` / ``actualTrainingTimeSeconds`` hold only values actually
  observed or reported by the user *after* training — empty when unknown.
  Nothing is ever invented to fill them.
* ``status``: ``estimated`` (pre-training record) | ``completed`` | ``failed``
  (with a scrubbed ``error`` explaining the failure).
* All stored text/JSON is secret-scrubbed via ``core.redaction``.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select

from chat.store import normalize_project_path
from core.errors import StorageError, ValidationError
from core.redaction import scrub, scrub_obj
from storage.analysis_store import get_or_create_project
from storage.database import get_database, get_session, storage_error_from
from storage.models import Analysis, Experiment, Project

logger = logging.getLogger(__name__)

#: Allowed experiment lifecycle states.
VALID_STATUSES = frozenset({"estimated", "completed", "failed"})

#: Honest placeholder for the NOT NULL ``dataset_version`` column when the
#: caller does not provide a dataset identifier (never a fabricated value).
UNKNOWN_DATASET = "unspecified"


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _number(value: Any) -> Optional[float]:
    """Coerce to float strictly — returns None instead of guessing."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_api(experiment: Experiment, project: Optional[Project] = None) -> Dict[str, Any]:
    """Serialize an experiment row into the API shape (camelCase)."""
    return {
        "experimentId": experiment.experiment_id,
        "projectPath": project.project_path if project else None,
        "projectName": project.project_name if project else None,
        "status": experiment.status,
        "model": experiment.model,
        "tokenizer": experiment.tokenizer,
        "fineTuningMethod": experiment.fine_tuning_method,
        "datasetVersion": experiment.dataset_version,
        "hyperparameters": experiment.hyperparameters or {},
        "hardware": experiment.hardware or {},
        # ESTIMATED — agent's pre-training predictions (may be empty).
        "estimates": experiment.estimates or {},
        # ACTUAL — observed/reported after training (empty when unknown).
        "metrics": experiment.metrics or {},
        "actualTrainingTimeSeconds": experiment.actual_training_time,
        "analysisId": experiment.analysis_id,
        "notes": experiment.notes,
        "tags": experiment.tags or [],
        "artifacts": experiment.artifacts or [],
        "error": experiment.error,
        "createdAt": _iso(experiment.created_at),
    }


class ExperimentService:
    """Create / list / get / delete / compare experiment records."""

    async def save_experiment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and persist one experiment; returns the API representation."""
        status = str(payload.get("status") or "estimated")
        if status not in VALID_STATUSES:
            raise ValidationError(
                f"Invalid status '{status}'. Expected one of: {sorted(VALID_STATUSES)}"
            )
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValidationError("A 'model' identifier is required to record an experiment.")

        try:
            async with get_session() as session:
                project_id = await get_or_create_project(
                    session,
                    payload["projectPath"],
                    project_name=payload.get("projectName"),
                )

                analysis_id = payload.get("analysisId")
                if analysis_id:
                    analysis = await session.get(Analysis, analysis_id)
                    if analysis is None:
                        raise ValidationError(f"analysisId '{analysis_id}' does not exist.")
                    if analysis.project_id != project_id:
                        raise ValidationError(
                            f"analysisId '{analysis_id}' belongs to a different project "
                            "and cannot be attached to this experiment."
                        )

                experiment = Experiment(
                    experiment_id=str(uuid.uuid4()),
                    project_id=project_id,
                    dataset_version=str(payload.get("datasetVersion") or UNKNOWN_DATASET)[:255],
                    model=scrub(model)[:255],
                    tokenizer=scrub(str(payload["tokenizer"]))[:255] if payload.get("tokenizer") else None,
                    hyperparameters=scrub_obj(payload.get("hyperparameters") or {}),
                    metrics=scrub_obj(payload.get("metrics") or {}),
                    notes=scrub(str(payload["notes"]))[:4000] if payload.get("notes") else None,
                    tags=scrub_obj(list(payload.get("tags") or [])),
                    artifacts=scrub_obj(list(payload.get("artifacts") or [])),
                    status=status,
                    fine_tuning_method=scrub(str(payload["fineTuningMethod"]))[:100]
                    if payload.get("fineTuningMethod") else None,
                    hardware=scrub_obj(payload.get("hardware") or {}),
                    estimates=scrub_obj(payload.get("estimates") or {}),
                    actual_training_time=_number(payload.get("actualTrainingTimeSeconds")),
                    analysis_id=analysis_id,
                    error=scrub(str(payload["error"]))[:2000] if payload.get("error") else None,
                )
                session.add(experiment)
                await session.commit()
                await session.refresh(experiment)

                project = await session.get(Project, project_id)
                logger.info(
                    "Experiment %s saved (status=%s)",
                    experiment.experiment_id,
                    status,
                )
                return _to_api(experiment, project)
        except StorageError:
            raise
        except ValidationError:
            raise
        except Exception as error:
            raise storage_error_from(error, get_database().database_url) from error

    async def list_experiments(
        self, project_path: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Experiments of one project, newest first (never other projects)."""
        try:
            async with get_session() as session:
                identity = normalize_project_path(project_path)
                result = await session.execute(
                    select(Project).where(Project.project_path == identity)
                )
                project = result.scalar_one_or_none()
                if project is None:
                    return []
                rows = await session.execute(
                    select(Experiment)
                    .where(Experiment.project_id == project.project_id)
                    .order_by(desc(Experiment.created_at), desc(Experiment.experiment_id))
                    .limit(max(1, min(int(limit or 50), 200)))
                )
                return [_to_api(row, project) for row in rows.scalars().all()]
        except StorageError:
            raise
        except Exception as error:
            raise storage_error_from(error, get_database().database_url) from error

    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """One experiment by id (None when it does not exist)."""
        try:
            async with get_session() as session:
                experiment = await session.get(Experiment, experiment_id)
                if experiment is None:
                    return None
                project = await session.get(Project, experiment.project_id)
                return _to_api(experiment, project)
        except StorageError:
            raise
        except Exception as error:
            raise storage_error_from(error, get_database().database_url) from error

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Explicitly delete one experiment (user action only — no auto-pruning)."""
        try:
            async with get_session() as session:
                experiment = await session.get(Experiment, experiment_id)
                if experiment is None:
                    return False
                await session.delete(experiment)
                await session.commit()
                logger.info("Experiment %s deleted", experiment_id)
                return True
        except StorageError:
            raise
        except Exception as error:
            raise storage_error_from(error, get_database().database_url) from error

    async def compare_experiments(
        self,
        experiment_ids: List[str],
        project_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the requested experiments in the requested order.

        Records retain model, method, dataset, hyperparameters, hardware,
        estimates and actual values so a client can line them up side by side.
        When ``project_path`` is given, every record must belong to it.
        """
        if len(experiment_ids) < 2:
            raise ValidationError("At least two experimentIds are required to compare.")
        try:
            async with get_session() as session:
                project_id = None
                if project_path:
                    result = await session.execute(
                        select(Project).where(
                            Project.project_path == normalize_project_path(project_path)
                        )
                    )
                    project = result.scalar_one_or_none()
                    if project is None:
                        # A projectPath was supplied but matches no known project.
                        # Returning the experiments anyway would silently defeat
                        # project isolation, so this is a hard validation error.
                        raise ValidationError(
                            "No stored experiments exist for the requested project."
                        )
                    project_id = project.project_id

                ordered: List[Dict[str, Any]] = []
                missing: List[str] = []
                for experiment_id in experiment_ids:
                    experiment = await session.get(Experiment, experiment_id)
                    if experiment is None:
                        missing.append(experiment_id)
                        continue
                    if project_id is not None and experiment.project_id != project_id:
                        raise ValidationError(
                            f"Experiment '{experiment_id}' belongs to a different project."
                        )
                    experiment_project = await session.get(Project, experiment.project_id)
                    ordered.append(_to_api(experiment, experiment_project))
                if missing:
                    raise ValidationError(
                        f"Experiments not found: {', '.join(missing)}",
                        details={"missingExperimentIds": missing},
                    )
                return ordered
        except StorageError:
            raise
        except ValidationError:
            raise
        except Exception as error:
            raise storage_error_from(error, get_database().database_url) from error


__all__ = ["ExperimentService", "VALID_STATUSES", "UNKNOWN_DATASET"]