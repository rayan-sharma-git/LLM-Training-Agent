"""FastAPI route handlers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from models.schemas import (
    ProjectContext,
    DatasetAnalysisResult,
    DatasetCleaningResult,
    PromptAnalysisResult,
    HyperparameterAnalysisResult,
    ModelAnalysisResult,
    PredictionResult,
    CostEstimate,
    Recommendation,
    EngineeringReport,
    ChatMessage,
)
from hardware.gpu_detector import detect_gpus, GPUSpec
from hardware.gpu_time_estimator import (
    GPUTimeEstimator,
    TrainingConfig,
    extract_training_config_from_context,
    run_calibration_benchmark,
)
from scanner.scanner import ProjectScanner
from scanner.context_builder import ContextBuilder
from analyzers.dataset_analyzer import DatasetAnalyzer
from analyzers.prompt_analyzer import PromptAnalyzer
from cleaning.dataset_cleaner import DatasetCleaner
from analyzers.hyperparameter_analyzer import HyperparameterAnalyzer
from analyzers.model_advisor import ModelAdvisor
from analyzers.cost_estimator import CostEstimator
from analyzers.use_case_detector import UseCaseDetector
from analysis.orchestrator import AnalysisOrchestrator
from prediction.engine import PredictionEngine
from recommendation.engine import RecommendationEngine
from reports.generator import ReportGenerator
from chat.engine import ChatEngine
from chat.store import store, save_pipeline_snapshot
from core.errors import AppError, StorageError, ValidationError
from core.config import get_settings, get_active_provider, get_active_model
from core import runtime_config
from core.redaction import scrub, scrub_error
from core.security import (
    MAX_CHUNK_CHARS,
    MAX_CHUNK_RECORDS,
    MAX_DATASET_ENTRIES,
    MAX_DATASET_FILES,
    bounded_int,
    bounded_text,
    resolve_dataset_path,
    resolve_workspace_root,
    validate_provider_url,
)
from ai.providers import list_providers, get_models, get_provider
from experiments.service import ExperimentService
from storage.analysis_store import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    complete_run,
    fail_run,
    get_latest_run,
    get_run,
    list_runs,
    run_failure,
    start_run,
    upsert_project_metadata,
)

logger = logging.getLogger(__name__)
router = APIRouter()

#: Stateless experiment service (all state lives in SQLite).
_experiment_service = ExperimentService()


# ---------------------------------------------------------------------------
# Storage error translation
#
# Database failures never surface as fake empty payloads: they become an
# explicit 503 carrying a scrubbed, actionable message.
# ---------------------------------------------------------------------------

def _storage_http(error: StorageError) -> HTTPException:
    """Map a StorageError to an HTTP 503 with a useful message."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "errorCode": error.error_code,
            "message": error.message,
            "details": error.details,
        },
    )


def _app_http(error: AppError) -> HTTPException:
    """Map application errors to HTTP 400 (503 for storage problems)."""
    code = status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(error, StorageError) else status.HTTP_400_BAD_REQUEST
    return HTTPException(
        status_code=code,
        detail={
            "errorCode": error.error_code,
            "message": error.message,
            "details": error.details,
        },
    )


def _require_project_path(project_path: Optional[str]) -> str:
    """Validate a projectPath against the server-authorized workspace roots."""
    value = bounded_text(project_path, name="projectPath", maximum=4096)
    try:
        return str(resolve_workspace_root(value, must_exist=False))
    except HTTPException:
        raise
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail={"errorCode": "INVALID_PATH", "message": "Invalid project path."}) from exc


# ---------------------------------------------------------------------------
# Helpers for safe typed-result reconstruction
#
# The conversion helpers live in ``api/result_mapping.py`` so that the WebSocket
# pipeline uses exactly the same fallbacks (no duplicated/diverging logic).
# ---------------------------------------------------------------------------

from api.result_mapping import (  # noqa: E402
    safe_model as _safe_model,
    default_dataset_result as _default_dataset_result,
    default_prompt_result as _default_prompt_result,
    default_hp_result as _default_hp_result,
    default_model_result as _default_model_result,
    default_cost_result as _default_cost_result,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Coerce a JSON/string flag into a bool (used for optional request flags)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Simple health endpoint - used by the extension to check backend availability."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Provider / configuration
# ---------------------------------------------------------------------------

class ProviderConfigRequest(BaseModel):
    provider: str
    model: str


class SaveConfigRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    apiKey: Optional[str] = None
    baseUrl: Optional[str] = None


class TestConnectionRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    apiKey: Optional[str] = None
    baseUrl: Optional[str] = None


@router.get("/providers")
async def get_providers():
    """Return all available AI providers and their supported models."""
    settings = get_settings()
    provider_info = list_providers()
    result = {}
    for name, info in provider_info.items():
        configured = True
        if info["requires_key"]:
            key_field = f"{name}_api_key" if name != "openai_compatible" else "openai_compatible_api_key"
            if name == "openai_compatible":
                configured = bool(runtime_config.get_api_key("openai_compatible") or settings.openai_compatible_api_key)
            else:
                configured = bool(runtime_config.get_api_key(name) or getattr(settings, key_field, None))
        result[name] = {
            "requiresKey": info["requires_key"],
            "models": info["models"],
            "description": info["description"],
            "configured": configured,
        }
    return {"providers": result}


@router.get("/provider/models")
async def get_provider_models(provider: str = "ollama"):
    """Get available models for the specified provider."""
    try:
        models = get_models(provider)
        return {"provider": provider, "models": models}
    except Exception as e:
        return {"provider": provider, "models": [], "error": str(e)}


@router.post("/provider/select")
async def select_provider(request: ProviderConfigRequest):
    """Select active AI provider and model (persists in runtime config)."""
    valid_providers = list_providers()
    if request.provider not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{request.provider}'. Valid: {list(valid_providers.keys())}",
        )
    runtime_config.set_runtime_config("provider", request.provider)
    runtime_config.set_runtime_config("model", request.model)
    return {
        "provider": request.provider,
        "model": request.model,
        "message": f"Provider set to {request.provider}, model {request.model}.",
    }


@router.get("/config")
async def get_config():
    """Return current configuration."""
    settings = get_settings()
    provider = get_active_provider()
    model = get_active_model(provider)
    return {
        "default_provider": provider,
        "default_model": model,
        "ollama_base_url": runtime_config.get_runtime_config("ollama_base_url") or settings.ollama_base_url,
        "openai_compatible_base_url": runtime_config.get_runtime_config("openai_compatible_base_url") or settings.openai_compatible_base_url,
    }


@router.put("/config")
async def update_config(request: SaveConfigRequest):
    """Update configuration (provider, model, API keys, base URLs).

    API keys are stored in memory only (never written to disk).
    The extension sends them securely from VS Code SecretStorage.
    """
    updated: Dict[str, Any] = {}

    if request.provider:
        valid_providers = list_providers()
        if request.provider not in valid_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider '{request.provider}'. Valid: {list(valid_providers.keys())}",
            )
        runtime_config.set_runtime_config("provider", request.provider)
        updated["provider"] = request.provider

    if request.model:
        runtime_config.set_runtime_config("model", bounded_text(request.model, name="model", maximum=255))
        updated["model"] = bounded_text(request.model, name="model", maximum=255)

    if request.apiKey:
        provider = request.provider or get_active_provider()
        runtime_config.set_api_key(provider, bounded_text(request.apiKey, name="apiKey", maximum=4096))

    if request.baseUrl:
        provider = request.provider or get_active_provider()
        try:
            safe_url = validate_provider_url(request.baseUrl, allow_remote=provider != "ollama")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"errorCode": "INVALID_PROVIDER_URL", "message": str(exc)}) from exc
        if provider == "ollama":
            runtime_config.set_runtime_config("ollama_base_url", safe_url)
            updated["ollama_base_url"] = safe_url
        elif provider == "openai_compatible":
            runtime_config.set_runtime_config("openai_compatible_base_url", safe_url)
            updated["openai_compatible_base_url"] = safe_url

    return {"status": "updated", "changed": updated}


@router.delete("/config/key")
async def remove_api_key(provider: str):
    """Remove the in-memory API key for a provider."""
    runtime_config.remove_api_key(provider)
    return {"status": "removed", "provider": provider}


@router.post("/provider/test")
async def test_connection(request: TestConnectionRequest):
    """Test the connection to an AI provider."""
    provider_name = request.provider.lower()
    valid_providers = list_providers()
    if provider_name not in valid_providers:
        return {
            "success": False,
            "message": f"Unknown provider '{provider_name}'.",
            "error": "UNKNOWN_PROVIDER",
        }

    if request.apiKey:
        runtime_config.set_api_key(provider_name, request.apiKey)
    if request.baseUrl:
        if provider_name == "ollama":
            runtime_config.set_runtime_config("ollama_base_url", request.baseUrl)
        elif provider_name == "openai_compatible":
            runtime_config.set_runtime_config("openai_compatible_base_url", request.baseUrl)
    if request.model:
        runtime_config.set_runtime_config("model", request.model)

    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        return {
            "success": False,
            "message": str(e),
            "error": "PROVIDER_NOT_CONFIGURED",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to initialize provider: {str(e)}",
            "error": "PROVIDER_INIT_FAILED",
        }

    try:
        ok = await provider.validate_credentials()
        if ok:
            return {
                "success": True,
                "message": f"Connection to {provider_name} successful.",
                "provider": provider_name,
                "model": request.model or get_active_model(provider_name),
            }
        else:
            return {
                "success": False,
                "message": f"Connection to {provider_name} failed. Check your API key and model.",
                "error": "VALIDATION_FAILED",
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Connection test failed: {str(e)}",
            "error": "CONNECTION_FAILED",
        }


# ---------------------------------------------------------------------------
# Project analysis
# ---------------------------------------------------------------------------

async def _run_and_persist_analysis(
    project_path: str,
    *,
    clean_datasets: bool = False,
) -> Dict[str, Any]:
    """Execute the analysis pipeline and persist the run as one history record.

    * Opens a ``running`` run row first, so even a crash leaves a *failed*
      record instead of a silent gap.
    * Status: ``completed`` (no failures), ``partial`` (critical step failed),
      ``failed`` (pipeline raised).  A run with critical failures is never
      reported as successful.
    * Persistence is surfaced honestly via the ``persistence`` response field:
      a database problem never fabricates results, but it also never discards
      a successful analysis — both outcomes are visible to the client.
    """
    logger.info(f"Starting project analysis: {project_path}")

    analysis_id: Optional[str] = None
    persistence: Dict[str, Any] = {"status": "unavailable"}
    try:
        analysis_id = await start_run(
            project_path,
            project_name=Path(project_path).name,
            config_snapshot={
                "provider": get_active_provider(),
                "model": get_active_model(),
                "cleanDatasets": bool(clean_datasets),
            },
        )
        persistence = {"status": "saved", "analysisId": analysis_id}
    except StorageError as error:
        logger.error(f"Could not record analysis start: {error.message}")
        persistence = {
            "status": "unavailable",
            "errorCode": error.error_code,
            "error": error.message,
        }

    try:
        pipeline = await _execute_analysis_pipeline(
            project_path, clean_datasets=clean_datasets
        )
    except Exception as error:
        if analysis_id:
            try:
                await fail_run(analysis_id, error)
            except StorageError as persist_error:
                logger.error(f"Could not mark run failed: {persist_error.message}")
        raise

    context = pipeline["context"]
    report = pipeline["report"]
    results = pipeline["results"]
    failures = pipeline["failures"]

    # Feed the chat engine with this analysis (best-effort, project-scoped)
    # so chat can answer "why did you recommend X?" from real results.
    save_pipeline_snapshot(
        project_path,
        context=context,
        report=report,
        recommendations=pipeline["recommendations"],
        analyzer_results=results,
        prediction=pipeline["prediction"],
        cost=pipeline["cost"],
        gpu_time_estimate=pipeline["gpu_time_estimate"],
        hardware_detection=pipeline["hardware_result"],
    )

    critical_failures = [f for f in failures if f.get("critical")]
    run_status = STATUS_PARTIAL if critical_failures else STATUS_COMPLETED

    if analysis_id:
        try:
            saved = await complete_run(
                analysis_id,
                context=context,
                report=report,
                results=results,
                recommendations=pipeline["recommendations"],
                partial_failures=failures,
                health_score=context.project_health_score,
                readiness_score=context.training_readiness_score,
                status=run_status,
            )
            if not saved:
                persistence = {
                    "status": "missing",
                    "error": "The analysis run row disappeared before results "
                    "could be stored (was the database reset mid-run?).",
                }
        except StorageError as error:
            logger.error(f"Could not persist analysis results: {error.message}")
            persistence = {
                "status": "failed",
                "errorCode": error.error_code,
                "error": error.message,
            }

    response: Dict[str, Any] = {
        "project": context.model_dump(),
        "report": report.model_dump(),
        "analysisId": analysis_id,
        "analysisStatus": run_status,
        "partialFailures": failures,
        "persistence": persistence,
    }
    if results.get("dataset_cleaning") is not None:
        response["cleaning"] = results["dataset_cleaning"]
    if pipeline.get("complete_analysis") is not None:
        # The single authoritative end-to-end analysis, rendered for the chat window.
        response["analysis"] = pipeline["complete_analysis"].model_dump(mode="json")
    return response

async def _execute_analysis_pipeline(
    project_path: str,
    *,
    clean_datasets: bool = False,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run scan → context → analyzers → prediction → recommendations → report.

    Individual step failures are *collected* (with a criticality flag) instead
    of being silently dropped, so the caller can classify the run as
    ``completed`` or ``partial`` and persist the failures.  The function only
    raises when no context/report can be produced at all (the run then counts
    as ``failed``).

    ``progress`` is an optional async callback invoked as
    ``await progress(stage: str, detail: str)`` after each stage.  It exists so
    the WebSocket transport can stream progress while reusing this exact
    pipeline, instead of re-implementing the analyzer sequence.
    """
    async def emit(stage: str, detail: str) -> None:
        if progress is None:
            return
        try:
            await progress(stage, detail)
        except Exception as error:  # pragma: no cover - progress must never break a run
            logger.warning(f"Progress callback failed at '{stage}': {error}")

    # Step 1: Scan project
    await emit("scanner", "Scanning project...")
    scanner = ProjectScanner(project_path)
    scan_result = scanner.scan()

    # Step 2: Build context
    await emit("context", "Building project context...")
    context_builder = ContextBuilder()
    context = context_builder.build(scan_result)

    # Step 3: Run analyzers (with graceful degradation)
    results: Dict[str, Any] = {}
    failures: List[Dict[str, Any]] = []

    dataset_analyzer = DatasetAnalyzer()
    prompt_analyzer = PromptAnalyzer()
    hp_analyzer = HyperparameterAnalyzer()
    model_advisor = ModelAdvisor()
    cost_estimator = CostEstimator()

    # Dataset analysis
    await emit("dataset", "Analyzing dataset...")
    dataset_result_obj = None
    try:
        dataset_result_obj = await dataset_analyzer.analyze(context)
        results["dataset"] = dataset_result_obj.model_dump()
    except Exception as e:
        logger.error(f"DatasetAnalyzer failed: {e}")
        results["dataset"] = {"error": str(e), "quality_score": 0.0, "confidence": "low"}
        failures.append(run_failure("dataset", e, critical=True))

    # Prompt analysis
    await emit("prompt", "Analyzing prompts...")
    try:
        prompt_result_obj = await prompt_analyzer.analyze(context)
        results["prompt"] = prompt_result_obj.model_dump()
    except Exception as e:
        logger.error(f"PromptAnalyzer failed: {e}")
        results["prompt"] = {"error": str(e), "clarity_score": 0.5, "confidence": "low"}
        failures.append(run_failure("prompt", e, critical=True))

    # Hyperparameter analysis
    await emit("hyperparameters", "Analyzing training configuration...")
    hp_result_obj = None
    try:
        hp_result_obj = await hp_analyzer.analyze(context)
        results["hyperparameters"] = hp_result_obj.model_dump()
    except Exception as e:
        logger.error(f"HyperparameterAnalyzer failed: {e}")
        results["hyperparameters"] = {"error": str(e), "efficiency_score": 0.5, "confidence": "low"}
        failures.append(run_failure("hyperparameters", e, critical=True))

    # Model analysis
    await emit("model", "Evaluating models...")
    try:
        model_result_obj = await model_advisor.analyze(context)
        results["model"] = model_result_obj.model_dump()
    except Exception as e:
        logger.error(f"ModelAdvisor failed: {e}")
        results["model"] = {"error": str(e), "confidence": "low"}
        failures.append(run_failure("model", e, critical=True))

    # Cost estimation (receives the real dataset & hyperparameter results so
    # the estimates are derived from measured values, not placeholders)
    await emit("cost", "Estimating cost...")
    try:
        cost_result_obj = await cost_estimator.analyze(
            context, dataset_result=dataset_result_obj, hp_result=hp_result_obj,
        )
        results["cost"] = cost_result_obj.model_dump()
    except Exception as e:
        logger.error(f"CostEstimator failed: {e}")
        results["cost"] = {"error": str(e), "confidence": "low"}
        failures.append(run_failure("cost", e, critical=False))

    logger.info("All analyzers completed (with possible partial failures)")

    # Step 3b: Optional chunked dataset cleaning (opt-in, one LLM call per chunk)
    if clean_datasets and context.dataset_paths:
        try:
            cleaner = DatasetCleaner()
            cleaning_result = await cleaner.clean_project(context)
            results["dataset_cleaning"] = cleaning_result.model_dump()
            logger.info(
                f"Dataset cleaning finished: {cleaning_result.total_files} file(s), "
                f"{cleaning_result.total_chunks} chunk(s), "
                f"records preserved={cleaning_result.records_preserved}"
            )
        except Exception as e:
            logger.error(f"Dataset cleanup failed: {e}")
            results["dataset_cleaning"] = {"error": str(e)}
            failures.append(run_failure("dataset_cleaning", e, critical=False))

    # Step 4: Predictions (built from the real analyzer outputs)
    await emit("prediction", "Predicting training outcomes...")
    ds_result = _safe_model(results["dataset"], DatasetAnalysisResult, **_default_dataset_result().__dict__)
    hp_result = _safe_model(results["hyperparameters"], HyperparameterAnalysisResult, **_default_hp_result().__dict__)
    model_result = _safe_model(results["model"], ModelAnalysisResult, **_default_model_result().__dict__)

    try:
        prediction_engine = PredictionEngine()
        prediction_result = await prediction_engine.predict(
            context, ds_result, hp_result, model_result
        )
    except Exception as e:
        logger.error(f"PredictionEngine failed: {e}")
        prediction_result = PredictionResult(
            instruction_following_prediction="unknown",
            hallucination_risk="unknown",
            confidence="low",
        )
        failures.append(run_failure("prediction", e, critical=False))

    # Step 5: Recommendations
    await emit("recommendations", "Generating recommendations...")
    prompt_result = _safe_model(results["prompt"], PromptAnalysisResult, **_default_prompt_result().__dict__)
    cost_result = _safe_model(results["cost"], CostEstimate, **_default_cost_result().__dict__)

    try:
        rec_engine = RecommendationEngine()
        recommendations = await rec_engine.generate(
            context, ds_result, prompt_result, hp_result, model_result,
            cost_result, prediction_result
        )
    except Exception as e:
        logger.error(f"RecommendationEngine failed: {e}")
        recommendations = []
        failures.append(run_failure("recommendations", e, critical=False))

    # Step 5b: GPU detection & time estimation (integrated into analysis)
    await emit("hardware", "Analyzing hardware and GPU requirements...")
    gpu_time_estimate = None
    hardware_result = None
    try:
        hardware_result = detect_gpus().to_dict()
        training_config = extract_training_config_from_context(context)
        # The estimator can only see the dataset size if it is told: the
        # scanner's project_statistics carries no sample count, so without this
        # the estimate would silently degenerate to 0 seconds / 0 steps.
        # Measured analyzer outputs take precedence over file-scan guesses.
        if dataset_result_obj is not None and getattr(dataset_result_obj, "sample_count", 0):
            training_config.dataset_samples = int(dataset_result_obj.sample_count)
        if hp_result_obj is not None:
            if hp_result_obj.batch_size:
                training_config.batch_size = int(hp_result_obj.batch_size)
            if hp_result_obj.epochs:
                training_config.epochs = int(hp_result_obj.epochs)
            if hp_result_obj.gradient_accumulation:
                training_config.gradient_accumulation = int(hp_result_obj.gradient_accumulation)
            if hp_result_obj.sequence_length:
                training_config.sequence_length = int(hp_result_obj.sequence_length)
            if hp_result_obj.lora_rank:
                training_config.training_method = "lora"
        if model_result_obj is not None and getattr(model_result_obj, "parameter_count", None):
            try:
                training_config.param_count_billions = float(
                    str(model_result_obj.parameter_count).lower().replace("b", "").strip()
                )
            except (TypeError, ValueError):
                pass
        estimator = GPUTimeEstimator()
        estimate_result = estimator.estimate(training_config, mode="quick")
        gpu_time_estimate = estimate_result.to_dict()
    except Exception as e:
        logger.warning(f"GPU time estimation failed during analysis: {e}")
        failures.append(run_failure("gpu_time_estimate", e, critical=False))

    # Step 5c: Use-case detection (evidence-based, runs on the real dataset)
    await emit("use_case", "Identifying the use case...")
    use_case_result = None
    try:
        use_case_detector = UseCaseDetector()
        use_case_result = use_case_detector.analyze(context, dataset_result=dataset_result_obj)
        results["use_case"] = use_case_result.model_dump()
    except Exception as e:
        logger.error(f"UseCaseDetector failed: {e}")
        failures.append(run_failure("use_case", e, critical=False))

    # Step 6: Report
    try:
        report_gen = ReportGenerator()
        report = report_gen.generate(
            context, ds_result, prompt_result, hp_result, model_result,
            cost_result, prediction_result, recommendations,
            gpu_time_estimate=gpu_time_estimate,
            hardware_detection=hardware_result,
        )
    except Exception as e:
        logger.error(f"ReportGenerator failed: {e}")
        report = EngineeringReport(
            executive_summary=f"Analysis of {context.project_name} completed with errors.",
            project_health_score=0.0,
            training_readiness_score=0.0,
            prioritized_recommendations=[],
            action_plan=["Fix backend errors and re-run analysis."],
        )
        failures.append(run_failure("report", e, critical=True))

    # Step 7: Complete end-to-end analysis (composes every real output above)
    complete_analysis = None
    await emit("report", "Building the final analysis...")
    try:
        run_status = STATUS_PARTIAL if any(f.get("critical") for f in failures) else STATUS_COMPLETED
        complete_analysis = AnalysisOrchestrator().build(
            context,
            results,
            report=report,
            prediction=prediction_result,
            recommendations=recommendations,
            gpu_estimate=gpu_time_estimate,
            hardware=hardware_result,
            use_case=use_case_result,
            failures=failures,
            analysis_status=run_status,
        )
    except Exception as e:
        # A composition failure must never discard the analysis that already ran.
        logger.error(f"AnalysisOrchestrator failed: {e}")
        failures.append(run_failure("complete_analysis", e, critical=False))

    return {
        "context": context,
        "results": results,
        "recommendations": recommendations,
        "prediction": prediction_result,
        "cost": cost_result,
        "gpu_time_estimate": gpu_time_estimate,
        "hardware_result": hardware_result,
        "report": report,
        "failures": failures,
        "complete_analysis": complete_analysis,
    }

@router.post("/project/analyze", response_model=Dict[str, Any])
async def analyze_project(request: Dict[str, Any]):
    """Analyze an entire fine-tuning project.

    Request body:  {"projectPath": "/path/to/project"}

    Optional flag: {"cleanDatasets": true} additionally runs the chunked
    dataset cleaning pipeline (one LLM request per chunk, all dataset files)
    and returns its summary under the ``cleaning`` key.

    The run is persisted as an analysis record; the response carries
    ``analysisId``/``analysisStatus`` and a ``persistence`` object describing
    whether history storage succeeded.  Previous runs are never overwritten —
    see GET /api/v1/analyses.
    """
    try:
        # The project must live inside an authorized root (the folders open in
        # VS Code right now) and must actually exist. Without the first check
        # any directory on disk could be analyzed, and Project A would remain
        # reachable after the user switched to Project B.
        project_path = str(
            resolve_workspace_root(
                bounded_text(request.get("projectPath"), name="projectPath", maximum=4096)
            )
        )

        return await _run_and_persist_analysis(
            project_path,
            clean_datasets=_as_bool(request.get("cleanDatasets"), False),
        )
    except HTTPException:
        raise
    except StorageError as e:
        raise _storage_http(e)
    except AppError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "errorCode": "ANALYSIS_FAILED",
                "message": "Project analysis failed. See server logs for details.",
                "error": str(e),
            },
        )


@router.get("/project/context")
async def get_project_context(projectPath: Optional[str] = None):
    """Return the latest stored project context (and report) for one project.

    Requires ``projectPath`` so a project can only ever read its own stored
    results — a project never receives another project's context.  Returns
    404 when nothing is stored yet and 503 when the database cannot be read;
    fabricated stand-in data is never returned.
    """
    project_path = _require_project_path(projectPath)
    try:
        run = await get_latest_run(project_path, require_field="context")
    except StorageError as e:
        raise _storage_http(e)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "errorCode": "ANALYSIS_NOT_FOUND",
                "message": (
                    "No stored analysis for this project yet. "
                    "Run POST /api/v1/project/analyze first."
                ),
            },
        )
    return {
        "project": run["context"],
        "report": run["report"],
        "analysisId": run["analysisId"],
        "status": run["status"],
        "startedAt": run["startedAt"],
        "completedAt": run["completedAt"],
    }


@router.post("/project/refresh")
async def refresh_project(request: Dict[str, Any]):
    """Re-scan the project, rebuild its context and refresh stored metadata."""
    try:
        project_path = _require_project_path(request.get("projectPath"))
        if not Path(project_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Project path does not exist: {project_path}",
            )
        scanner = ProjectScanner(project_path)
        scan_result = scanner.scan()
        context = ContextBuilder().build(scan_result)
        project_info = await upsert_project_metadata(
            project_path=project_path,
            project_name=context.project_name,
            framework=context.detected_framework,
            framework_version=context.framework_version,
            base_model=context.base_model,
            tokenizer=context.tokenizer,
        )
        return {"project": context.model_dump(), **project_info}
    except HTTPException:
        raise
    except StorageError as e:
        raise _storage_http(e)
    except AppError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Project refresh failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "errorCode": "ANALYSIS_FAILED",
                "message": "Project refresh failed. See server logs for details.",
                "error": str(e),
            },
        )


# ---------------------------------------------------------------------------
# Analysis history (stored runs — always project-scoped)
# ---------------------------------------------------------------------------

@router.get("/analyses")
async def list_analysis_runs(projectPath: Optional[str] = None, limit: int = 20):
    """List stored analysis runs for one project (newest first).

    Each run has its own ``analysisId``; previous runs are never overwritten.
    """
    project_path = _require_project_path(projectPath)
    try:
        return {"analyses": await list_runs(project_path, limit=limit)}
    except StorageError as e:
        raise _storage_http(e)


@router.get("/analyses/{analysis_id}")
async def get_analysis_run(analysis_id: str):
    """Return one stored analysis run (results, failures, config, report)."""
    try:
        run = await get_run(analysis_id)
    except StorageError as e:
        raise _storage_http(e)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "errorCode": "ANALYSIS_NOT_FOUND",
                "message": f"Analysis run '{analysis_id}' not found.",
            },
        )
    return run


# ---------------------------------------------------------------------------
# Individual analyzer endpoints
# ---------------------------------------------------------------------------

@router.post("/dataset/analyze")
async def analyze_dataset(request: Dict[str, Any]):
    """Analyze a single dataset file.

    The dataset must live inside an authorized workspace root, exactly like
    every other project-scoped route: without this check the endpoint would
    read and describe any file the backend process can open.
    """
    try:
        dataset_path_value = request.get("datasetPath")
        if not dataset_path_value:
            raise HTTPException(
                status_code=400,
                detail={"errorCode": "INVALID_REQUEST", "message": "datasetPath is required."},
            )

        from pathlib import Path

        raw = Path(str(dataset_path_value))
        # An explicit projectPath scopes the dataset; otherwise the dataset's
        # own parent directory is treated as the project root. Either way the
        # result must resolve inside an authorized root.
        project_path_value = request.get("projectPath")
        try:
            if project_path_value:
                project_path = str(resolve_workspace_root(str(project_path_value)))
                dataset_path = resolve_dataset_path(project_path, str(dataset_path_value))
            else:
                if not raw.is_absolute():
                    project_path = str(resolve_workspace_root(str(Path.cwd())))
                else:
                    project_path = str(resolve_workspace_root(str(raw.parent)))
                dataset_path = resolve_dataset_path(project_path, str(raw))
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail={"errorCode": "PATH_TRAVERSAL", "message": str(exc)},
            ) from exc

        if not dataset_path.is_file():
            raise HTTPException(
                status_code=404,
                detail={
                    "errorCode": "DATASET_NOT_FOUND",
                    "message": "The requested dataset file was not found.",
                },
            )

        analyzer = DatasetAnalyzer()
        analysis_context = ProjectContext(
            project_name=dataset_path.stem,
            project_path=project_path,
            dataset_paths=[str(dataset_path)],
        )
        result = await analyzer.analyze(analysis_context, dataset_path=str(dataset_path))
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset analysis failed: {scrub_error(e)}")
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "ANALYSIS_FAILED", "message": scrub_error(e)},
        )


@router.post("/dataset/clean", response_model=DatasetCleaningResult)
async def clean_datasets(request: Dict[str, Any]):
    """Clean every dataset file of a project in LLM-sized chunks.

    Request body (all fields optional except a project or dataset reference):

    ```json
    {
      "projectPath": "/path/to/project",
      "datasetPaths": ["data/train.jsonl", "data/val.jsonl"],
      "chunkSize": 25,
      "maxChunkChars": 12000,
      "outputDirectory": ".llm-training-agent/cleaned",
      "maxFiles": 0,
      "useLlm": true
    }
    ```

    * Every dataset file is discovered (a directory entry is expanded
      recursively), then split into sequential chunks of ``chunkSize`` records
      (and at most ``maxChunkChars`` characters).
    * Each chunk is sent to the AI provider in its own request, so an entire
      dataset is never sent in a single call.
    * Cleaned files are written per source file, preserving the relative path
      and format, together with a manifest for verification.  Every record of
      every file is preserved; a chunk that fails validation keeps its original
      records instead of losing data.
    """
    try:
        from pathlib import Path

        project_path_value = request.get("projectPath")
        dataset_paths = list(request.get("datasetPaths") or [])
        if len(dataset_paths) > MAX_DATASET_ENTRIES:
            raise HTTPException(status_code=422, detail={"errorCode": "INVALID_REQUEST", "message": "Too many dataset paths."})
        if project_path_value:
            project_path = str(resolve_workspace_root(str(project_path_value)))
        else:
            if not dataset_paths:
                raise HTTPException(status_code=400, detail={"errorCode": "INVALID_REQUEST", "message": "projectPath or datasetPaths is required."})
            first = Path(str(dataset_paths[0])).expanduser()
            project_path = str(resolve_workspace_root(str(first.parent if first.is_absolute() else Path.cwd())))
        if not Path(project_path).is_dir():
            raise HTTPException(status_code=404, detail={"errorCode": "PROJECT_NOT_FOUND", "message": "Project directory was not found."})

        if not dataset_paths:
            # Discover dataset files with the same rules the scanner uses.
            scanner = ProjectScanner(project_path)
            dataset_paths = scanner.discover_datasets()
            if not dataset_paths:
                raise HTTPException(
                    status_code=404,
                    detail="No dataset files were found in this project.",
                )

        chunk_size = bounded_int(request.get("chunkSize"), name="chunkSize", minimum=1, maximum=MAX_CHUNK_RECORDS, default=25)
        max_chunk_chars = bounded_int(request.get("maxChunkChars"), name="maxChunkChars", minimum=200, maximum=MAX_CHUNK_CHARS, default=12000)
        max_files = bounded_int(request.get("maxFiles"), name="maxFiles", minimum=0, maximum=MAX_DATASET_FILES, default=0)

        cleaner = DatasetCleaner(
            chunk_size=chunk_size or 25,
            max_chunk_chars=max_chunk_chars or 12000,
        )

        context = ProjectContext(
            project_name=Path(str(project_path)).name or "dataset_cleanup",
            project_path=str(project_path),
            dataset_paths=dataset_paths,
        )

        logger.info(f"Cleaning {len(dataset_paths)} dataset entry(ies) of {project_path}")
        result = await cleaner.clean_project(
            context,
            output_dir=request.get("outputDirectory"),
            max_files=max_files,
            use_llm=_as_bool(request.get("useLlm"), True),
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Dataset cleaning rejected an invalid path or parameter")
        raise HTTPException(status_code=400, detail={"errorCode": "INVALID_REQUEST", "message": str(e)}) from e
    except Exception as e:
        logger.error(f"Dataset cleaning failed: {scrub_error(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "CLEANING_FAILED", "message": "Dataset cleaning failed. See server logs for details."},
        )


@router.post("/prompt/analyze")
async def analyze_prompt(request: Dict[str, Any]):
    """Analyze a single prompt template."""
    try:
        template = request.get("promptTemplate")
        template_path = request.get("templatePath")
        if not template and not template_path:
            raise HTTPException(status_code=400, detail="promptTemplate or templatePath is required")

        analyzer = PromptAnalyzer()
        context = ProjectContext(project_name="temp", project_path="")
        if template_path:
            context.prompt_templates = [template_path]
        if template:
            context.prompt_templates = [template]
        result = await analyzer.analyze(context)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prompt analysis failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "ANALYSIS_FAILED", "message": str(e)})


@router.post("/hyperparameters/analyze")
async def analyze_hyperparameters(request: Dict[str, Any]):
    """Analyze training hyperparameters from a config file or inline values."""
    try:
        config_path = request.get("configPath")
        analyzer = HyperparameterAnalyzer()
        context = ProjectContext(project_name="temp", project_path="")
        result = await analyzer.analyze(context, config_path=config_path, **{k: v for k, v in request.items() if k != "configPath"})
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hyperparameter analysis failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "ANALYSIS_FAILED", "message": str(e)})


@router.post("/model/analyze")
async def analyze_model(request: Dict[str, Any]):
    """Analyze model suitability."""
    try:
        model_name = request.get("modelName") or request.get("model")
        advisor = ModelAdvisor()
        context = ProjectContext(project_name="temp", project_path="")
        if model_name:
            context.base_model = model_name
        result = await advisor.analyze(context)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model analysis failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "ANALYSIS_FAILED", "message": str(e)})


@router.post("/predict/training")
async def predict_training(request: Dict[str, Any]):
    """Predict training outcomes."""
    try:
        context = ProjectContext(project_name="temp", project_path="")
        dataset_result = DatasetAnalysisResult(dataset_name="unknown", sample_count=0, token_count=0,
            average_prompt_length=0, average_response_length=0, duplicate_percentage=0,
            near_duplicate_percentage=0, missing_field_percentage=0,
            formatting_consistency_score=0.9, language_consistency_score=0.9,
            instruction_consistency_score=0.9, response_consistency_score=0.9,
            quality_score=0.85, confidence="medium")
        hp_result = HyperparameterAnalysisResult(efficiency_score=0.75, confidence="medium")
        model_result = ModelAnalysisResult(selected_model="unknown", parameter_count="unknown",
            context_length=4096, estimated_vram="unknown", confidence="medium")
        engine = PredictionEngine()
        result = await engine.predict(context, dataset_result, hp_result, model_result)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "PREDICTION_FAILED", "message": str(e)})


@router.post("/cost/estimate")
async def estimate_cost(request: Dict[str, Any]):
    """Estimate training cost."""
    try:
        estimator = CostEstimator()
        context = ProjectContext(project_name="temp", project_path="")
        result = await estimator.analyze(context)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "ESTIMATION_FAILED", "message": str(e)})


# ---------------------------------------------------------------------------
# GPU detection & time estimation
# ---------------------------------------------------------------------------

@router.get("/gpu/detect")
async def detect_gpu_hardware():
    """Detect GPU hardware on the current machine."""
    try:
        hardware = detect_gpus()
        return hardware.to_dict()
    except Exception as e:
        logger.error(f"GPU detection failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "GPU_DETECTION_FAILED", "message": str(e)})


@router.get("/gpu/specs")
async def list_gpu_specs():
    """List known GPU specifications in the performance database."""
    return {"gpus": GPUSpec.get_known_keys()}


@router.post("/gpu/estimate")
async def estimate_gpu_time(request: Dict[str, Any]):
    """Estimate GPU training time.

    Request body:
    {
        "projectPath": "/path/to/project",  # optional - uses existing context
        "mode": "quick" | "calibrated",     # default: quick
        "measuredStepsPerSec": 2.84,        # optional - for calibrated mode
        "config": {                          # optional - override training config
            "modelName": "TinyLlama 1.1B",
            "paramCountBillions": 1.1,
            "datasetSamples": 50000,
            "sequenceLength": 512,
            "batchSize": 4,
            "gradientAccumulation": 8,
            "epochs": 3,
            "maxSteps": null,
            "precision": "4bit",
            "trainingMethod": "qlora",
            "gradientCheckpointing": false,
            "dataloaderWorkers": 0,
            "framework": "huggingface",
            "distributedStrategy": "single"
        }
    }
    """
    try:
        # Build training config
        config_data = request.get("config") or {}
        training_config = TrainingConfig(
            model_name=config_data.get("modelName"),
            param_count_billions=config_data.get("paramCountBillions"),
            dataset_samples=config_data.get("datasetSamples"),
            sequence_length=config_data.get("sequenceLength", 512),
            batch_size=config_data.get("batchSize", 8),
            gradient_accumulation=config_data.get("gradientAccumulation", 1),
            epochs=config_data.get("epochs", 3),
            max_steps=config_data.get("maxSteps"),
            precision=config_data.get("precision", "fp16"),
            training_method=config_data.get("trainingMethod", "full"),
            gradient_checkpointing=config_data.get("gradientCheckpointing", False),
            dataloader_workers=config_data.get("dataloaderWorkers", 0),
            framework=config_data.get("framework", "huggingface"),
            distributed_strategy=config_data.get("distributedStrategy", "single"),
        )

        # If projectPath provided, try to extract config from project
        project_path = request.get("projectPath")
        if project_path and not config_data:
            try:
                scanner = ProjectScanner(project_path)
                scan_result = scanner.scan()
                context_builder = ContextBuilder()
                context = context_builder.build(scan_result)
                training_config = extract_training_config_from_context(context)
            except Exception as e:
                logger.warning(f"Failed to extract config from project: {e}")

        mode = request.get("mode", "quick")
        measured_steps = request.get("measuredStepsPerSec")

        estimator = GPUTimeEstimator()
        result = estimator.estimate(training_config, mode=mode, measured_steps_per_sec=measured_steps)
        return result.to_dict()
    except Exception as e:
        logger.error(f"GPU time estimation failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "GPU_ESTIMATION_FAILED", "message": str(e)})


@router.post("/gpu/calibrate")
async def calibrate_gpu(request: Dict[str, Any]):
    """Run a short calibration benchmark to measure actual GPU throughput.

    Request body:
    {
        "durationSeconds": 30,  # optional - default 30, max 60
        "config": {              # optional - training config for benchmark
            "batchSize": 4,
            "sequenceLength": 512,
            "paramCountBillions": 1.1
        }
    }
    """
    try:
        duration = request.get("durationSeconds", 30)
        config_data = request.get("config") or {}
        training_config = TrainingConfig(
            batch_size=config_data.get("batchSize", 4),
            sequence_length=config_data.get("sequenceLength", 512),
            param_count_billions=config_data.get("paramCountBillions"),
        )
        result = run_calibration_benchmark(training_config, duration_seconds=duration)
        return result
    except Exception as e:
        logger.error(f"Calibration benchmark failed: {e}")
        raise HTTPException(status_code=500, detail={"errorCode": "CALIBRATION_FAILED", "message": str(e)})


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def _history_messages(project_path: Optional[str], session_id: str) -> List[ChatMessage]:
    """Load the bounded conversation history for a project session."""
    return [
        ChatMessage(role=item.get("role", "user"), content=item.get("content", ""))
        for item in store.get_history(project_path, session_id)
        if item.get("content")
    ]


def _snapshot_models(snapshot, project_path):
    """Reconstruct typed models from the stored analysis snapshot.

    Falls back to a minimal ProjectContext (and no report/recommendations)
    when no analysis has been run yet, so chat still works with less context.
    """
    context = None
    report = None
    recommendations: List[Recommendation] = []
    if snapshot:
        raw_context = snapshot.get("context")
        if isinstance(raw_context, dict):
            try:
                context = ProjectContext(**raw_context)
            except Exception:
                context = None
        raw_report = snapshot.get("report")
        if isinstance(raw_report, dict):
            try:
                report = EngineeringReport(**raw_report)
            except Exception:
                report = None
        for raw_rec in snapshot.get("recommendations") or []:
            if isinstance(raw_rec, dict):
                try:
                    recommendations.append(Recommendation(**raw_rec))
                except Exception:
                    continue
    if context is None:
        name = Path(project_path).name if project_path else "current_project"
        context = ProjectContext(project_name=name, project_path=project_path or "")
    return context, report, recommendations


@router.post("/chat/message")
async def chat_message(request: Dict[str, Any]):
    """Send a message to the AI chat.

    Request body:
        {
            "message": "Why is my batch size bad?",  # required
            "sessionId": "default",                  # optional conversation id
            "projectPath": "/path/to/project",       # optional - scopes history + context
            "provider": "openai"                     # optional provider override
        }

    The message reaches the configured LLM provider through ChatEngine with
    the project's analysis snapshot and conversation history. Anticipated
    failures return HTTP 200 with a friendly assistant message plus an
    ``error`` code (chat-UX convention, matching PROVIDER_NOT_CONFIGURED);
    unexpected failures return 500 with a scrubbed message. Only completed
    turns (user + assistant) are persisted.
    """
    try:
        message = request.get("message")
        if not message or not str(message).strip():
            raise HTTPException(status_code=422, detail="message must not be empty")
        message = str(message).strip()

        provider_name = request.get("provider")
        project_path = str(request.get("projectPath") or "").strip()
        session_id = str(request.get("sessionId") or "default").strip() or "default"

        history = _history_messages(project_path, session_id)
        snapshot = store.get_analysis(project_path)
        context, report, recommendations = _snapshot_models(snapshot, project_path)

        engine = ChatEngine(provider_name=provider_name)
        try:
            response = await engine.ask(
                context, report, recommendations, message,
                chat_history=history, snapshot=snapshot,
            )
        except ValueError as e:
            # Provider could not be constructed (missing/empty API key, ...).
            return {
                "assistantResponse": (
                    "I cannot answer because no AI provider is configured. "
                    f"Reason: {scrub(str(e))}. "
                    "Go to Settings (Command Palette: 'AI Provider: Configure') to set up a provider."
                ),
                "references": [],
                "confidence": "low",
                "error": "PROVIDER_NOT_CONFIGURED",
                "sessionId": session_id,
            }
        except AppError as e:
            # Classified provider failure: friendly message, scrubbed details.
            logger.warning(f"Chat provider error ({e.error_code}): {e.details}")
            return {
                "assistantResponse": e.message,
                "references": [],
                "confidence": "low",
                "error": e.error_code,
                "sessionId": session_id,
            }

        # Persist the completed turn for future context (bounded by ChatStore).
        store.append_message(project_path, session_id, "user", message)
        store.append_message(project_path, session_id, "assistant", response["assistantResponse"])

        return {
            "assistantResponse": response["assistantResponse"],
            "references": response["references"],
            "confidence": response["confidence"],
            "model": response.get("model", provider_name or "unknown"),
            "sessionId": session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "CHAT_FAILED", "message": scrub_error(e)},
        )


@router.get("/chat/history")
async def get_chat_history(
    projectPath: Optional[str] = None,
    sessionId: Optional[str] = None,
):
    """Return the conversation history for a project session (bounded)."""
    return {"messages": store.get_history(projectPath, sessionId)}


@router.delete("/chat/history")
async def delete_chat_history(
    projectPath: Optional[str] = None,
    sessionId: Optional[str] = None,
):
    """Clear one project session's conversation history."""
    removed = store.clear_history(projectPath, sessionId)
    return {"status": "cleared", "removed": removed}


# ---------------------------------------------------------------------------
# Recommendations, Experiments, Files
# ---------------------------------------------------------------------------

@router.get("/recommendations")
async def get_recommendations(projectPath: Optional[str] = None, limit: int = 50):
    """Return the stored recommendations of a project's latest analysis.

    Requires ``projectPath``: recommendations always belong to the run that
    produced them, and ``analysisId``/``startedAt`` are included so clients
    can judge freshness after configuration changes.
    """
    project_path = _require_project_path(projectPath)
    try:
        run = await get_latest_run(project_path, require_field="report")
    except StorageError as e:
        raise _storage_http(e)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "errorCode": "ANALYSIS_NOT_FOUND",
                "message": (
                    "No stored analysis for this project yet. "
                    "Run POST /api/v1/project/analyze first."
                ),
            },
        )
    bounded = max(1, min(int(limit or 50), 200))
    return {
        "recommendations": run["recommendations"][:bounded],
        "analysisId": run["analysisId"],
        "status": run["status"],
        "startedAt": run["startedAt"],
    }


@router.post("/recommendations/refresh")
async def refresh_recommendations(request: Dict[str, Any]):
    """Regenerate recommendations by re-running the analysis pipeline.

    Returns only the recommendations of the *new* run (identified by
    ``analysisId``), so clients can never mistake a previous run's suggestions
    for fresh ones.
    """
    try:
        project_path = _require_project_path(request.get("projectPath"))
        if not Path(project_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Project path does not exist: {project_path}",
            )
        response = await _run_and_persist_analysis(project_path, clean_datasets=False)
        report = response.get("report") or {}
        return {
            "recommendations": report.get("prioritized_recommendations", []),
            "analysisId": response.get("analysisId"),
            "analysisStatus": response.get("analysisStatus"),
            "partialFailures": response.get("partialFailures", []),
            "persistence": response.get("persistence"),
        }
    except HTTPException:
        raise
    except StorageError as e:
        raise _storage_http(e)
    except AppError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Recommendation refresh failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "errorCode": "ANALYSIS_FAILED",
                "message": "Recommendation refresh failed. See server logs for details.",
                "error": str(e),
            },
        )


@router.get("/report")
async def get_report(projectPath: Optional[str] = None):
    """Return the stored engineering report of a project's latest analysis."""
    project_path = _require_project_path(projectPath)
    try:
        run = await get_latest_run(project_path, require_field="report")
    except StorageError as e:
        raise _storage_http(e)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "errorCode": "ANALYSIS_NOT_FOUND",
                "message": "No report available. Run POST /api/v1/project/analyze first.",
            },
        )
    return {
        "report": run["report"],
        "analysisId": run["analysisId"],
        "status": run["status"],
        "startedAt": run["startedAt"],
    }


class ExperimentCreateRequest(BaseModel):
    """Request body for POST /experiments.

    ``estimates`` holds the agent's *pre-training predictions*; ``metrics``
    and ``actualTrainingTimeSeconds`` hold *actual* observed/reported values
    (empty when unknown — never fabricated).
    """

    projectPath: str
    model: str
    tokenizer: Optional[str] = None
    fineTuningMethod: Optional[str] = None
    datasetVersion: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    estimates: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    actualTrainingTimeSeconds: Optional[float] = None
    status: str = "estimated"
    error: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    artifacts: List[Any] = Field(default_factory=list)
    analysisId: Optional[str] = None


class ExperimentCompareRequest(BaseModel):
    """Request body for POST /experiments/compare."""

    experimentIds: List[str]
    projectPath: Optional[str] = None


@router.get("/experiments")
async def list_experiments(projectPath: Optional[str] = None, limit: int = 50):
    """List stored experiments for one project (newest first, project-isolated)."""
    if not projectPath:
        raise HTTPException(status_code=400, detail={"errorCode": "VALIDATION_ERROR", "message": "projectPath is required."})
    project_path = _require_project_path(projectPath)
    try:
        return {
            "experiments": await _experiment_service.list_experiments(
                project_path, limit=limit
            )
        }
    except StorageError as e:
        raise _storage_http(e)


@router.post("/experiments", status_code=status.HTTP_201_CREATED)
async def create_experiment(request: ExperimentCreateRequest):
    """Create an experiment record (estimated vs actual kept strictly apart)."""
    try:
        payload = request.model_dump()
        payload["projectPath"] = _require_project_path(payload.get("projectPath"))
        return await _experiment_service.save_experiment(payload)
    except HTTPException:
        raise
    except StorageError as e:
        raise _storage_http(e)
    except ValidationError as e:
        raise _app_http(e)


@router.post("/experiments/compare")
async def compare_experiments(request: ExperimentCompareRequest):
    """Return the requested experiments (in the requested order).

    Only stored fields are returned — no metric is inferred or synthesized —
    so a client can line experiments up side by side.  When ``projectPath``
    is provided, every record must belong to that project.
    """
    ids = [str(value).strip() for value in (request.experimentIds or []) if str(value).strip()]
    try:
        # Validate projectPath against the authorized roots when supplied, so
        # this route enforces the same boundary as every other project-scoped one.
        project_path = _require_project_path(request.projectPath) if request.projectPath else None
        experiments = await _experiment_service.compare_experiments(
            ids, project_path
        )
    except StorageError as e:
        raise _storage_http(e)
    except ValidationError as e:
        raise _app_http(e)
    return {"experiments": experiments}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Return one stored experiment by id."""
    try:
        record = await _experiment_service.get_experiment(experiment_id)
    except StorageError as e:
        raise _storage_http(e)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "errorCode": "EXPERIMENT_NOT_FOUND",
                "message": f"Experiment '{experiment_id}' not found.",
            },
        )
    return record


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete one experiment (explicit user action — history is never auto-pruned)."""
    try:
        removed = await _experiment_service.delete_experiment(experiment_id)
    except StorageError as e:
        raise _storage_http(e)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail={
                "errorCode": "EXPERIMENT_NOT_FOUND",
                "message": f"Experiment '{experiment_id}' not found.",
            },
        )
    return {"experimentId": experiment_id, "status": "deleted"}


@router.get("/logs")
async def get_logs():
    """Return recent log entries (dev only)."""
    return {"logs": []}


@router.get("/metrics")
async def get_metrics():
    """Return performance metrics."""
    return {"uptime_seconds": 0, "requests_total": 0}


# ---------------------------------------------------------------------------
# File changes: propose → review → approve → apply → verify → rollback
#
# This is the ONLY route family able to modify a project file.  It is a thin,
# validating adapter over ``editing.file_editor.ChangeStore``; all safety
# rules (workspace containment, lifecycle, conflict detection, atomic writes,
# verification) live in that module so there is a single authoritative
# mechanism rather than competing implementations.
# ---------------------------------------------------------------------------

from editing.file_editor import (  # noqa: E402
    ALLOWED_OPS,
    ChangeStore,
    ChangeStoreError,
    OP_DELETE,
    OP_MODIFY,
)


def _change_store(project_root: Optional[str]) -> ChangeStore:
    """Build a store bound to an authorized, existing project root.

    The workspace identity is always taken from the request (never from a
    module-level global), so two open workspaces cannot share a store.
    """
    if not project_root or not str(project_root).strip():
        raise HTTPException(
            status_code=422,
            detail={"errorCode": "WORKSPACE_REQUIRED", "message": "A projectRoot is required."},
        )
    root = resolve_workspace_root(str(project_root))
    return ChangeStore(root)


def _change_http(exc: ChangeStoreError) -> HTTPException:
    """Map a domain error onto a scrubbed HTTP error response."""
    return HTTPException(
        status_code=exc.status_code,
        detail={"errorCode": exc.code, "message": exc.message, **({"details": exc.details} if exc.details else {})},
    )


def _public(record: Dict[str, Any], include_contents: bool = True) -> Dict[str, Any]:
    """Shape a stored record for the API, optionally dropping file contents."""
    if include_contents:
        return dict(record)
    return {k: v for k, v in record.items() if k not in ("originalContent", "proposedContent")}


@router.post("/files/propose")
async def propose_file_change(request: Dict[str, Any]):
    """Propose a change for a project file **without modifying it**.

    Request body:
        {
            "filePath": "src/app.py",      # required, workspace-relative
            "proposedContent": "...",      # required except for "delete"
            "operation": "modify",         # optional: create | modify | delete
            "projectRoot": "/path/to/proj",# required, authorized workspace
            "reason": "why"                # optional, shown in the UI
        }

    Returns the proposal including a unified diff, additions/deletions and the
    original content, so the user can review the change before approving it.
    """
    try:
        file_path = request.get("filePath")
        if not isinstance(file_path, str) or not file_path.strip() or len(file_path) > 4096 or "\x00" in file_path:
            raise HTTPException(
                status_code=422,
                detail={"errorCode": "INVALID_PATH", "message": "filePath is invalid."},
            )
        operation = str(request.get("operation") or OP_MODIFY).strip().lower()
        if operation not in ALLOWED_OPS:
            raise HTTPException(
                status_code=422,
                detail={
                    "errorCode": "INVALID_OPERATION",
                    "message": f"Unsupported operation. Allowed: {', '.join(sorted(ALLOWED_OPS))}.",
                },
            )

        proposed_content = request.get("proposedContent")
        if operation != OP_DELETE and not isinstance(proposed_content, str):
            raise HTTPException(
                status_code=422,
                detail={"errorCode": "INVALID_REQUEST", "message": "proposedContent must be a string."},
            )
        if operation == OP_DELETE:
            proposed_content = None

        reason = request.get("reason")
        reason = reason.strip() if isinstance(reason, str) and reason.strip() else None

        store = _change_store(request.get("projectRoot"))
        return store.propose(
            file_path=file_path.strip(),
            proposed_content=proposed_content,
            operation=operation,
            reason=reason,
        )
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Propose file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "PROPOSE_FAILED", "message": "The change could not be proposed."},
        )


@router.get("/files/changes")
async def list_file_changes(projectRoot: Optional[str] = None, includeTerminal: bool = True):
    """List change proposals for one workspace (metadata only, no contents)."""
    try:
        store = _change_store(projectRoot)
        return {"changes": store.list(include_terminal=includeTerminal)}
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("List file changes failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "LIST_CHANGES_FAILED", "message": "The changes could not be listed."},
        )


@router.get("/files/changes/{change_id}")
async def get_file_change(change_id: str, projectRoot: Optional[str] = None, includeContents: bool = True):
    """Return one proposal, optionally including the original/proposed content.

    Also reports whether the target still matches the proposal (``conflicted``)
    so the UI can warn the user before they approve a stale change.
    """
    try:
        store = _change_store(projectRoot)
        record = store.require(change_id)
        payload = _public(record, include_contents=includeContents)
        payload["conflicted"] = store.check_conflict(record) is not None
        return payload
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Get file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "GET_CHANGE_FAILED", "message": "The change could not be loaded."},
        )


@router.post("/files/changes/{change_id}/approve")
async def approve_file_change(change_id: str, request: Dict[str, Any]):
    """Explicitly approve a proposal. Required before it can be applied."""
    try:
        store = _change_store(request.get("projectRoot"))
        return _public(store.approve(change_id), include_contents=False)
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Approve file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "APPROVE_FAILED", "message": "The change could not be approved."},
        )


@router.post("/files/changes/{change_id}/reject")
async def reject_file_change(change_id: str, request: Dict[str, Any]):
    """Reject a proposal. The target file is never modified."""
    try:
        store = _change_store(request.get("projectRoot"))
        return _public(store.reject(change_id), include_contents=False)
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Reject file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "REJECT_FAILED", "message": "The change could not be rejected."},
        )


@router.post("/files/changes/{change_id}/cancel")
async def cancel_file_change(change_id: str, request: Dict[str, Any]):
    """Cancel a proposal (e.g. one that has been superseded)."""
    try:
        store = _change_store(request.get("projectRoot"))
        return _public(store.cancel(change_id), include_contents=False)
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Cancel file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "CANCEL_FAILED", "message": "The change could not be cancelled."},
        )


@router.post("/files/changes/{change_id}/apply")
async def apply_file_change(change_id: str, request: Dict[str, Any]):
    """Apply an **approved** change, then verify the result.

    Returns 409 when the change was never approved, was already applied, or
    when the target file changed since the proposal was generated.  A conflict
    never overwrites the user's newer content.
    """
    try:
        store = _change_store(request.get("projectRoot"))
        record = store.apply(change_id)
        verification = store.verify(record)
        payload = _public(record, include_contents=False)
        payload["verification"] = verification
        payload["verified"] = verification["verified"]
        return payload
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Apply file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "APPLY_FAILED", "message": "The change could not be applied."},
        )


@router.get("/files/changes/{change_id}/verify")
async def verify_file_change(change_id: str, projectRoot: Optional[str] = None):
    """Report whether a proposal's target file still matches its expected state."""
    try:
        store = _change_store(projectRoot)
        record = store.require(change_id)
        conflict = store.check_conflict(record)
        return {
            "changeId": change_id,
            "status": record.get("status"),
            "conflicted": conflict is not None,
            "conflict": conflict,
        }
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Verify file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "VERIFY_FAILED", "message": "The change could not be verified."},
        )


@router.post("/files/changes/{change_id}/rollback")
async def rollback_file_change(change_id: str, request: Dict[str, Any]):
    """Restore the pre-apply content from the stored backup."""
    try:
        store = _change_store(request.get("projectRoot"))
        record = store.rollback(change_id)
        return _public(record, include_contents=False)
    except ChangeStoreError as e:
        raise _change_http(e)
    except HTTPException:
        raise
    except Exception:
        logger.error("Rollback file change failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"errorCode": "ROLLBACK_FAILED", "message": "The change could not be rolled back."},
        )
