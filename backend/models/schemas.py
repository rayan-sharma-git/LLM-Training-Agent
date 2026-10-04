"""Pydantic schemas for all data models."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProjectContext(BaseModel):
    """Structured project representation."""
    project_name: str
    project_path: str
    detected_framework: Optional[str] = None
    framework_version: Optional[str] = None
    base_model: Optional[str] = None
    tokenizer: Optional[str] = None
    dataset_paths: List[str] = []
    prompt_templates: List[str] = []
    configuration_files: List[str] = []
    training_scripts: List[str] = []
    evaluation_scripts: List[str] = []
    inference_scripts: List[str] = []
    hardware_information: Dict[str, Any] = {}
    project_statistics: Dict[str, Any] = {}
    project_health_score: Optional[float] = None
    training_readiness_score: Optional[float] = None
    analysis_timestamp: Optional[datetime] = None


class DatasetAnalysisResult(BaseModel):
    """Dataset analysis results."""
    dataset_name: str
    sample_count: int
    token_count: int
    average_prompt_length: float
    average_response_length: float
    duplicate_percentage: float
    near_duplicate_percentage: float
    missing_field_percentage: float
    formatting_consistency_score: float
    language_consistency_score: float
    instruction_consistency_score: float
    response_consistency_score: float
    quality_score: float
    findings: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []
    confidence: str = "medium"


class ChunkCleaningSummary(BaseModel):
    """Outcome of cleaning one chunk of records with a single LLM request."""

    chunk_index: int
    record_count: int
    records_cleaned: int = 0
    llm_used: bool = False
    status: str = "cleaned"  # cleaned | fallback | failed
    warnings: List[str] = []


class FileCleaningSummary(BaseModel):
    """Outcome of cleaning a single dataset file (all of its chunks)."""

    source_file: str
    relative_path: str
    output_path: Optional[str] = None
    format: str = "jsonl"
    records_in: int = 0
    records_out: int = 0
    chunks_total: int = 0
    chunks_cleaned: int = 0
    chunks_fallback: int = 0
    chunks_deterministic: int = 0
    chunk_summaries: List[ChunkCleaningSummary] = []
    llm_used: bool = False
    records_preserved: bool = True
    warnings: List[str] = []
    errors: List[str] = []


class DatasetCleaningResult(BaseModel):
    """Aggregated result of cleaning every discovered dataset file."""

    files: List[FileCleaningSummary] = []
    skipped_files: List[str] = []
    output_directory: Optional[str] = None
    manifest_path: Optional[str] = None
    total_files: int = 0
    total_records_in: int = 0
    total_records_out: int = 0
    total_chunks: int = 0
    total_chunks_cleaned: int = 0
    total_chunks_fallback: int = 0
    total_chunks_deterministic: int = 0
    records_preserved: bool = True
    cross_file_contamination: bool = False
    llm_used: bool = False
    chunk_size: int = 0
    max_chunk_chars: int = 0
    warnings: List[str] = []
    confidence: str = "medium"


class PromptAnalysisResult(BaseModel):
    """Prompt analysis results."""
    template_name: str
    prompt_complexity: str = "moderate"
    ambiguity_score: float
    clarity_score: float
    formatting_score: float
    instruction_quality_score: float
    consistency_score: float
    detected_issues: List[str] = []
    recommendations: List[str] = []
    confidence: str = "medium"


class HyperparameterAnalysisResult(BaseModel):
    """Hyperparameter analysis results."""
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    scheduler: Optional[str] = None
    gradient_accumulation: Optional[int] = None
    weight_decay: Optional[float] = None
    warmup_ratio: Optional[float] = None
    sequence_length: Optional[int] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    overfitting_risk: str = "medium"
    underfitting_risk: str = "medium"
    efficiency_score: float
    findings: List[str] = []
    recommendations: List[str] = []
    confidence: str = "medium"


class ModelAnalysisResult(BaseModel):
    """Model analysis results."""
    selected_model: str
    parameter_count: str
    context_length: int
    estimated_vram: str
    reasoning_capability: str = "medium"
    coding_capability: str = "medium"
    multilingual_capability: str = "medium"
    instruction_following_capability: str = "medium"
    speed_score: str = "medium"
    memory_efficiency: str = "medium"
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommended_alternatives: List[str] = []
    confidence: str = "medium"


class PredictionEvidence(BaseModel):
    """A single input the prediction is based on, labelled by provenance.

    ``category`` is one of:
      * ``measured``  - read from the user's actual project files
      * ``calculated``- derived from measured values by a documented formula
      * ``heuristic`` - an engineering rule of thumb applied to measured values
      * ``assumed``   - a value that had to be assumed to produce an estimate
      * ``unknown``   - information that was required but not available
    """
    category: str
    item: str
    value: str
    source: str = ""


class PredictionRisk(BaseModel):
    """A structured, explainable risk indicator.

    Thresholds are engineering heuristics, not scientific laws; ``rationale``
    records why the threshold exists and ``trigger`` records what was observed.
    """
    category: str
    severity: str = "medium"          # critical | high | medium | low | info
    trigger: str = ""
    evidence: str = ""
    rationale: str = ""
    mitigation: str = ""


class PredictionResult(BaseModel):
    """Training outcome predictions.

    Fields describe engineering *estimates and indicators* derived from the
    actual analyzer outputs. They are never guaranteed post-training results:
    ``quality_note``/``evaluation_required`` make the limits explicit, and
    ``evidence`` records what each conclusion rests on.

    Unavailable information is reported with the literal value ``unknown``
    rather than a fabricated default.
    """
    instruction_following_prediction: str = "unknown"
    hallucination_risk: str = "unknown"
    reasoning_prediction: str = "unknown"
    response_consistency_prediction: str = "unknown"
    creativity_prediction: str = "unknown"
    formatting_prediction: str = "unknown"
    likely_failure_modes: List[str] = []
    expected_strengths: List[str] = []
    expected_weaknesses: List[str] = []
    confidence: str = "very_low"

    # --- Status & provenance ------------------------------------------------
    status: str = "unknown"  # ok | partial | insufficient_data | not_estimated
    confidence_basis: str = ""
    uncertainty: str = ""
    unknowns: List[str] = []
    assumptions: List[str] = []
    evidence: List[PredictionEvidence] = []
    risks: List[PredictionRisk] = []
    quality_note: str = ""
    evaluation_required: bool = True

    # --- Training-time / resource estimates (numeric, comparable with runs) --
    estimated_total_tokens: Optional[int] = None
    estimated_training_steps: Optional[int] = None
    estimated_training_seconds: Optional[float] = None
    estimated_training_seconds_lower: Optional[float] = None
    estimated_training_seconds_upper: Optional[float] = None
    estimated_gpu_hours: Optional[float] = None
    estimated_vram_gb: Optional[float] = None
    estimated_throughput_samples_per_sec: Optional[float] = None
    estimate_basis: str = ""


class CostEstimate(BaseModel):
    """Training cost estimates."""
    estimated_training_time: str
    estimated_gpu_hours: float
    estimated_vram_usage: str
    estimated_checkpoint_size: str
    estimated_storage_requirement: str
    compatible_hardware: List[str] = []
    assumptions: List[str] = []
    confidence: str = "medium"


class GPUInfo(BaseModel):
    """Information about a single detected GPU."""
    index: int
    name: str
    vram_mb: Optional[int] = None
    vram_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None


class HardwareInfo(BaseModel):
    """Overall hardware detection result."""
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    gpus: List[GPUInfo] = []
    gpu_count: int = 0
    heterogeneous: bool = False
    total_vram_mb: int = 0
    max_vram_mb: int = 0
    min_vram_mb: int = 0
    detection_method: str = "none"
    error: Optional[str] = None


class GPUTimeEstimate(BaseModel):
    """GPU training time estimation result."""
    mode: str = "quick"  # quick or calibrated
    estimated_seconds: float = 0
    lower_bound_seconds: float = 0
    upper_bound_seconds: float = 0
    estimated_time: str = "unknown"
    range: str = "unknown"
    confidence: str = "low"
    throughput_steps_per_sec: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    total_steps: int = 0
    vram_feasible: bool = True
    vram_warning: Optional[str] = None
    vram_estimated_gb: Optional[float] = None
    vram_available_gb: Optional[float] = None
    assumptions: List[str] = []
    warnings: List[str] = []
    gpu_info: Dict[str, Any] = {}
    calibration_used: bool = False


class CalibrationResult(BaseModel):
    """Calibration benchmark result."""
    success: bool = False
    measured_steps_per_sec: Optional[float] = None
    measured_samples_per_sec: Optional[float] = None
    measured_tokens_per_sec: Optional[float] = None
    benchmark_duration_seconds: Optional[float] = None
    benchmark_batch_size: Optional[int] = None
    benchmark_seq_len: Optional[int] = None
    benchmark_model_params: Optional[float] = None
    gpu_name: Optional[str] = None
    error: Optional[str] = None


class Recommendation(BaseModel):
    """Single recommendation."""
    recommendation_id: str
    category: str
    title: str
    description: str
    reasoning: str
    evidence: str
    severity: str = "medium"
    confidence: str = "medium"
    estimated_benefit: str = ""
    implementation_difficulty: str = "moderate"
    estimated_engineering_time: str = ""
    affected_files: List[str] = []
    suggested_actions: List[str] = []
    references: List[str] = []


class EngineeringReport(BaseModel):
    """Complete engineering report."""
    executive_summary: str
    project_health_score: float
    training_readiness_score: float
    dataset_summary: Dict[str, Any] = {}
    prompt_summary: Dict[str, Any] = {}
    hyperparameter_summary: Dict[str, Any] = {}
    model_summary: Dict[str, Any] = {}
    prediction_summary: Dict[str, Any] = {}
    cost_summary: Dict[str, Any] = {}
    gpu_time_estimate: Optional[GPUTimeEstimate] = None
    hardware_detection: Optional[HardwareInfo] = None
    prioritized_recommendations: List[Recommendation] = []
    action_plan: List[str] = []


class UseCaseSignal(BaseModel):
    """A single piece of evidence supporting (or opposing) a use-case label.

    ``provenance`` records where the evidence came from so the final report can
    distinguish a *measured* project fact from a *heuristic* keyword match.  The
    detector never invents evidence: a signal is only emitted for something
    actually observed in the user's project.
    """

    task: str
    weight: float
    provenance: str = "measured"  # measured | calculated | heuristic
    evidence: str = ""


class UseCaseAnalysisResult(BaseModel):
    """Automatically inferred task/use case for the project.

    Every field is derived from evidence found in the project (dataset field
    names and values, prompt templates, training config, framework, scripts).
    When the evidence is weak or contradictory the confidence is lowered and
    ``uncertainties`` explains what could not be resolved — the use case is
    never guessed from the project *name* alone.
    """

    primary_task: str = "unknown"
    secondary_tasks: List[str] = []
    confidence: str = "very_low"  # very_high | high | medium | low | very_low
    confidence_score: float = 0.0
    evidence: List[UseCaseSignal] = []
    reasoning: str = ""
    uncertainties: List[str] = []
    recommended_approach: str = "unknown"  # prompting | rag | fine_tuning | lora | qlora | combination
    approach_rationale: str = ""


class RiskItem(BaseModel):
    """A single, evidence-backed risk with an actionable mitigation."""

    risk: str
    why: str
    severity: str = "medium"  # critical | high | medium | low | info
    basis: str = ""  # why this severity — the defensible grounds
    mitigation: str = ""


class ModelCandidate(BaseModel):
    """One viable model option for the identified use case."""

    model: str
    why_it_fits: str = ""
    expected_strengths: List[str] = []
    expected_limitations: List[str] = []
    resource_requirements: str = ""
    cost_considerations: str = ""
    best_use_case: str = ""
    provenance: str = "reference_data"  # reference_data | externally_researched


class BudgetOption(BaseModel):
    """One budget tier (low cost / balanced / higher capability)."""

    tier: str
    approach: str
    estimated_api_cost_per_month: Optional[float] = None
    estimated_training_cost: Optional[float] = None
    rationale: str = ""
    assumptions: List[str] = []


class CompleteAnalysis(BaseModel):
    """The single authoritative end-to-end analysis produced by one Analyze run.

    This is an *orchestration* result: every section is derived from the real
    analyzer / prediction / recommendation / hardware outputs.  Sections that
    could not be produced are ``None`` and listed in ``unavailable_sections``
    with a reason — they are never filled with placeholder values.
    """

    project_name: str
    generated_at: Optional[datetime] = None
    analysis_status: str = "unknown"
    partial_failures: List[Dict[str, Any]] = []

    project_understanding: Optional[Dict[str, Any]] = None
    use_case: Optional[UseCaseAnalysisResult] = None
    dataset_analysis: Optional[Dict[str, Any]] = None
    prompt_analysis: Optional[Dict[str, Any]] = None
    model_analysis: Optional[Dict[str, Any]] = None
    model_candidates: List[ModelCandidate] = []
    model_comparison: Optional[Dict[str, Any]] = None
    hardware: Optional[Dict[str, Any]] = None
    training_configuration: Optional[Dict[str, Any]] = None
    training_time_estimate: Optional[Dict[str, Any]] = None
    cost_estimate: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    expected_behavior: Optional[Dict[str, Any]] = None
    approach_recommendation: Optional[Dict[str, Any]] = None
    risks: List[RiskItem] = []
    recommendations: List[Dict[str, Any]] = []
    next_experiment: Optional[Dict[str, Any]] = None

    model_information_notice: str = ""
    unavailable_sections: List[Dict[str, str]] = []
    markdown: str = ""


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ApiError(BaseModel):
    """API error response."""
    error_code: str
    message: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str