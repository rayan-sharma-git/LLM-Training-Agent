"""Hyperparameter configuration analyzer — reads real YAML/TOML config files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.schemas import ProjectContext, HyperparameterAnalysisResult
from ai.llm import LLMHelper
from cleaning import dataset_io

logger = logging.getLogger(__name__)


class HyperparameterAnalyzer:
    """Analyzes training hyperparameters from config files."""

    def __init__(self):
        self._llm = LLMHelper()

    async def analyze(self, context: ProjectContext, **kwargs) -> HyperparameterAnalysisResult:
        """Analyze hyperparameters from config files or inline values."""
        config_path = kwargs.get("config_path") or kwargs.get("configPath")
        logger.info("Analyzing hyperparameters")

        lr = kwargs.get("learning_rate")
        batch_size = kwargs.get("batch_size")
        epochs = kwargs.get("epochs")
        optimizer = kwargs.get("optimizer")
        lora_rank = kwargs.get("lora_rank")
        lora_alpha = kwargs.get("lora_alpha")
        lora_dropout = kwargs.get("lora_dropout")
        gradient_accumulation = kwargs.get("gradient_accumulation")

        # Try to read from config files if values not provided inline
        config_values: Dict[str, Any] = {}
        if not any([lr, batch_size, epochs]):
            config_values = self._read_config_files(context, config_path)
            lr = lr or config_values.get("learning_rate")
            batch_size = batch_size or config_values.get("batch_size")
            epochs = epochs or config_values.get("epochs")
            optimizer = optimizer or config_values.get("optimizer")
            lora_rank = lora_rank or config_values.get("lora_rank")
            lora_alpha = lora_alpha or config_values.get("lora_alpha")
            lora_dropout = lora_dropout or config_values.get("lora_dropout")
            gradient_accumulation = gradient_accumulation or config_values.get("gradient_accumulation")

        # IMPORTANT: do not fabricate values when the user's configuration does
        # not define them. Missing values stay None and are reported explicitly
        # in the findings/recommendations so the report never presents invented
        # numbers as the user's configuration.
        #
        # Track the provenance of every value so downstream layers (and the
        # user) can distinguish user/config-provided values from missing ones.
        sources = {
            "learning_rate": "user" if kwargs.get("learning_rate") else ("config" if lr else "missing"),
            "batch_size": "user" if kwargs.get("batch_size") else ("config" if batch_size else "missing"),
            "epochs": "user" if kwargs.get("epochs") else ("config" if epochs else "missing"),
            "optimizer": "user" if kwargs.get("optimizer") else ("config" if optimizer else "missing"),
        }

        # Step 1: Deterministic risk assessment (only over values that exist)
        overfitting_risk, underfitting_risk, efficiency_score, findings, recommendations = \
            self._assess_hyperparameters(lr, batch_size, epochs, optimizer,
                                         lora_rank, lora_alpha, lora_dropout,
                                         gradient_accumulation)

        known_core = [lr, batch_size, epochs]
        if all(v is not None for v in known_core):
            confidence = "high"
        elif any(v is not None for v in known_core):
            confidence = "medium"
        else:
            confidence = "very_low"
            recommendations.append(
                "No training configuration was found in this project "
                "(learning rate, batch size and epochs are all unknown). "
                "Add a training config (e.g. config.yaml) so they can be analyzed."
            )

        # Record provenance for the core values so estimates can never pass
        # missing values off as user-provided.
        display_names = {
            "learning_rate": "learning rate",
            "batch_size": "batch size",
            "epochs": "epochs",
            "optimizer": "optimizer",
        }
        for key, origin in sources.items():
            if origin == "missing":
                findings.append(f"{display_names[key]}: not found in the project configuration.")
            else:
                findings.append(f"{display_names[key]}: from {'request' if origin == 'user' else 'project configuration'}.")

        # Step 2: Optional LLM enhancement (only when real values exist)
        if self._llm.is_available and any(v is not None for v in known_core):
            llm_result = await self._llm_enhance(lr, batch_size, epochs, optimizer,
                                                  lora_rank, lora_alpha, lora_dropout)
            if llm_result:
                confidence = "high"
                if "overfittingRisk" in llm_result:
                    overfitting_risk = llm_result["overfittingRisk"]
                if "underfittingRisk" in llm_result:
                    underfitting_risk = llm_result["underfittingRisk"]

        return HyperparameterAnalysisResult(
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=kwargs.get("scheduler"),
            gradient_accumulation=gradient_accumulation,
            weight_decay=kwargs.get("weight_decay"),
            warmup_ratio=kwargs.get("warmup_ratio"),
            sequence_length=kwargs.get("sequence_length"),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            overfitting_risk=overfitting_risk,
            underfitting_risk=underfitting_risk,
            efficiency_score=round(efficiency_score, 2),
            findings=findings,
            recommendations=recommendations,
            confidence=confidence,
        )

    def _read_config_files(self, context: ProjectContext, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Read hyperparameter values from YAML/TOML/JSON config files."""
        values: Dict[str, Any] = {}

        # Determine which config files to read
        config_files: List[Path] = []
        if config_path:
            p = Path(config_path)
            if not p.is_absolute():
                p = Path(context.project_path) / config_path
            config_files.append(p)
        else:
            for cf in context.configuration_files:
                p = Path(context.project_path) / cf if not Path(cf).is_absolute() else Path(cf)
                if p.exists():
                    config_files.append(p)

        for cfg in config_files:
            if not cfg.exists():
                continue
            try:
                if cfg.suffix in (".yaml", ".yml"):
                    values.update(self._parse_yaml(cfg))
                elif cfg.suffix == ".toml":
                    values.update(self._parse_toml(cfg))
                elif cfg.suffix == ".json":
                    values.update(self._parse_json(cfg))
            except Exception as e:
                logger.warning(f"Failed to parse config {cfg}: {e}")

        return self._normalize_config_values(values)

    def _parse_yaml(self, path: Path) -> Dict[str, Any]:
        """Parse YAML config (without external dependencies)."""
        try:
            import yaml
            data = yaml.safe_load(dataset_io.read_text(path))
            if isinstance(data, dict):
                return data
            return {}
        except ImportError:
            # Fallback: simple regex-based extraction
            return self._extract_yaml_values(dataset_io.read_text(path))
        except Exception:
            return {}

    def _parse_toml(self, path: Path) -> Dict[str, Any]:
        """Parse TOML config."""
        try:
            import tomllib
            with open(path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import tomli
                with open(path, "rb") as f:
                    return tomli.load(f)
            except ImportError:
                return self._extract_toml_values(dataset_io.read_text(path))
        except Exception:
            return {}

    def _parse_json(self, path: Path) -> Dict[str, Any]:
        """Parse JSON config."""
        import json
        try:
            data = json.loads(dataset_io.read_text(path))
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}

    def _extract_yaml_values(self, text: str) -> Dict[str, Any]:
        """Extract hyperparameter values from YAML text using regex (fallback)."""
        result: Dict[str, Any] = {}
        patterns = {
            "learning_rate": r"learning_rate\s*:\s*([\d.eE+-]+)",
            "lr": r"\blr\s*:\s*([\d.eE+-]+)",
            "batch_size": r"batch_size\s*:\s*(\d+)",
            "per_device_train_batch_size": r"per_device_train_batch_size\s*:\s*(\d+)",
            "gradient_accumulation_steps": r"gradient_accumulation_steps\s*:\s*(\d+)",
            "num_train_epochs": r"num_train_epochs\s*:\s*([\d.]+)",
            "epochs": r"epochs\s*:\s*(\d+)",
            "weight_decay": r"weight_decay\s*:\s*([\d.eE+-]+)",
            "warmup_ratio": r"warmup_ratio\s*:\s*([\d.eE+-]+)",
            "max_seq_length": r"max_seq_length\s*:\s*(\d+)",
            "lora_rank": r"r\s*:\s*(\d+)",
            "lora_alpha": r"lora_alpha\s*:\s*(\d+)",
            "lora_dropout": r"lora_dropout\s*:\s*([\d.eE+-]+)",
            "optimizer": r"optim\w*\s*:\s*(\S+)",
        }
        for key, pat in patterns.items():
            match = __import__("re").search(pat, text)
            if match:
                val = match.group(1).strip().strip('"').strip("'")
                result[key] = val
        return result

    def _extract_toml_values(self, text: str) -> Dict[str, Any]:
        """Extract hyperparameter values from TOML text using regex (fallback)."""
        return self._extract_yaml_values(text)  # Same format, different brackets

    def _normalize_config_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config values to standard field names."""
        result: Dict[str, Any] = {}

        # Learning rate
        for key in ("learning_rate", "lr", "learningRate"):
            if key in values:
                try:
                    result["learning_rate"] = float(values[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Batch size
        for key in ("batch_size", "per_device_train_batch_size", "batchSize"):
            if key in values:
                try:
                    result["batch_size"] = int(values[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Epochs
        for key in ("num_train_epochs", "epochs", "numTrainEpochs"):
            if key in values:
                try:
                    result["epochs"] = int(float(values[key]))
                    break
                except (ValueError, TypeError):
                    pass

        # Gradient accumulation
        for key in ("gradient_accumulation_steps", "gradient_accumulation", "gradientAccumulationSteps"):
            if key in values:
                try:
                    result["gradient_accumulation"] = int(values[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Weight decay
        for key in ("weight_decay", "weightDecay"):
            if key in values:
                try:
                    result["weight_decay"] = float(values[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Optimizer
        for key in ("optim", "optimizer", "optim"):
            if key in values:
                result["optimizer"] = str(values[key])
                break

        # LoRA
        for key in ("r", "lora_rank", "loraRank"):
            if key in values:
                try:
                    result["lora_rank"] = int(values[key])
                    break
                except (ValueError, TypeError):
                    pass
        for key in ("lora_alpha", "loraAlpha"):
            if key in values:
                try:
                    result["lora_alpha"] = int(values[key])
                    break
                except (ValueError, TypeError):
                    pass
        for key in ("lora_dropout", "loraDropout"):
            if key in values:
                try:
                    result["lora_dropout"] = float(values[key])
                    break
                except (ValueError, TypeError):
                    pass

        return result

    def _assess_hyperparameters(
        self, lr, batch_size, epochs, optimizer,
        lora_rank, lora_alpha, lora_dropout, gradient_accumulation
    ) -> tuple[str, str, float, List[str], List[str]]:
        """Assess *provided* hyperparameters using deterministic rules.

        Only values that actually exist are assessed. Missing values are
        skipped (never defaulted), and their absence is reported instead so
        that recommendations never rest on invented numbers.
        """
        recommendations: List[str] = []
        findings: List[str] = []
        scores: List[float] = []
        risk_levels: List[str] = []

        # Learning rate risk (heuristic guidance, not guaranteed optimal values)
        if lr is not None:
            if lr > 1e-3:
                recommendations.append("Learning rate is high (>1e-3). Consider reducing for stability.")
                findings.append(f"Learning rate {lr} may cause training instability.")
                lr_risk = "high"
            elif lr > 5e-4:
                recommendations.append("Learning rate is moderate. Monitor for stability.")
                lr_risk = "medium"
            else:
                lr_risk = "low"
            risk_levels.append(lr_risk)
            scores.append({"high": 0.3, "medium": 0.7, "low": 1.0}[lr_risk])

        # Batch size efficiency
        if batch_size is not None:
            if batch_size < 4:
                recommendations.append("Batch size is small (<4). Consider increasing or using gradient accumulation.")
                findings.append(f"Batch size {batch_size} is below typical values for fine-tuning.")
                batch_risk = "high"
            elif batch_size < 8:
                batch_risk = "medium"
            else:
                batch_risk = "low"
            risk_levels.append(batch_risk)
            scores.append({"high": 0.3, "medium": 0.7, "low": 1.0}[batch_risk])

        # Epochs overfitting
        if epochs is not None:
            if epochs > 10:
                recommendations.append("Too many epochs (>10) may cause overfitting.")
                findings.append(f"Epochs {epochs} may lead to overfitting.")
                epoch_risk = "high"
            elif epochs > 5:
                epoch_risk = "medium"
            else:
                epoch_risk = "low"
            risk_levels.append(epoch_risk)
            scores.append({"high": 0.3, "medium": 0.7, "low": 1.0}[epoch_risk])

        # Gradient accumulation
        if gradient_accumulation and gradient_accumulation > 1:
            findings.append(f"Gradient accumulation steps: {gradient_accumulation}")

        if not scores:
            # Nothing was provided — report a neutral-but-unverifiable state
            # instead of pretending the configuration is fine.
            findings.append("No hyperparameters were available to assess.")
            return "unknown", "unknown", 0.5, findings, recommendations

        # Determine overall risk levels from the values that were assessed
        risk_map = {"high": 0, "medium": 1, "low": 2}
        overfitting_risk = max(risk_levels, key=lambda x: risk_map.get(x, 1))
        epochs_known = epochs is not None
        if epochs_known and batch_size is not None:
            underfitting_risk = (
                "low" if batch_risk == "low" and epochs < 5
                else "medium" if epochs < 10
                else "high"
            )
        else:
            underfitting_risk = "unknown"

        efficiency_score = sum(scores) / len(scores)
        return overfitting_risk, underfitting_risk, efficiency_score, findings, recommendations

    async def _llm_enhance(self, lr, batch_size, epochs, optimizer, lora_rank, lora_alpha, lora_dropout) -> Optional[Dict[str, Any]]:
        """Use LLM to enhance hyperparameter assessment."""
        try:
            prompt = f"""Analyze the following training hyperparameters for potential issues and optimization opportunities.

Learning rate: {lr}
Batch size: {batch_size}
Epochs: {epochs}
Optimizer: {optimizer or 'unknown'}
LoRA rank: {lora_rank or 'N/A'}
LoRA alpha: {lora_alpha or 'N/A'}
LoRA dropout: {lora_dropout or 'N/A'}

Return JSON with: efficiency_score (0-1), overfitting_risk (low|medium|high|very_high),
underfitting_risk (low|medium|high|very_high), recommendations (list), confidence (very_high|high|medium|low|very_low).
Base on training best practices. Include numerical evidence."""
            return await self._llm.call_structured(prompt, temperature=0.3)
        except Exception as e:
            logger.debug(f"LLM hyperparameter enhancement failed: {e}")
            return None
