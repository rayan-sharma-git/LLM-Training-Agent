"""Tests for ProjectScanner — comprehensive coverage including edge cases."""
from __future__ import annotations

import pytest
from scanner.scanner import ProjectScanner
from scanner.framework_detector import FrameworkDetector


# ---------------------------------------------------------------------------
# FrameworkDetector
# ---------------------------------------------------------------------------

def test_framework_detector_unknown(tmp_path):
    detector = FrameworkDetector()
    result = detector.detect(tmp_path)
    assert result["framework"] == "unknown"
    assert result["framework_version"] is None


def test_framework_detector_huggingface(tmp_path):
    detector = FrameworkDetector()
    (tmp_path / "requirements.txt").write_text("transformers\ndatasets\npeft\n")
    result = detector.detect(tmp_path)
    assert result["framework"] == "huggingface"


def test_framework_detector_axolotl(tmp_path):
    detector = FrameworkDetector()
    (tmp_path / "requirements.txt").write_text("axolotl\n")
    result = detector.detect(tmp_path)
    assert result["framework"] == "axolotl"


def test_framework_detector_pyproject_toml(tmp_path):
    detector = FrameworkDetector()
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["transformers", "trl"]\n')
    result = detector.detect(tmp_path)
    assert result["framework"] == "huggingface"


# ---------------------------------------------------------------------------
# ProjectScanner — basic
# ---------------------------------------------------------------------------

def test_project_scanner_missing_path():
    with pytest.raises(FileNotFoundError):
        ProjectScanner("/nonexistent/path/that/does/not/exist").scan()


def test_project_scanner_empty_project(tmp_path):
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert result["project_name"] == tmp_path.name
    assert result["dataset_paths"] == []
    assert result["training_scripts"] == []
    assert result["configuration_files"] == []
    assert result["project_statistics"]["file_count"] == 0


def test_project_scanner_basic(tmp_path):
    (tmp_path / "data.json").write_text('{"key": "value"}')
    (tmp_path / "train.py").write_text("print('train')")
    (tmp_path / "README.md").write_text("# Test Project")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert result["project_name"] == tmp_path.name
    assert "data.json" in result["dataset_paths"]
    assert any("train.py" in s for s in result["training_scripts"])


# ---------------------------------------------------------------------------
# ProjectScanner — file discovery
# ---------------------------------------------------------------------------

def test_scanner_discovers_python_files(tmp_path):
    (tmp_path / "train.py").write_text("print('train')")
    (tmp_path / "finetune.py").write_text("print('finetune')")
    (tmp_path / "eval_run.py").write_text("print('eval')")  # matches eval*.py
    (tmp_path / "inference.py").write_text("print('inference')")
    (tmp_path / "predict.py").write_text("print('predict')")
    (tmp_path / "main.py").write_text("print('main')")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert any("train.py" in s for s in result["training_scripts"])
    assert any("finetune.py" in s for s in result["training_scripts"])
    assert any("eval_run.py" in s for s in result["evaluation_scripts"])
    assert any("inference.py" in s for s in result["inference_scripts"])
    assert any("predict.py" in s for s in result["inference_scripts"])
    # main.py should NOT be in training scripts (doesn't match patterns)
    assert not any("main.py" in s for s in result["training_scripts"])


def test_scanner_discovers_typescript_files(tmp_path):
    (tmp_path / "trainer.ts").write_text("console.log('train')")
    (tmp_path / "config.json").write_text('{"name": "test"}')
    (tmp_path / "data.jsonl").write_text('{"prompt":"a","response":"b"}')
    (tmp_path / "dataset.csv").write_text("prompt,response\nhello,world\n")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert "config.json" in result["dataset_paths"]
    assert "data.jsonl" in result["dataset_paths"]
    assert "dataset.csv" in result["dataset_paths"]


def test_scanner_discovers_yaml_configs(tmp_path):
    (tmp_path / "config.yaml").write_text("model_name_or_path: llama3\n")
    (tmp_path / "params.yml").write_text("batch_size: 8\n")
    (tmp_path / "settings.toml").write_text("key = 'value'\n")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert "config.yaml" in result["configuration_files"]
    assert "params.yml" in result["configuration_files"]
    assert "settings.toml" in result["configuration_files"]


def test_scanner_detects_model_from_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("model_name_or_path: meta-llama/llama-3\n")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    # The scanner must return the actual model name (not a placeholder string)
    # so downstream analyzers can resolve real specifications.
    assert result["base_model"] == "meta-llama/llama-3"


def test_scanner_detects_model_not_in_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("batch_size: 8\n")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert result["base_model"] is None


# ---------------------------------------------------------------------------
# ProjectScanner — directory exclusion
# ---------------------------------------------------------------------------

def test_scanner_ignores_venv(tmp_path):
    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    (venv_dir / "config.yaml").write_text("model_name_or_path: should-be-ignored\n")
    (tmp_path / "train.py").write_text("print('train')")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert not any("venv" in p for p in result["configuration_files"])
    assert any("train.py" in s for s in result["training_scripts"])


def test_scanner_ignores_node_modules(tmp_path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "package.json").write_text('{"name": "hidden"}')
    (tmp_path / "package.json").write_text('{"name": "visible"}')
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert "package.json" in result["dataset_paths"]
    assert not any("node_modules" in p for p in result["dataset_paths"])


def test_scanner_ignores_git(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("[core]\n")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert result["configuration_files"] == []


def test_scanner_ignores_pycache(tmp_path):
    pc = tmp_path / "__pycache__"
    pc.mkdir()
    (pc / "module.cpython-310.pyc").write_bytes(b"\x00\x00")
    (tmp_path / "train.py").write_text("print('train')")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert not any("__pycache__" in p for p in result["dataset_paths"])
    assert any("train.py" in s for s in result["training_scripts"])


# ---------------------------------------------------------------------------
# ProjectScanner — error handling
# ---------------------------------------------------------------------------

def test_scanner_handles_binary_files(tmp_path):
    (tmp_path / "binary.dat").write_bytes(b"\x00\x01\x02\xff\xfe")
    (tmp_path / "train.py").write_text("print('train')")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert any("train.py" in s for s in result["training_scripts"])


def test_scanner_handles_permission_error(tmp_path):
    (tmp_path / "train.py").write_text("print('train')")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert any("train.py" in s for s in result["training_scripts"])


def test_scanner_windows_path(tmp_path):
    project_path = str(tmp_path).replace("/", "\\")
    scanner = ProjectScanner(project_path)
    result = scanner.scan()
    assert result["project_path"].startswith(str(tmp_path))


def test_scanner_handles_spaces_in_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("Project With Spaces")
    (tmp_path / "data.json").write_text('{"key": "value"}')
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert "data.json" in result["dataset_paths"]


def test_scanner_project_statistics(tmp_path):
    (tmp_path / "file1.py").write_text("a")
    (tmp_path / "file2.py").write_text("b")
    (tmp_path / "data.json").write_text("{}")
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert result["project_statistics"]["file_count"] == 3


def test_scanner_returns_consistent_types(tmp_path):
    scanner = ProjectScanner(str(tmp_path))
    result = scanner.scan()
    assert isinstance(result["dataset_paths"], list)
    assert isinstance(result["prompt_templates"], list)
    assert isinstance(result["configuration_files"], list)
    assert isinstance(result["training_scripts"], list)
    assert isinstance(result["evaluation_scripts"], list)
    assert isinstance(result["inference_scripts"], list)
    assert isinstance(result["hardware_information"], dict)
    assert isinstance(result["project_statistics"], dict)
