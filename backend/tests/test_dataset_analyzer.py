"""Tests for DatasetAnalyzer."""
import json
import pytest
from analyzers.dataset_analyzer import DatasetAnalyzer
from models.schemas import ProjectContext


@pytest.fixture
def analyzer():
    return DatasetAnalyzer()


@pytest.fixture
def sample_context(tmp_path):
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps({"prompt": "hello", "response": "world"}))
    return ProjectContext(project_name="test", project_path=str(tmp_path), dataset_paths=["data.json"])


@pytest.mark.asyncio
async def test_dataset_analyzer_success(analyzer, sample_context):
    result = await analyzer.analyze(sample_context)
    assert result.dataset_name == "data.json"
    assert hasattr(result, "sample_count")
    assert hasattr(result, "quality_score")
    assert result.confidence in ("low", "medium", "high")


@pytest.mark.asyncio
async def test_dataset_analyzer_no_dataset(analyzer):
    context = ProjectContext(project_name="test", project_path="/tmp", dataset_paths=[])
    with pytest.raises(Exception):
        await analyzer.analyze(context)
