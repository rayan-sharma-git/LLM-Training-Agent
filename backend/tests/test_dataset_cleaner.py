"""Tests for the chunked, multi-file DatasetCleaner pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from api.routes import router
from ai.llm import LLMHelper
from cleaning import dataset_io
from cleaning.dataset_cleaner import DatasetCleaner
from models.schemas import ProjectContext


class RecordingLLM(LLMHelper):
    """Deterministic LLMHelper test double that records every request.

    ``responder`` receives the JSON array of records that was actually placed
    inside the prompt, which lets tests prove *what* the model was sent.
    """

    def __init__(self, responder: Optional[Any] = None, available: bool = True):
        super().__init__()
        self._available = available
        self.prompts: List[str] = []
        self.system_prompts: List[str] = []
        self.sent_batches: List[List[Dict[str, Any]]] = []
        self._responder = responder or (lambda chunk: {"records": chunk})

    @property
    def is_available(self) -> bool:  # type: ignore[override]
        return self._available

    async def call_structured(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt or "")
        chunk = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        self.sent_batches.append(chunk)
        return self._responder(chunk)


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def make_records(count: int, prefix: str = "p", source: str = "train") -> List[Dict[str, Any]]:
    return [
        {"id": i, "source": source, "prompt": f"{prefix}{i}", "response": f"answer {i}"}
        for i in range(count)
    ]


def context_for(root: Path, dataset_paths: List[str]) -> ProjectContext:
    return ProjectContext(
        project_name="test_project",
        project_path=str(root),
        dataset_paths=dataset_paths,
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------

def test_chunk_records_is_order_preserving_and_bounded():
    records = make_records(60)
    chunks = dataset_io.chunk_records(records, chunk_size=25, max_chunk_chars=10**6)

    assert [len(c) for c in chunks] == [25, 25, 10]
    flattened = [r["id"] for chunk in chunks for r in chunk]
    assert flattened == list(range(60))


def test_chunk_records_respects_char_budget():
    records = [
        {"id": i, "prompt": "x" * 400, "response": "y" * 400} for i in range(10)
    ]
    chunks = dataset_io.chunk_records(records, chunk_size=100, max_chunk_chars=1200)

    assert len(chunks) > 1
    for chunk in chunks:
        serialized = json.dumps(chunk, ensure_ascii=False)
        assert len(serialized) <= 1200 or len(chunk) == 1


def test_oversized_record_gets_its_own_chunk_and_is_never_dropped():
    records = [{"id": 0, "text": "z" * 5000}, {"id": 1, "text": "small"}]
    chunks = dataset_io.chunk_records(records, chunk_size=10, max_chunk_chars=500)

    assert [r["id"] for chunk in chunks for r in chunk] == [0, 1]


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def test_discover_expands_directories_and_deduplicates(tmp_path):
    write_jsonl(tmp_path / "data" / "a.jsonl", make_records(3))
    write_jsonl(tmp_path / "data" / "nested" / "b.json", make_records(2))
    (tmp_path / "data" / "ignored.bin").write_text("nope", encoding="utf-8")

    files, skipped = dataset_io.discover_dataset_files(
        tmp_path, ["data", "data/a.jsonl", "missing.jsonl"]
    )

    names = sorted(f.name for f in files)
    assert names == ["a.jsonl", "b.json"]
    assert any("missing.jsonl" in entry for entry in skipped)


# ----------------------------------------------------------------------
# Multi-file chunked cleaning
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_chunk_is_sent_in_its_own_request(tmp_path):
    write_jsonl(tmp_path / "data" / "train.jsonl", make_records(60))
    llm = RecordingLLM()
    cleaner = DatasetCleaner(llm=llm, chunk_size=25, max_chunk_chars=10**6)

    result = await cleaner.clean_project(
        context_for(tmp_path, ["data/train.jsonl"]), write_output=False
    )

    assert len(llm.sent_batches) == 3, "one LLM request per chunk"
    assert [len(b) for b in llm.sent_batches] == [25, 25, 10]
    assert result.total_chunks == 3
    assert result.total_chunks_cleaned == 3
    # The full dataset was never in a single request.
    assert all(len(batch) < 60 for batch in llm.sent_batches)
    # Placeholders were substituted with the real chunk metadata.
    assert "<<" not in llm.prompts[0]
    assert "train.jsonl" in llm.prompts[0]
    assert llm.system_prompts[0].startswith("# System Prompt - Dataset Cleaner")


@pytest.mark.asyncio
async def test_all_files_are_processed_and_never_mixed(tmp_path):
    write_jsonl(tmp_path / "data" / "train.jsonl", make_records(30, source="train"))
    write_jsonl(tmp_path / "data" / "val.jsonl", make_records(40, source="val"))
    llm = RecordingLLM()
    cleaner = DatasetCleaner(llm=llm, chunk_size=20, max_chunk_chars=10**6)

    result = await cleaner.clean_project(
        context_for(tmp_path, ["data"]), write_output=False
    )

    assert result.total_files == 2
    assert {f.relative_path for f in result.files} == {
        "data/train.jsonl",
        "data/val.jsonl",
    }
    assert result.total_records_in == 70
    assert result.total_records_out == 70
    assert result.records_preserved is True
    assert result.cross_file_contamination is False
    # Every single request contained records from exactly one file.
    for batch in llm.sent_batches:
        assert len({record["source"] for record in batch}) == 1


@pytest.mark.asyncio
async def test_cleaned_files_are_reconstructed_per_source_file(tmp_path):
    write_jsonl(tmp_path / "data" / "train.jsonl", make_records(12, source="train"))
    write_jsonl(tmp_path / "data" / "val.jsonl", make_records(7, source="val"))
    llm = RecordingLLM()
    cleaner = DatasetCleaner(llm=llm, chunk_size=5, max_chunk_chars=10**6)
    out_dir = tmp_path / "out"

    result = await cleaner.clean_project(
        context_for(tmp_path, ["data"]), output_dir=str(out_dir)
    )

    train_out = out_dir / "data" / "train.jsonl"
    val_out = out_dir / "data" / "val.jsonl"
    assert train_out.exists() and val_out.exists()

    train_records = read_jsonl(train_out)
    val_records = read_jsonl(val_out)
    assert len(train_records) == 12
    assert len(val_records) == 7
    assert {r["source"] for r in train_records} == {"train"}
    assert {r["source"] for r in val_records} == {"val"}
    # Original order is preserved after reconstruction.
    assert [r["id"] for r in train_records] == list(range(12))

    assert result.records_preserved is True
    assert result.manifest_path is not None
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["records_preserved"] is True
    assert {entry["relative_path"] for entry in manifest["files"]} == {
        "data/train.jsonl",
        "data/val.jsonl",
    }
    assert all(entry["records_in"] == entry["records_out"] for entry in manifest["files"])


@pytest.mark.asyncio
async def test_llm_output_values_are_applied_without_losing_fields(tmp_path):
    write_jsonl(tmp_path / "data" / "train.jsonl", [{"id": 1, "prompt": "  hi  ", "response": "yo"}] * 2)

    def responder(chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "records": [
                {"id": record["id"], "prompt": "hi", "invented_field": "drop me"}
                for record in chunk
            ]
        }

    llm = RecordingLLM(responder=responder)
    cleaner = DatasetCleaner(llm=llm, chunk_size=10)

    records, summary = await cleaner.clean_file(
        path=tmp_path / "data" / "train.jsonl", write_output=False
    )

    assert len(records) == 2
    assert records[0]["prompt"] == "hi"
    # Dropped key restored from the source record, invented key ignored.
    assert records[0]["response"] == "yo"
    assert "invented_field" not in records[0]
    assert summary.records_preserved is True
    assert summary.chunks_cleaned == 1


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contract_violation_falls_back_and_keeps_originals(tmp_path):
    write_jsonl(tmp_path / "data" / "train.jsonl", make_records(6))
    # Emulate a model that silently drops half of the records.
    llm = RecordingLLM(responder=lambda chunk: {"records": chunk[: len(chunk) // 2]})
    cleaner = DatasetCleaner(llm=llm, chunk_size=3)

    records, summary = await cleaner.clean_file(
        path=tmp_path / "data" / "train.jsonl", write_output=False
    )

    assert len(records) == 6
    assert summary.records_preserved is True
    assert summary.chunks_cleaned == 0
    assert summary.chunks_fallback == 2
    assert any("record contract" in w for w in summary.warnings)


@pytest.mark.asyncio
async def test_unparsable_llm_response_falls_back(tmp_path):
    write_jsonl(tmp_path / "data" / "train.jsonl", make_records(4))
    llm = RecordingLLM(responder=lambda _chunk: None)
    cleaner = DatasetCleaner(llm=llm, chunk_size=2)

    records, summary = await cleaner.clean_file(
        path=tmp_path / "data" / "train.jsonl", write_output=False
    )

    assert [r["id"] for r in records] == [0, 1, 2, 3]
    assert summary.records_preserved is True
    assert summary.llm_used is False


@pytest.mark.asyncio
async def test_no_provider_applies_deterministic_cleaning(tmp_path):
    write_jsonl(
        tmp_path / "data" / "train.jsonl",
        [{"id": 1, "prompt": "  Hello   \n\n\n\nWorld \t\n", "response": "Fine"}],
    )
    llm = RecordingLLM(available=False)
    cleaner = DatasetCleaner(llm=llm, chunk_size=5)

    result = await cleaner.clean_project(context_for(tmp_path, ["data"]))

    assert result.llm_used is False
    assert llm.prompts == [], "no request may be sent when no provider is configured"
    assert result.records_preserved is True
    cleaned = read_jsonl(tmp_path / ".llm-training-agent" / "cleaned" / "data" / "train.jsonl")
    assert cleaned[0]["prompt"] == "Hello\n\nWorld"
    assert result.files[0].chunk_summaries[0].status == "deterministic"
    assert any("No AI provider" in w for w in result.warnings)


# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------

@pytest.fixture
def api_client():
    """Sync FastAPI test client (same pattern as tests/test_api.py)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_clean_endpoint_accepts_explicit_dataset_paths(api_client, tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(9, source="train"))
    out_dir = tmp_path / "cleaned_out"

    response = api_client.post(
        "/api/v1/dataset/clean",
        json={
            "datasetPaths": [str(tmp_path / "train.jsonl")],
            "chunkSize": 4,
            "outputDirectory": str(out_dir),
            "useLlm": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_files"] == 1
    assert payload["total_records_in"] == payload["total_records_out"] == 9
    assert payload["records_preserved"] is True
    assert payload["cross_file_contamination"] is False
    assert payload["total_chunks"] == 3
    assert (out_dir / "train.jsonl").exists()


def test_clean_endpoint_discovers_project_dataset_files(api_client, tmp_path):
    write_jsonl(tmp_path / "dataset" / "a.jsonl", make_records(3, source="a"))
    write_jsonl(tmp_path / "dataset" / "b.jsonl", make_records(4, source="b"))

    response = api_client.post(
        "/api/v1/dataset/clean",
        json={"projectPath": str(tmp_path), "useLlm": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_files"] == 2
    assert payload["total_records_in"] == payload["total_records_out"] == 7
    assert {f["relative_path"] for f in payload["files"]} == {
        "dataset/a.jsonl",
        "dataset/b.jsonl",
    }


def test_clean_endpoint_requires_a_target(api_client):
    response = api_client.post("/api/v1/dataset/clean", json={})
    assert response.status_code == 400


def test_clean_endpoint_rejects_unknown_project_path(api_client):
    response = api_client.post(
        "/api/v1/dataset/clean", json={"projectPath": "Z:/does/not/exist"}
    )
    assert response.status_code == 404


# ----------------------------------------------------------------------
# LLMHelper JSON handling (used by the chunked call path)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_helper_wraps_bare_json_array(monkeypatch):
    helper = LLMHelper()

    class FakeProvider:
        async def chat_completion(self, messages, **kwargs):
            return {"content": "```json\n[{\"prompt\": \"a\"}]\n```", "model": "fake"}

    monkeypatch.setattr(LLMHelper, "is_available", property(lambda self: True))
    monkeypatch.setattr(LLMHelper, "provider", property(lambda self: FakeProvider()))

    result = await helper.call_structured("chunk prompt")

    assert result == {"items": [{"prompt": "a"}]}


@pytest.mark.asyncio
async def test_llm_helper_rejects_non_json_payload(monkeypatch):
    helper = LLMHelper()

    class FakeProvider:
        async def chat_completion(self, messages, **kwargs):
            return {"content": "I cleaned your data, trust me.", "model": "fake"}

    monkeypatch.setattr(LLMHelper, "is_available", property(lambda self: True))
    monkeypatch.setattr(LLMHelper, "provider", property(lambda self: FakeProvider()))

    assert await helper.call_structured("chunk prompt") is None