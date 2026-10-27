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
        context_for(tmp_path, ["data"]), output_dir=str(out_dir), in_place=False
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
    # No manifest or log files are ever written next to the copies.
    assert sorted(p.name for p in out_dir.rglob("*") if p.is_file()) == ["train.jsonl", "val.jsonl"]
    # The originals were not touched in copy mode.
    assert len(read_jsonl(tmp_path / "data" / "train.jsonl")) == 12


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
        path=tmp_path / "data" / "train.jsonl", write_output=False, remove_duplicates=False
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
    cleaned = read_jsonl(tmp_path / "data" / "train.jsonl")
    assert cleaned[0]["prompt"] == "Hello\n\nWorld"
    assert not (tmp_path / ".llm-training-agent").exists()
    assert result.files[0].chunk_summaries[0].status == "deterministic"
    assert any("No AI provider" in w for w in result.warnings)


# ----------------------------------------------------------------------
# Script-only mode and duplicate removal
# ----------------------------------------------------------------------

async def test_scripts_mode_never_calls_the_llm_even_when_one_is_available(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(30))
    llm = RecordingLLM(available=True)
    cleaner = DatasetCleaner(llm=llm, chunk_size=10)

    result = await cleaner.clean_project(context_for(tmp_path, ["train.jsonl"]), use_llm=False)

    assert llm.prompts == []
    assert result.llm_used is False
    assert result.records_preserved is True
    assert not any("No AI provider" in w for w in result.warnings)


async def test_duplicates_can_be_kept_when_asked(tmp_path):
    records = make_records(3) + make_records(3)
    write_jsonl(tmp_path / "train.jsonl", records)
    cleaner = DatasetCleaner(llm=RecordingLLM(available=False), chunk_size=4)

    result = await cleaner.clean_project(context_for(tmp_path, ["train.jsonl"]), remove_duplicates=False)

    assert result.total_records_out == 6
    assert result.total_duplicates_removed == 0


async def test_remove_duplicates_drops_exact_copies_across_chunks_and_keeps_order(tmp_path):
    base = make_records(5)
    # The copy of record 0 sits in a different chunk; the third copy differs only by whitespace.
    records = base + [dict(base[0]), {**base[1], "prompt": "  p1  "}, {"id": 99, "prompt": "new", "response": "x"}]
    write_jsonl(tmp_path / "train.jsonl", records)
    cleaner = DatasetCleaner(llm=RecordingLLM(available=False), chunk_size=3)

    result = await cleaner.clean_project(
        context_for(tmp_path, ["train.jsonl"]), use_llm=False, remove_duplicates=True
    )

    cleaned = read_jsonl(tmp_path / "train.jsonl")
    assert [r["id"] for r in cleaned] == [0, 1, 2, 3, 4, 99]
    summary = result.files[0]
    assert summary.records_in == 8 and summary.records_out == 6
    assert summary.duplicates_removed == 2 == result.total_duplicates_removed
    assert result.records_preserved is True, "records_out + duplicates_removed must equal records_in"


async def test_remove_duplicates_is_a_script_step_in_llm_mode_too(tmp_path):
    records = make_records(4) + make_records(4)
    write_jsonl(tmp_path / "train.jsonl", records)
    llm = RecordingLLM(available=True)
    cleaner = DatasetCleaner(llm=llm, chunk_size=4)

    result = await cleaner.clean_project(
        context_for(tmp_path, ["train.jsonl"]), use_llm=True, remove_duplicates=True
    )

    assert len(llm.prompts) == 2, "the model still sees every chunk"
    assert result.total_records_out == 4 and result.total_duplicates_removed == 4


# ----------------------------------------------------------------------
# In-place cleaning, junk removal, AI removals, merge / discard
# ----------------------------------------------------------------------

async def test_scripts_rewrite_the_original_and_create_nothing_else(tmp_path):
    rows = [
        {"id": 1, "prompt": " a  ", "response": "b"},
        {"id": 1, "prompt": "a", "response": "b"},
        {"id": 2, "prompt": "???", "response": "c"},
        {"id": 3, "prompt": "q", "response": "  "},
        {"id": 4, "prompt": "ok", "response": "fine"},
    ]
    write_jsonl(tmp_path / "train.jsonl", rows)
    cleaner = DatasetCleaner(llm=RecordingLLM(available=True))

    result = await cleaner.clean_project(context_for(tmp_path, ["train.jsonl"]), use_llm=False)

    assert [r["id"] for r in read_jsonl(tmp_path / "train.jsonl")] == [1, 4]
    file = result.files[0]
    assert (file.records_in, file.records_out, file.duplicates_removed, file.junk_removed) == (5, 2, 1, 2)
    assert result.records_preserved and file.changed and result.in_place
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["train.jsonl"], "no extra files or folders"


async def test_already_clean_file_is_not_rewritten(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(3))
    before = (tmp_path / "train.jsonl").stat().st_mtime_ns
    result = await DatasetCleaner().clean_project(context_for(tmp_path, ["train.jsonl"]), use_llm=False)
    assert result.files[0].changed is False
    assert (tmp_path / "train.jsonl").stat().st_mtime_ns == before


async def test_json_wrapper_and_other_fields_survive_in_place(tmp_path):
    (tmp_path / "d.json").write_text(
        json.dumps({"name": "set", "data": [{"prompt": "a", "response": "b"}, {"prompt": "a", "response": "b"}]}),
        encoding="utf-8",
    )
    await DatasetCleaner().clean_project(context_for(tmp_path, ["d.json"]), use_llm=False)
    saved = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert saved["name"] == "set" and len(saved["data"]) == 1


async def test_chat_style_records_are_not_mistaken_for_junk(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    write_jsonl(tmp_path / "chat.jsonl", rows)
    result = await DatasetCleaner().clean_project(context_for(tmp_path, ["chat.jsonl"]), use_llm=False)
    assert result.total_junk_removed == 0 and len(read_jsonl(tmp_path / "chat.jsonl")) == 1


async def test_ai_can_fix_text_and_remove_a_hopeless_record(tmp_path):
    rows = make_records(6)
    write_jsonl(tmp_path / "train.jsonl", rows)

    def responder(chunk):
        out = []
        for r in chunk:
            r = dict(r)
            r["prompt"] = r["prompt"] + " fixed"
            if r["id"] == 2:
                r["_remove"] = True
            out.append(r)
        return {"records": out}

    cleaner = DatasetCleaner(llm=RecordingLLM(responder=responder), chunk_size=6)
    result = await cleaner.clean_project(context_for(tmp_path, ["train.jsonl"]), use_llm=True)

    saved = read_jsonl(tmp_path / "train.jsonl")
    assert [r["id"] for r in saved] == [0, 1, 3, 4, 5]
    assert saved[0]["prompt"].endswith("fixed") and all("_remove" not in r for r in saved)
    assert result.files[0].junk_removed == 1 and result.records_preserved and result.llm_used


async def test_model_that_flags_most_records_is_not_trusted(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(4))
    llm = RecordingLLM(responder=lambda chunk: {"records": [{**r, "_remove": True} for r in chunk]})
    result = await DatasetCleaner(llm=llm, chunk_size=4).clean_project(context_for(tmp_path, ["train.jsonl"]))
    assert len(read_jsonl(tmp_path / "train.jsonl")) == 4
    assert result.total_chunks_fallback == 1 and any("too many" in w for w in result.warnings)


async def test_model_cannot_blank_out_a_field(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(2))
    llm = RecordingLLM(responder=lambda chunk: {"records": [{**r, "response": ""} for r in chunk]})
    await DatasetCleaner(llm=llm).clean_project(context_for(tmp_path, ["train.jsonl"]))
    assert [r["response"] for r in read_jsonl(tmp_path / "train.jsonl")] == ["answer 0", "answer 1"]


async def test_progress_is_reported_per_chunk(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(9))
    seen = []
    await DatasetCleaner(llm=RecordingLLM(), chunk_size=4).clean_project(
        context_for(tmp_path, ["train.jsonl"]), on_progress=seen.append
    )
    assert [(p["chunk_index"], p["chunk_total"]) for p in seen] == [(1, 3), (2, 3), (3, 3)]
    assert seen[0]["file_name"] == "train.jsonl" and seen[0]["file_index"] == 1


async def test_copy_mode_then_merge_replaces_originals_and_removes_the_folder(tmp_path):
    from cleaning.dataset_cleaner import list_cleaned_copies, merge_cleaned_copies

    write_jsonl(tmp_path / "data" / "train.jsonl", make_records(3) + make_records(3))
    await DatasetCleaner().clean_project(context_for(tmp_path, ["data"]), use_llm=False, in_place=False)
    assert len(read_jsonl(tmp_path / "data" / "train.jsonl")) == 6, "original untouched in copy mode"
    assert list_cleaned_copies(str(tmp_path)) == ["data/train.jsonl"]

    assert merge_cleaned_copies(str(tmp_path)) == ["data/train.jsonl"]
    assert len(read_jsonl(tmp_path / "data" / "train.jsonl")) == 3
    assert not (tmp_path / ".llm-training-agent").exists()


async def test_discard_removes_copies_and_keeps_originals(tmp_path):
    from cleaning.dataset_cleaner import discard_cleaned_copies

    write_jsonl(tmp_path / "train.jsonl", make_records(3) + make_records(3))
    await DatasetCleaner().clean_project(context_for(tmp_path, ["train.jsonl"]), use_llm=False, in_place=False)
    assert discard_cleaned_copies(str(tmp_path)) == 1
    assert len(read_jsonl(tmp_path / "train.jsonl")) == 6
    assert not (tmp_path / ".llm-training-agent").exists()


def test_merge_ignores_a_leftover_manifest_from_older_versions(tmp_path):
    from cleaning.dataset_cleaner import merge_cleaned_copies

    old = tmp_path / ".llm-training-agent" / "cleaned"
    write_jsonl(old / "train.jsonl", make_records(2))
    (old / "cleaning_manifest.json").write_text("{}", encoding="utf-8")
    write_jsonl(tmp_path / "train.jsonl", make_records(5))
    merge_cleaned_copies(str(tmp_path))
    assert len(read_jsonl(tmp_path / "train.jsonl")) == 2
    assert not (tmp_path / "cleaning_manifest.json").exists()
    assert not (tmp_path / ".llm-training-agent").exists()


async def test_cleaned_folder_is_never_treated_as_a_dataset(tmp_path):
    write_jsonl(tmp_path / "data" / "a.jsonl", make_records(2))
    write_jsonl(tmp_path / ".llm-training-agent" / "cleaned" / "data" / "a.jsonl", make_records(2))
    files, _ = dataset_io.discover_dataset_files(tmp_path, ["."])
    assert [f.name for f in files] == ["a.jsonl"] and ".llm-training-agent" not in str(files[0])


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
            "inPlace": False,
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


def test_clean_endpoint_remove_duplicates_flag(api_client, tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(3) + make_records(3))
    response = api_client.post(
        "/api/v1/dataset/clean",
        json={"projectPath": str(tmp_path), "datasetPaths": ["train.jsonl"], "useLlm": False, "removeDuplicates": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_records_in"] == 6 and body["total_records_out"] == 3
    assert body["total_duplicates_removed"] == 3


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

def test_duplicates_that_differ_only_by_whitespace_are_removed():
    # AI output is not whitespace-normalised, so the comparison must be.
    from cleaning.dataset_cleaner import remove_duplicate_records

    kept, removed = remove_duplicate_records(
        [
            {"prompt": "q1", "response": "a1"},
            {"prompt": "q1 ", "response": " a1\n"},
            {"prompt": "q2", "response": "a2"},
        ]
    )
    assert removed == 1
    assert [r["prompt"] for r in kept] == ["q1", "q2"]


def test_background_job_reports_progress_and_result(api_client, tmp_path):
    import time

    write_jsonl(tmp_path / "train.jsonl", make_records(6) + make_records(6))
    start = api_client.post(
        "/api/v1/dataset/clean/start",
        json={"projectPath": str(tmp_path), "useLlm": False},
    )
    assert start.status_code == 200
    job_id = start.json()["jobId"]
    for _ in range(50):
        state = api_client.get(f"/api/v1/dataset/clean/jobs/{job_id}").json()
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "done", state
    assert state["result"]["total_duplicates_removed"] == 6
    assert len(read_jsonl(tmp_path / "train.jsonl")) == 6
    assert api_client.get("/api/v1/dataset/clean/jobs/nope").status_code == 404


def test_merge_and_discard_endpoints(api_client, tmp_path):
    write_jsonl(tmp_path / "train.jsonl", make_records(4) + make_records(4))
    api_client.post(
        "/api/v1/dataset/clean",
        json={"projectPath": str(tmp_path), "useLlm": False, "inPlace": False},
    )
    pending = api_client.post("/api/v1/dataset/clean/pending", json={"projectPath": str(tmp_path)}).json()
    assert pending["files"] == ["train.jsonl"]
    merged = api_client.post("/api/v1/dataset/clean/merge", json={"projectPath": str(tmp_path)}).json()
    assert merged["merged"] == ["train.jsonl"]
    assert len(read_jsonl(tmp_path / "train.jsonl")) == 4
    assert api_client.post("/api/v1/dataset/clean/discard", json={"projectPath": str(tmp_path)}).json() == {"discarded": 0}
