"""Cleaning flow: separate vs in-place output, durable jobs, safe merge, concise failures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleaning.dataset_cleaner import DatasetCleaner
from cleaning.jobs import CleaningJobError, CleaningJobs
from tests.test_dataset_cleaner import RecordingLLM, context_for, make_records, read_jsonl, write_jsonl


def _project(tmp_path: Path, count: int = 6) -> Path:
    root = tmp_path / "proj"
    write_jsonl(root / "data" / "train.jsonl", [
        {"id": i, "prompt": f"  q{i}  \n\n\n\nmore ", "response": f"a{i}"} for i in range(count)])
    write_jsonl(root / "data" / "eval.jsonl", make_records(3, source="eval"))
    return root


def _cleaner(**kwargs):
    return DatasetCleaner(llm=RecordingLLM(**kwargs), chunk_size=2)


@pytest.mark.asyncio
async def test_separate_mode_leaves_originals_untouched_and_records_the_job(tmp_path):
    root = _project(tmp_path)
    before = (root / "data" / "train.jsonl").read_bytes()
    jobs = CleaningJobs(root)
    job = await jobs.run(context_for(root, ["data"]), _cleaner(available=False), "script", "separate")

    assert (root / "data" / "train.jsonl").read_bytes() == before, "separate mode must not touch originals"
    assert job["status"] == "completed" and job["outputMode"] == "separate" and job["method"] == "script"
    cleaned = root / job["outputDirectory"] / "data" / "train.jsonl"
    assert read_jsonl(cleaned)[0]["prompt"] == "q0\n\nmore"
    # the job is on disk, so a new object (restart / other chat) finds it
    again = CleaningJobs(root).pending_merges()
    assert [j["jobId"] for j in again] == [job["jobId"]]
    assert {f["relativePath"] for f in again[0]["files"]} == {"data/train.jsonl", "data/eval.jsonl"}


@pytest.mark.asyncio
async def test_merge_replaces_originals_backs_up_and_removes_temp_folder(tmp_path):
    root = _project(tmp_path)
    original = (root / "data" / "train.jsonl").read_bytes()
    jobs = CleaningJobs(root)
    job = await jobs.run(context_for(root, ["data"]), _cleaner(available=False), "script", "separate")
    cleaned_bytes = (root / job["outputDirectory"] / "data" / "train.jsonl").read_bytes()

    report = CleaningJobs(root).merge(job["jobId"])  # fresh object: "after restarting VS Code"

    assert report["status"] == "merged" and not report["skipped"]
    assert (root / "data" / "train.jsonl").read_bytes() == cleaned_bytes
    assert report["removedTemp"] is True
    assert not (root / job["outputDirectory"]).exists()
    assert not (root / ".llm-training-agent" / "cleaned").exists()
    backup = root / report["backupDirectory"] / "data" / "train.jsonl"
    assert backup.read_bytes() == original
    assert CleaningJobs(root).pending_merges() == []
    with pytest.raises(CleaningJobError):
        jobs.merge(job["jobId"])  # cannot merge twice


@pytest.mark.asyncio
async def test_merge_refuses_to_overwrite_a_file_edited_after_cleaning(tmp_path):
    root = _project(tmp_path)
    jobs = CleaningJobs(root)
    job = await jobs.run(context_for(root, ["data"]), _cleaner(available=False), "script", "separate")
    edited = root / "data" / "train.jsonl"
    edited.write_text(edited.read_text(encoding="utf-8") + json.dumps({"id": 99, "prompt": "new", "response": "x"}) + "\n",
                      encoding="utf-8")
    mine = edited.read_bytes()

    report = jobs.merge(job["jobId"])

    assert edited.read_bytes() == mine, "the user's newer edit must survive"
    assert [s["file"] for s in report["skipped"]] == ["data/train.jsonl"]
    assert "changed after it was cleaned" in report["skipped"][0]["reason"]
    assert report["merged"] == ["data/eval.jsonl"]
    assert report["removedTemp"] is False, "leftover cleaned copies must be kept"
    assert (root / job["outputDirectory"] / "data" / "train.jsonl").exists()
    assert jobs.get(job["jobId"])["status"] == "partially_merged"


@pytest.mark.asyncio
async def test_in_place_mode_modifies_originals_immediately(tmp_path):
    root = _project(tmp_path)
    job = await CleaningJobs(root).run(context_for(root, ["data"]), _cleaner(available=False), "script", "in_place")
    assert job["status"] == "merged"
    assert read_jsonl(root / "data" / "train.jsonl")[0]["prompt"] == "q0\n\nmore"
    assert not (root / ".llm-training-agent" / "cleaned").exists()
    assert (root / job["mergeReport"]["backupDirectory"] / "data" / "train.jsonl").exists()


@pytest.mark.asyncio
async def test_restore_brings_back_originals_unless_edited(tmp_path):
    root = _project(tmp_path)
    original = (root / "data" / "train.jsonl").read_bytes()
    jobs = CleaningJobs(root)
    job = await jobs.run(context_for(root, ["data"]), _cleaner(available=False), "script", "in_place")
    result = jobs.restore(job["jobId"])
    assert sorted(result["restored"]) == ["data/eval.jsonl", "data/train.jsonl"]
    assert (root / "data" / "train.jsonl").read_bytes() == original


@pytest.mark.asyncio
async def test_unparsable_llm_output_is_summarised_once_and_originals_kept(tmp_path):
    root = tmp_path / "proj"
    records = make_records(10)
    write_jsonl(root / "data" / "train.jsonl", records)
    llm = RecordingLLM(responder=lambda chunk: None)  # every chunk unparsable
    cleaner = DatasetCleaner(llm=llm, chunk_size=2)

    result = await cleaner.clean_project(context_for(root, ["data"]), write_output=False)

    file_summary = result.files[0]
    assert file_summary.chunks_total == 5 and file_summary.chunks_fallback == 5
    # one line for the file, not five
    assert len([w for w in file_summary.warnings if "Chunk" in w]) == 0
    assert len(file_summary.warnings) == 1
    assert "5 of 5 chunk(s)" in file_summary.warnings[0] and "10 record(s)" in file_summary.warnings[0]
    assert len(result.warnings) == 1
    assert result.failure_summary["recordsKeptOriginal"] == 10
    assert "kept unchanged" in result.summary_text
    # originals are kept verbatim (not even whitespace-normalised)
    cleaned, _ = await cleaner.clean_file(root / "data" / "train.jsonl", write_output=False)
    assert cleaned == records


@pytest.mark.asyncio
async def test_script_method_never_calls_the_llm(tmp_path):
    root = _project(tmp_path)
    llm = RecordingLLM()
    await CleaningJobs(root).run(context_for(root, ["data"]), DatasetCleaner(llm=llm, chunk_size=2),
                                 "script", "separate")
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_invalid_options_are_rejected(tmp_path):
    root = _project(tmp_path)
    with pytest.raises(CleaningJobError):
        await CleaningJobs(root).run(context_for(root, ["data"]), _cleaner(), "magic", "separate")
    with pytest.raises(CleaningJobError):
        await CleaningJobs(root).run(context_for(root, ["data"]), _cleaner(), "script", "elsewhere")
    with pytest.raises(CleaningJobError):
        await CleaningJobs(root).run(context_for(root, ["missing"]), _cleaner(), "script", "separate")


def test_job_ids_cannot_escape_the_jobs_folder(tmp_path):
    with pytest.raises(CleaningJobError):
        CleaningJobs(tmp_path).get("../../etc/passwd")
