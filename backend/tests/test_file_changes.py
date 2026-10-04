"""Tests for the propose → review → approve → apply → verify → rollback workflow.

These tests assert *real* behaviour against a real filesystem: they read files
back after every operation, so a placeholder or stubbed implementation would
fail.  The security cases (traversal, absolute escape, symlink escape,
workspace mismatch, malicious LLM output) verify that no file is created or
modified outside the authorized project.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient whose authorized workspace is this test's tmp_path."""
    root = tmp_path / "project"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(root))
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app), root


def _write(root: Path, name: str, content: str) -> str:
    """Create a file with exactly *content* (no Windows newline translation)."""
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return name


def _propose(test_client, root, name, content, **extra):
    payload = {"filePath": name, "proposedContent": content, "projectRoot": str(root)}
    payload.update(extra)
    return test_client.post("/api/v1/files/propose", json=payload)


def _approve(test_client, root, change_id):
    return test_client.post(
        f"/api/v1/files/changes/{change_id}/approve", json={"projectRoot": str(root)}
    )


def _apply(test_client, root, change_id):
    return test_client.post(
        f"/api/v1/files/changes/{change_id}/apply", json={"projectRoot": str(root)}
    )


# ---------------------------------------------------------------------------
# Propose / review
# ---------------------------------------------------------------------------
def test_propose_generates_diff_without_modifying_file(client):
    test_client, root = client
    name = _write(root, "src/app.py", "line1\nline2\n")

    resp = _propose(test_client, root, name, "line1\nline2_changed\nline3\n")
    assert resp.status_code == 200
    data = resp.json()

    assert data["filePath"] == name
    assert data["status"] == "PROPOSED"
    assert data["operation"] == "modify"
    assert data["additions"] == 2
    assert data["deletions"] == 1
    assert "-line2" in data["diff"] and "+line2_changed" in data["diff"]
    assert data["originalContent"] == "line1\nline2\n"
    assert data["proposedContent"] == "line1\nline2_changed\nline3\n"
    assert data["baseHash"]

    # The file on disk is untouched at proposal time.
    assert (root / name).read_text(encoding="utf-8") == "line1\nline2\n"


def test_list_omits_contents_and_get_includes_them(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    change_id = _propose(test_client, root, name, "y\n").json()["changeId"]

    listed = test_client.get("/api/v1/files/changes", params={"projectRoot": str(root)})
    assert listed.status_code == 200
    changes = listed.json()["changes"]
    assert len(changes) == 1
    assert "proposedContent" not in changes[0]
    assert "originalContent" not in changes[0]

    detail = test_client.get(
        f"/api/v1/files/changes/{change_id}", params={"projectRoot": str(root)}
    )
    assert detail.status_code == 200
    assert detail.json()["originalContent"] == "x\n"
    assert detail.json()["conflicted"] is False


def test_proposal_survives_a_new_store_instance(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    change_id = _propose(test_client, root, name, "y\n").json()["changeId"]

    # Each request builds a fresh store; the proposal must persist on disk.
    stored = root / ".llm_training_agent" / "pending_changes" / f"{change_id}.json"
    assert stored.is_file()
    assert json.loads(stored.read_text(encoding="utf-8"))["changeId"] == change_id

    resp = test_client.get(
        f"/api/v1/files/changes/{change_id}", params={"projectRoot": str(root)}
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# The approval gate
# ---------------------------------------------------------------------------
def test_apply_without_approval_is_refused(client):
    test_client, root = client
    name = _write(root, "a.py", "original\n")
    change_id = _propose(test_client, root, name, "modified\n").json()["changeId"]

    resp = _apply(test_client, root, change_id)
    assert resp.status_code == 409
    assert resp.json()["detail"]["errorCode"] == "NOT_APPROVED"
    # Nothing was written.
    assert (root / name).read_text(encoding="utf-8") == "original\n"


def test_apply_after_rejection_is_refused(client):
    test_client, root = client
    name = _write(root, "a.py", "original\n")
    change_id = _propose(test_client, root, name, "modified\n").json()["changeId"]

    rejected = test_client.post(
        f"/api/v1/files/changes/{change_id}/reject", json={"projectRoot": str(root)}
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "REJECTED"

    assert _apply(test_client, root, change_id).status_code == 409
    assert _approve(test_client, root, change_id).status_code == 409
    assert (root / name).read_text(encoding="utf-8") == "original\n"


def test_approve_then_apply_writes_and_verifies(client):
    test_client, root = client
    name = _write(root, "a.py", "original\n")
    change_id = _propose(test_client, root, name, "modified\n").json()["changeId"]

    approved = _approve(test_client, root, change_id)
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    # Approving alone must not touch the file.
    assert (root / name).read_text(encoding="utf-8") == "original\n"

    applied = _apply(test_client, root, change_id)
    assert applied.status_code == 200
    body = applied.json()
    assert body["status"] == "APPLIED"
    assert body["verified"] is True
    assert (root / name).read_text(encoding="utf-8") == "modified\n"


def test_duplicate_apply_is_refused(client):
    test_client, root = client
    name = _write(root, "a.py", "original\n")
    change_id = _propose(test_client, root, name, "modified\n").json()["changeId"]
    _approve(test_client, root, change_id)
    assert _apply(test_client, root, change_id).status_code == 200

    second = _apply(test_client, root, change_id)
    assert second.status_code == 409
    assert second.json()["detail"]["errorCode"] == "ALREADY_APPLIED"
    assert (root / name).read_text(encoding="utf-8") == "modified\n"


def test_double_approve_is_refused(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    change_id = _propose(test_client, root, name, "y\n").json()["changeId"]

    assert _approve(test_client, root, change_id).status_code == 200
    second = _approve(test_client, root, change_id)
    assert second.status_code == 409
    assert second.json()["detail"]["errorCode"] == "ALREADY_APPROVED"


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------
def test_file_changed_after_proposal_yields_conflict_and_no_overwrite(client):
    test_client, root = client
    name = _write(root, "a.py", "original\n")
    change_id = _propose(test_client, root, name, "agent version\n").json()["changeId"]
    _approve(test_client, root, change_id)

    # The user edits the file while the proposal awaits review.
    (root / name).write_text("user edited this\n", encoding="utf-8")

    resp = _apply(test_client, root, change_id)
    assert resp.status_code == 409
    assert resp.json()["detail"]["errorCode"] == "FILE_CONFLICT"
    # The user's newer content is preserved, not discarded.
    assert (root / name).read_text(encoding="utf-8") == "user edited this\n"

    state = test_client.get(
        f"/api/v1/files/changes/{change_id}", params={"projectRoot": str(root)}
    ).json()
    assert state["status"] == "CONFLICTED"
    assert state["conflicted"] is True


def test_verify_endpoint_reports_conflict(client):
    test_client, root = client
    name = _write(root, "a.py", "original\n")
    change_id = _propose(test_client, root, name, "agent version\n").json()["changeId"]

    clean = test_client.get(
        f"/api/v1/files/changes/{change_id}/verify", params={"projectRoot": str(root)}
    )
    assert clean.json()["conflicted"] is False

    (root / name).write_text("changed elsewhere\n", encoding="utf-8")
    dirty = test_client.get(
        f"/api/v1/files/changes/{change_id}/verify", params={"projectRoot": str(root)}
    )
    assert dirty.json()["conflicted"] is True


# ---------------------------------------------------------------------------
# Rollback / recovery
# ---------------------------------------------------------------------------
def test_rollback_restores_original_content(client):
    test_client, root = client
    name = _write(root, "a.py", "line1\nline2\n")
    change_id = _propose(test_client, root, name, "replaced\n").json()["changeId"]
    _approve(test_client, root, change_id)
    _apply(test_client, root, change_id)
    assert (root / name).read_text(encoding="utf-8") == "replaced\n"

    resp = test_client.post(
        f"/api/v1/files/changes/{change_id}/rollback", json={"projectRoot": str(root)}
    )
    assert resp.status_code == 200
    assert (root / name).read_text(encoding="utf-8") == "line1\nline2\n"

    # A rolled-back change cannot be applied again.
    assert _apply(test_client, root, change_id).status_code == 409


def test_backup_is_written_before_overwrite(client):
    test_client, root = client
    name = _write(root, "a.py", "precious\n")
    change_id = _propose(test_client, root, name, "new\n").json()["changeId"]
    _approve(test_client, root, change_id)
    _apply(test_client, root, change_id)

    backups = list((root / ".llm_training_agent_backups").glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "precious\n"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
def test_create_operation_creates_file_only_after_approval(client):
    test_client, root = client
    change_id = _propose(
        test_client, root, "new_module.py", "print('hi')\n", operation="create"
    ).json()["changeId"]
    assert not (root / "new_module.py").exists()

    assert _apply(test_client, root, change_id).status_code == 409
    assert not (root / "new_module.py").exists()

    _approve(test_client, root, change_id)
    assert _apply(test_client, root, change_id).status_code == 200
    assert (root / "new_module.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_create_rejects_existing_file(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    resp = _propose(test_client, root, name, "y\n", operation="create")
    assert resp.status_code == 409
    assert resp.json()["detail"]["errorCode"] == "FILE_EXISTS"


def test_delete_operation_removes_file_only_after_approval(client):
    test_client, root = client
    name = _write(root, "obsolete.py", "bye\n")
    change_id = _propose(test_client, root, name, None, operation="delete").json()["changeId"]
    assert (root / name).exists()

    assert _apply(test_client, root, change_id).status_code == 409
    assert (root / name).exists()

    _approve(test_client, root, change_id)
    assert _apply(test_client, root, change_id).status_code == 200
    assert not (root / name).exists()

    # Rollback brings the deleted file back.
    test_client.post(
        f"/api/v1/files/changes/{change_id}/rollback", json={"projectRoot": str(root)}
    )
    assert (root / name).read_text(encoding="utf-8") == "bye\n"


def test_modify_of_missing_file_is_rejected(client):
    test_client, root = client
    resp = _propose(test_client, root, "missing.py", "x\n")
    assert resp.status_code == 404
    assert resp.json()["detail"]["errorCode"] == "FILE_NOT_FOUND"


def test_unknown_operation_is_rejected(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    resp = _propose(test_client, root, name, "y\n", operation="chmod")
    assert resp.status_code == 422
    assert resp.json()["detail"]["errorCode"] == "INVALID_OPERATION"


# ---------------------------------------------------------------------------
# Write safety
# ---------------------------------------------------------------------------
def test_line_endings_are_preserved(client):
    test_client, root = client
    target = root / "crlf.py"
    target.write_bytes(b"one\r\ntwo\r\n")

    change_id = _propose(test_client, root, "crlf.py", "one\ntwo\nthree\n").json()["changeId"]
    _approve(test_client, root, change_id)
    _apply(test_client, root, change_id)

    # The file must stay CRLF, not be rewritten as LF.
    assert target.read_bytes() == b"one\r\ntwo\r\nthree\r\n"


def test_apply_preserves_utf8_bom(client):
    test_client, root = client
    target = root / "bom.py"
    target.write_bytes(b"\xef\xbb\xbfvalue = 1\n")

    change_id = _propose(test_client, root, "bom.py", "value = 2\n").json()["changeId"]
    _approve(test_client, root, change_id)
    _apply(test_client, root, change_id)

    assert target.read_bytes().startswith(b"\xef\xbb\xbf")


def test_no_temporary_files_remain_after_apply(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    change_id = _propose(test_client, root, name, "y\n").json()["changeId"]
    _approve(test_client, root, change_id)
    _apply(test_client, root, change_id)

    leftovers = [p for p in root.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_failed_write_marks_change_failed_and_keeps_original(client, monkeypatch):
    test_client, root = client
    name = _write(root, "a.py", "original\n")
    change_id = _propose(test_client, root, name, "modified\n").json()["changeId"]
    _approve(test_client, root, change_id)

    import editing.file_editor as fe

    def boom(*_args, **_kwargs):
        raise fe.ChangeStoreError("WRITE_FAILED", "The file could not be written.", 500)

    monkeypatch.setattr(fe, "_atomic_write", boom)
    resp = _apply(test_client, root, change_id)
    assert resp.status_code == 500
    assert resp.json()["detail"]["errorCode"] == "WRITE_FAILED"
    # Original content is intact and success was NOT reported.
    assert (root / name).read_text(encoding="utf-8") == "original\n"

    state = test_client.get(
        f"/api/v1/files/changes/{change_id}", params={"projectRoot": str(root)}
    ).json()
    assert state["status"] == "FAILED"


def test_oversized_proposal_is_rejected(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    resp = _propose(test_client, root, name, "x" * 2_000_001)
    assert resp.status_code == 422
    assert (root / name).read_text(encoding="utf-8") == "x\n"


# ---------------------------------------------------------------------------
# Workspace boundaries
# ---------------------------------------------------------------------------
def test_path_traversal_is_rejected(client):
    test_client, root = client
    outside = root.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    resp = _propose(test_client, root, "../outside.txt", "pwned")
    assert resp.status_code in (400, 422)
    assert outside.read_text(encoding="utf-8") == "secret"


def test_absolute_path_outside_workspace_is_rejected(client):
    test_client, root = client
    victim = root.parent / "victim.txt"
    victim.write_text("safe", encoding="utf-8")

    resp = _propose(test_client, root, str(victim), "pwned")
    assert resp.status_code in (400, 422)
    assert victim.read_text(encoding="utf-8") == "safe"


def test_symlink_escape_is_rejected(client):
    test_client, root = client
    outside = root.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks unavailable on this platform")

    resp = _propose(test_client, root, "link.txt", "pwned")
    assert resp.status_code in (400, 403, 422)
    assert outside.read_text(encoding="utf-8") == "secret"


def test_project_root_outside_allowed_roots_is_rejected(client, tmp_path):
    test_client, root = client
    _write(root, "a.py", "x\n")
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    (stranger / "b.py").write_text("y\n", encoding="utf-8")

    resp = _propose(test_client, stranger, "b.py", "z\n")
    assert resp.status_code == 403
    assert (stranger / "b.py").read_text(encoding="utf-8") == "y\n"


def test_missing_project_root_is_rejected(client):
    test_client, root = client
    name = _write(root, "a.py", "x\n")
    resp = test_client.post(
        "/api/v1/files/propose",
        json={"filePath": name, "proposedContent": "y\n"},  # no projectRoot
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["errorCode"] == "WORKSPACE_REQUIRED"


def test_proposal_cannot_cross_workspaces(client, tmp_path, monkeypatch):
    """Project A's proposal must be invisible and unusable from project B."""
    test_client, root_a = client
    name_a = _write(root_a, "a.py", "A original\n")
    change_id = _propose(test_client, root_a, name_a, "A modified\n").json()["changeId"]
    _approve(test_client, root_a, change_id)

    # Project B is a *separately authorized* workspace (both are legitimately
    # open), so the only thing stopping the cross-write is proposal scoping.
    root_b = tmp_path / "project_b"
    root_b.mkdir()
    _write(root_b, "b.py", "B original\n")
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", f"{root_a}{os.pathsep}{root_b}")

    # Not visible from B.
    listed = test_client.get("/api/v1/files/changes", params={"projectRoot": str(root_b)})
    assert listed.status_code == 200
    assert listed.json()["changes"] == []

    # Not retrievable from B.
    assert test_client.get(
        f"/api/v1/files/changes/{change_id}", params={"projectRoot": str(root_b)}
    ).status_code == 404

    # Cannot be applied from B.
    assert test_client.post(
        f"/api/v1/files/changes/{change_id}/apply", json={"projectRoot": str(root_b)}
    ).status_code == 404

    # Neither project was modified by the cross-workspace attempt.
    assert (root_a / name_a).read_text(encoding="utf-8") == "A original\n"
    assert (root_b / "b.py").read_text(encoding="utf-8") == "B original\n"


def test_two_proposals_for_the_same_file_do_not_interfere(client):
    test_client, root = client
    name = _write(root, "a.py", "base\n")

    first = _propose(test_client, root, name, "first\n").json()["changeId"]
    second = _propose(test_client, root, name, "second\n").json()["changeId"]

    _approve(test_client, root, first)
    _approve(test_client, root, second)

    # Applying the first makes the second stale — it must conflict, not clobber.
    assert _apply(test_client, root, first).status_code == 200
    assert (root / name).read_text(encoding="utf-8") == "first\n"

    stale = _apply(test_client, root, second)
    assert stale.status_code == 409
    assert stale.json()["detail"]["errorCode"] == "FILE_CONFLICT"
    assert (root / name).read_text(encoding="utf-8") == "first\n"


def test_unknown_change_id_is_not_found(client):
    test_client, root = client
    _write(root, "a.py", "x\n")
    assert test_client.get(
        "/api/v1/files/changes/does-not-exist", params={"projectRoot": str(root)}
    ).status_code == 404


# ---------------------------------------------------------------------------
# Malicious / malformed agent output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"filePath": "../../../etc/passwd", "proposedContent": "x"},
        {"filePath": "/etc/passwd", "proposedContent": "x"},
        {"filePath": "a.py", "proposedContent": "x", "operation": "run_shell"},
        {"filePath": "", "proposedContent": "x"},
        {"filePath": "a.py", "proposedContent": None},
        {"filePath": "a.py", "proposedContent": 12345},
        {"filePath": "a.py\x00.txt", "proposedContent": "x"},
    ],
)
def test_malicious_agent_proposals_are_rejected(client, payload):
    """Malformed or hostile LLM output must never modify a file."""
    test_client, root = client
    name = _write(root, "a.py", "untouched\n")
    payload.setdefault("projectRoot", str(root))
    if payload.get("filePath") == "a.py":
        payload["filePath"] = name

    resp = test_client.post("/api/v1/files/propose", json=payload)
    assert resp.status_code >= 400
    assert (root / name).read_text(encoding="utf-8") == "untouched\n"


def test_agent_cannot_propose_a_change_outside_the_workspace(client):
    test_client, root = client
    outside = root.parent / "outside.txt"
    outside.write_text("safe\n", encoding="utf-8")

    resp = _propose(test_client, root, "../outside.txt", "pwned")
    assert resp.status_code in (400, 422)
    assert outside.read_text(encoding="utf-8") == "safe\n"


def test_error_responses_do_not_leak_absolute_paths(client):
    test_client, root = client
    _write(root, "a.py", "x\n")
    resp = _propose(test_client, root, "../outside.txt", "pwned")
    assert resp.status_code in (400, 422)
    body = json.dumps(resp.json())
    assert str(root) not in body
