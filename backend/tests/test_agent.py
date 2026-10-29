"""End-to-end tests of the chat agent over HTTP: edits, revert, permissions, cleaning, history."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router

TRAIN = '''from transformers import TrainingArguments

EPOCHS = 3
LR = 2e-4

args = TrainingArguments(output_dir="out", num_train_epochs=EPOCHS, per_device_train_batch_size=4,
                         learning_rate=LR, gradient_accumulation_steps=4)
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "train.py").write_text(TRAIN, encoding="utf-8")
    rows = [{"id": i, "prompt": f"  q{i}  \n\n\n\nmore ", "response": f"a{i}"} for i in range(6)]
    (root / "data" / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("LLM_TRAINING_AGENT_ALLOWED_ROOTS", str(tmp_path))
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    class Env:
        pass

    e = Env()
    e.client, e.root, e.path = client, root, str(root)

    def perms(**flags):
        body = {"readFiles": True, "writeFiles": False, "safeCommands": False, "unrestrictedCommands": False}
        body.update(flags)
        r = client.put("/api/v1/agent/permissions", json={"projectPath": e.path, "permissions": body})
        assert r.status_code == 200
        return r.json()

    def say(text, session=None):
        r = client.post("/api/v1/agent/chat", json={"projectPath": e.path, "sessionId": session, "message": text})
        assert r.status_code == 200, r.text
        return r.json()

    def answer(session, block, choice):
        r = client.post(f"/api/v1/agent/actions/{block['actionId']}/respond",
                        json={"projectPath": e.path, "sessionId": session, "choice": choice})
        return r

    e.perms, e.say, e.answer = perms, say, answer
    return e


def blocks(reply, kind):
    return [b for b in reply["message"]["blocks"] if b.get("type") == kind]


# --------------------------------------------------------------------------- settings edit + diff + revert
def test_change_epochs_with_write_allowed_edits_validates_and_shows_red_green_diff(env):
    env.perms(writeFiles=True)
    reply = env.say("change the number of training epochs from 3 to 5")
    assert "EPOCHS = 5" in (env.root / "train.py").read_text(encoding="utf-8")
    assert "num_train_epochs=EPOCHS" in (env.root / "train.py").read_text(encoding="utf-8")
    (diff,) = blocks(reply, "diff")
    assert diff["filePath"] == "train.py" and diff["status"] == "APPLIED"
    (hunk,) = diff["hunks"]
    kinds = [(l["type"], l["text"]) for l in hunk["lines"] if l["type"] != "context"]
    assert kinds == [("removed", "EPOCHS = 3"), ("added", "EPOCHS = 5")]
    assert diff["additions"] == 1 and diff["deletions"] == 1


def test_without_write_permission_the_file_is_untouched_until_allowed(env):
    env.perms(writeFiles=False)
    reply = env.say("set epochs to 5")
    assert (env.root / "train.py").read_text(encoding="utf-8") == TRAIN, "no write before permission"
    (diff,) = blocks(reply, "diff")
    assert diff["status"] == "PROPOSED"
    (card,) = blocks(reply, "actions")
    assert card["kind"] == "permission" and [b["choice"] for b in card["buttons"]] == ["allow", "deny"]
    session = reply["sessionId"]

    allowed = env.answer(session, card, "allow")
    assert allowed.status_code == 200
    assert "EPOCHS = 5" in (env.root / "train.py").read_text(encoding="utf-8")

    # the same card cannot be answered twice (replay / double click)
    assert env.answer(session, card, "allow").status_code == 409


def test_deny_discards_the_proposal_and_changes_nothing(env):
    reply = env.say("set epochs to 5")
    (card,) = blocks(reply, "actions")
    out = env.answer(reply["sessionId"], card, "deny").json()
    assert (env.root / "train.py").read_text(encoding="utf-8") == TRAIN
    assert "did not change" in out["message"]["content"]


def test_answering_a_card_cannot_authorise_a_different_action(env):
    reply = env.say("set epochs to 5")
    (card,) = blocks(reply, "actions")
    r = env.client.post(f"/api/v1/agent/actions/{card['actionId']}/respond",
                        json={"projectPath": env.path, "sessionId": reply["sessionId"], "choice": "revert"})
    assert r.status_code == 409  # 'revert' is not a valid answer to a permission card
    assert (env.root / "train.py").read_text(encoding="utf-8") == TRAIN
    other = env.client.post(f"/api/v1/agent/actions/{card['actionId']}/respond",
                            json={"projectPath": env.path + "_x", "sessionId": reply["sessionId"], "choice": "allow"})
    assert other.status_code == 409  # another project cannot answer it


def test_revert_restores_the_edit_but_keeps_unrelated_user_changes(env):
    env.perms(writeFiles=True)
    reply = env.say("set epochs to 5")
    (offer,) = blocks(reply, "actions")
    path = env.root / "train.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# my own note\n", encoding="utf-8")

    out = env.answer(reply["sessionId"], offer, "revert").json()

    text = path.read_text(encoding="utf-8")
    assert "EPOCHS = 3" in text and "# my own note" in text
    assert blocks(out, "diff")[0]["status"] == "REVERTED"


def test_revert_by_message_and_refuses_when_the_same_line_was_edited_again(env):
    env.perms(writeFiles=True)
    env.say("set learning rate to 1e-4")
    path = env.root / "train.py"
    assert "LR = 1e-4" in path.read_text(encoding="utf-8")
    path.write_text(path.read_text(encoding="utf-8").replace("LR = 1e-4", "LR = 5e-5"), encoding="utf-8")
    out = env.say("revert the learning rate change")
    assert "did not revert" in out["message"]["content"]
    assert "LR = 5e-5" in path.read_text(encoding="utf-8"), "the user's newer value must survive"
    path.write_text(path.read_text(encoding="utf-8").replace("LR = 5e-5", "LR = 1e-4"), encoding="utf-8")
    out = env.say("undo the last change")
    assert "LR = 2e-4" in path.read_text(encoding="utf-8")


def test_ambiguous_setting_asks_which_file(env):
    (env.root / "train.py").write_text("import yaml\ncfg = yaml.safe_load(open('configs/a.yaml'))\n", encoding="utf-8")
    (env.root / "configs").mkdir()
    (env.root / "configs" / "a.yaml").write_text("training:\n  epochs: 3\n", encoding="utf-8")
    (env.root / "configs" / "b.yaml").write_text("training:\n  epochs: 3\n", encoding="utf-8")
    env.perms(writeFiles=True)
    reply = env.say("set epochs to 5")
    (card,) = blocks(reply, "actions")
    assert card["kind"] == "choose_site"
    labels = [b["label"] for b in card["buttons"]]
    assert any("configs/a.yaml" in l for l in labels) and any("configs/b.yaml" in l for l in labels)
    env.answer(reply["sessionId"], card, "site:1")
    changed = [n for n in ("a", "b") if "epochs: 5" in (env.root / "configs" / f"{n}.yaml").read_text(encoding="utf-8")]
    assert len(changed) == 1, "exactly the chosen file is edited"
    assert env.answer(reply["sessionId"], card, "site:0").status_code == 409


def test_training_script_wins_over_a_config_when_both_hold_the_setting(env):
    (env.root / "config.yaml").write_text("training:\n  epochs: 3\n", encoding="utf-8")
    env.perms(writeFiles=True)
    env.say("set epochs to 5")
    assert "EPOCHS = 5" in (env.root / "train.py").read_text(encoding="utf-8")
    assert "epochs: 3" in (env.root / "config.yaml").read_text(encoding="utf-8")


def test_mismatched_old_value_changes_nothing(env):
    env.perms(writeFiles=True)
    out = env.say("change epochs from 7 to 9")
    assert (env.root / "train.py").read_text(encoding="utf-8") == TRAIN
    assert "found" in out["message"]["content"]


def test_view_changes_posts_inline_viewer_into_chat(env):
    env.perms(writeFiles=True)
    first = env.say("set epochs to 5")
    r = env.client.post("/api/v1/agent/changes/show", json={"projectPath": env.path, "sessionId": first["sessionId"]})
    assert r.status_code == 200
    diffs = [b for b in r.json()["message"]["blocks"] if b["type"] == "diff"]
    assert diffs and diffs[0]["filePath"] == "train.py" and diffs[0]["hunks"]
    blank = env.root.parent / "blank"
    blank.mkdir()
    empty = env.client.post("/api/v1/agent/changes/show", json={"projectPath": str(blank), "sessionId": None})
    assert "no changes" in empty.json()["message"]["content"].lower()


# --------------------------------------------------------------------------- permissions & terminal
def test_read_permission_off_asks_before_looking_at_files(env):
    env.perms(readFiles=False, writeFiles=True)
    reply = env.say("set epochs to 5")
    (card,) = blocks(reply, "actions")
    assert card["kind"] == "permission" and "Reading project files" in card["text"]
    assert (env.root / "train.py").read_text(encoding="utf-8") == TRAIN
    env.answer(reply["sessionId"], card, "allow")
    assert "EPOCHS = 5" in (env.root / "train.py").read_text(encoding="utf-8")


def test_safe_command_needs_the_safe_permission(env):
    env.perms(safeCommands=False)
    reply = env.say("run `ls`")
    (card,) = blocks(reply, "actions")
    assert card["kind"] == "permission"
    out = env.answer(reply["sessionId"], card, "allow").json()
    (cmd,) = blocks(out, "command")
    assert cmd["exitCode"] == 0 and "train.py" in cmd["output"] and cmd["tier"] == "safe"

    env.perms(safeCommands=True)
    direct = env.say("run `ls`")
    assert blocks(direct, "command") and not blocks(direct, "actions"), "permitted safe commands run without asking"


def test_risky_command_always_asks_unless_unrestricted(env):
    env.perms(safeCommands=True)
    reply = env.say("run `python --version && echo hi`")
    (card,) = blocks(reply, "actions")
    assert card["kind"] == "permission" and "unrestricted" in card["text"].lower()
    denied = env.answer(reply["sessionId"], card, "deny").json()
    assert not blocks(denied, "command")

    env.perms(safeCommands=True, unrestrictedCommands=True)
    ok = env.say("run `echo hi`")
    assert blocks(ok, "command")[0]["output"].strip() == "hi"


def test_catastrophic_commands_are_never_run_even_when_unrestricted(env):
    env.perms(safeCommands=True, unrestrictedCommands=True)
    victim = env.root / "keep.txt"
    victim.write_text("x", encoding="utf-8")
    out = env.say("run `rm -rf /`")
    assert not blocks(out, "command") and not blocks(out, "actions")
    assert "will not run" in out["message"]["content"]
    assert victim.exists()


def test_command_output_is_redacted_and_environment_secrets_are_not_inherited(env, monkeypatch):
    monkeypatch.setenv("MY_SECRET_TOKEN", "sk-supersecretvalue1234567890")
    env.perms(unrestrictedCommands=True, safeCommands=True)
    out = env.say("run `env`")
    text = json.dumps(out["message"])
    assert "supersecretvalue" not in text


def test_permissions_roundtrip_and_defaults(env):
    default = env.client.get("/api/v1/agent/permissions", params={"projectPath": env.path + "_fresh"}).json()
    assert default == {"readFiles": True, "writeFiles": False, "safeCommands": False, "unrestrictedCommands": False}
    env.perms(writeFiles=True)
    assert env.client.get("/api/v1/agent/permissions", params={"projectPath": env.path}).json()["writeFiles"] is True
    r = env.client.put("/api/v1/agent/permissions", json={"projectPath": env.path, "permissions": {"writeFiles": "yes"}})
    assert r.json()["writeFiles"] is False, "only a real true enables a permission"


def test_direct_file_routes_enforce_the_write_permission(env):
    env.perms(writeFiles=False)
    proposal = env.client.post("/api/v1/files/propose", json={
        "projectRoot": env.path, "filePath": "train.py", "proposedContent": "x = 1\n"}).json()
    cid = proposal["changeId"]
    env.client.post(f"/api/v1/files/changes/{cid}/approve", json={"projectRoot": env.path})
    r = env.client.post(f"/api/v1/files/changes/{cid}/apply", json={"projectRoot": env.path})
    assert r.status_code == 403 and r.json()["detail"]["errorCode"] == "PERMISSION_REQUIRED"
    assert (env.root / "train.py").read_text(encoding="utf-8") == TRAIN
    assert env.client.post("/api/v1/dataset/clean", json={"projectPath": env.path, "useLlm": False}).status_code == 403


# --------------------------------------------------------------------------- cleaning
def test_cleaning_always_asks_method_and_output_then_merge_works_from_another_chat(env):
    env.perms(writeFiles=True)
    original = (env.root / "data" / "train.jsonl").read_bytes()

    first = env.say("please clean my data")
    (method,) = blocks(first, "actions")
    assert method["kind"] == "clean_method"
    assert [b["choice"] for b in method["buttons"]][:2] == ["ai", "script"]
    session = first["sessionId"]

    step2 = env.answer(session, method, "script").json()
    (where,) = blocks(step2, "actions")
    assert where["kind"] == "clean_output", "the output question must always be asked"
    assert {b["choice"] for b in where["buttons"]} >= {"separate", "in_place"}
    assert (env.root / "data" / "train.jsonl").read_bytes() == original

    done = env.answer(session, where, "separate").json()
    assert (env.root / "data" / "train.jsonl").read_bytes() == original, "originals untouched in separate mode"
    assert "merge the cleaned data" in done["message"]["content"]
    cleaned_dirs = list((env.root / ".llm-training-agent" / "cleaned").iterdir())
    assert len(cleaned_dirs) == 1

    # a brand-new chat (as after restarting VS Code) can still merge it
    merged = env.say("merge the cleaned data into the original files")
    assert merged["sessionId"] != session
    assert "Merged 1 file" in merged["message"]["content"]
    assert "temporary cleaned-data folder was removed" in merged["message"]["content"]
    assert json.loads((env.root / "data" / "train.jsonl").read_text().splitlines()[0])["prompt"] == "q0\n\nmore"
    assert not (env.root / ".llm-training-agent" / "cleaned").exists()

    again = env.say("merge the cleaned data")
    assert "no cleaned data waiting" in again["message"]["content"]


def test_cleaning_in_place_replaces_files_and_keeps_backup(env):
    env.perms(writeFiles=True)
    first = env.say("clean the dataset")
    s = first["sessionId"]
    (m,) = blocks(first, "actions")
    (w,) = blocks(env.answer(s, m, "script").json(), "actions")
    done = env.answer(s, w, "in_place").json()
    assert "updated" in done["message"]["content"]
    assert json.loads((env.root / "data" / "train.jsonl").read_text().splitlines()[0])["prompt"] == "q0\n\nmore"
    backups = list((env.root / ".llm_training_agent_backups" / "cleaning").rglob("train.jsonl"))
    assert len(backups) == 1


def test_cleaning_without_write_permission_asks_first(env):
    env.perms(writeFiles=False)
    first = env.say("clean my data")
    s = first["sessionId"]
    (m,) = blocks(first, "actions")
    (w,) = blocks(env.answer(s, m, "script").json(), "actions")
    ask = env.answer(s, w, "separate").json()
    (card,) = blocks(ask, "actions")
    assert card["kind"] == "permission"
    assert not (env.root / ".llm-training-agent").exists()
    env.answer(s, card, "allow")
    assert (env.root / ".llm-training-agent" / "cleaned").exists()


# --------------------------------------------------------------------------- chat history
def test_new_chat_creates_a_session_with_title_and_history_persists(env):
    env.perms(writeFiles=True)
    first = env.say("change learning rate to 1e-4")
    sid = first["sessionId"]
    assert first["title"] == "Change learning rate"

    listing = env.client.get("/api/v1/chat/sessions", params={"projectPath": env.path}).json()["sessions"]
    assert [s["sessionId"] for s in listing] == [sid]
    assert listing[0]["title"] == "Change learning rate" and listing[0]["createdAt"] and listing[0]["messageCount"] == 2

    second = env.say("show my permissions")
    assert second["sessionId"] != sid
    full = env.client.get(f"/api/v1/chat/sessions/{sid}", params={"projectPath": env.path}).json()
    assert [m["role"] for m in full["messages"]] == ["user", "assistant"]
    assert any(b["type"] == "diff" for m in full["messages"] for b in m["blocks"])

    listing = env.client.get("/api/v1/chat/sessions", params={"projectPath": env.path}).json()["sessions"]
    assert len(listing) == 2 and listing[0]["sessionId"] == second["sessionId"], "newest activity first"


def test_resolved_cards_are_restored_as_resolved(env):
    first = env.say("clean my data")
    (m,) = blocks(first, "actions")
    env.answer(first["sessionId"], m, "script")
    full = env.client.get(f"/api/v1/chat/sessions/{first['sessionId']}", params={"projectPath": env.path}).json()
    card = [b for msg in full["messages"] for b in msg["blocks"] if b.get("actionId") == m["actionId"]][0]
    assert card["resolved"]["choice"] == "script"


def test_sessions_are_scoped_to_their_project(env, tmp_path):
    reply = env.say("set epochs to 5")
    other = tmp_path / "other"
    other.mkdir()
    assert env.client.get("/api/v1/chat/sessions", params={"projectPath": str(other)}).json()["sessions"] == []
    r = env.client.get(f"/api/v1/chat/sessions/{reply['sessionId']}", params={"projectPath": str(other)})
    assert r.status_code == 404
    r = env.client.post("/api/v1/agent/chat", json={"projectPath": str(other), "sessionId": reply["sessionId"], "message": "hi"})
    assert r.status_code == 404


def test_delete_session(env):
    reply = env.say("show my permissions")
    assert env.client.delete(f"/api/v1/chat/sessions/{reply['sessionId']}", params={"projectPath": env.path}).status_code == 200
    assert env.client.get("/api/v1/chat/sessions", params={"projectPath": env.path}).json()["sessions"] == []


def test_invalid_input_is_rejected(env):
    assert env.client.post("/api/v1/agent/chat", json={"projectPath": env.path, "message": "  "}).status_code == 422
    assert env.client.get("/api/v1/chat/sessions/../x", params={"projectPath": env.path}).status_code in (404, 422)
    assert env.client.post("/api/v1/agent/actions/nope/respond",
                           json={"projectPath": env.path, "sessionId": "abc", "choice": "allow"}).status_code == 409


def test_ordinary_questions_go_to_the_language_model_and_do_not_edit_files(env, monkeypatch):
    from chat.engine import ChatEngine

    async def fake_ask(self, context, report, recommendations, question, chat_history=None, snapshot=None):
        return {"assistantResponse": "Epochs are full passes over the data.", "references": [],
                "confidence": "high", "model": "fake"}

    monkeypatch.setattr(ChatEngine, "ask", fake_ask)
    env.perms(writeFiles=True)
    out = env.say("what are epochs?")
    assert out["message"]["content"] == "Epochs are full passes over the data."
    assert (env.root / "train.py").read_text(encoding="utf-8") == TRAIN


def test_hardware_statement_is_remembered_and_used_by_the_next_analysis(env):
    out = env.say("I train on a RunPod A100 at $2.00/hr")
    assert "cloud GPU" in out["message"]["content"]
    analysis = env.client.post("/api/v1/project/analyze", json={"projectPath": env.path}).json()["analysis"]
    cost = analysis["project_findings"]["cost"]
    assert cost and cost["hourly_rate_usd"] == 2.0


def test_allow_once_grants_exactly_one_request(env):
    env.perms(readFiles=False)
    body = {"projectPath": env.path}
    denied = env.client.post("/api/v1/project/analyze", json=body)
    assert denied.status_code == 403 and denied.json()["detail"]["errorCode"] == "PERMISSION_REQUIRED"
    assert env.client.post("/api/v1/agent/permissions/allow-once",
                           json={"projectPath": env.path, "action": "read"}).status_code == 200
    assert env.client.post("/api/v1/project/analyze", json=body).status_code == 200
    assert env.client.post("/api/v1/project/analyze", json=body).status_code == 403  # consumed


def test_allow_once_cannot_grant_commands(env):
    r = env.client.post("/api/v1/agent/permissions/allow-once",
                        json={"projectPath": env.path, "action": "command_risky"})
    assert r.status_code == 422


def test_extension_notes_are_persisted_and_create_the_chat(env):
    r = env.client.post("/api/v1/agent/chat/note", json={
        "projectPath": env.path, "role": "assistant", "title": "Project analysis",
        "content": "# Analysis\nsome text",
        "blocks": [{"type": "commandButtons", "buttons": []}, {"type": "actions", "actionId": "x"}]})
    assert r.status_code == 200
    sid = r.json()["sessionId"]
    chat = env.client.get(f"/api/v1/chat/sessions/{sid}", params={"projectPath": env.path}).json()
    assert chat["title"] == "Project analysis"
    msg = chat["messages"][0]
    assert msg["content"].startswith("# Analysis")
    assert [b["type"] for b in msg["blocks"]] == ["commandButtons"]  # client cannot forge action cards
    bad = env.client.post("/api/v1/agent/chat/note", json={"projectPath": env.path, "content": ""})
    assert bad.status_code == 422
