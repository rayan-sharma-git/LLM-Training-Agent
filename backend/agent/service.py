"""The chat agent: turns a message into an action, enforcing permissions on the way.

``handle_message`` routes a user message (see :mod:`agent.intents`); anything the
agent cannot do itself goes to the language model.  Every file or command
operation passes through :mod:`core.permissions`.  When a permission is missing
the agent does **not** fail silently: it stores the pending step server-side and
asks in chat with Allow / Deny buttons; Allow authorises exactly that stored
step, once.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent import intents, pending
from agent import settings_editor as editor
from agent.terminal import classify_command, run_command
from analysis import hardware_store
from analysis.code_inspector import inspect_project
from chat import sessions
from cleaning.dataset_cleaner import DatasetCleaner
from cleaning.jobs import CleaningJobError, CleaningJobs
from core import permissions as perms
from core.permissions_store import load_policy
from core.redaction import scrub
from core.security import resolve_workspace_root
from editing.diff_view import build_hunks
from editing.file_editor import ChangeStore, ChangeStoreError

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 8000


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------
def diff_block(view: Dict[str, Any], note: Optional[str] = None) -> Dict[str, Any]:
    return {
        "type": "diff",
        "changeId": view["changeId"],
        "filePath": view["filePath"],
        "operation": view.get("operation", "modify"),
        "status": view.get("status"),
        "additions": view.get("additions", 0),
        "deletions": view.get("deletions", 0),
        "hunks": view.get("hunks", []),
        "truncated": bool(view.get("truncated")),
        "note": note,
    }


def command_block(result: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "command", **{k: result[k] for k in
                                  ("command", "tier", "exitCode", "timedOut", "output", "truncated", "durationMs")}}


def buttons_block(buttons: List[Dict[str, str]]) -> Dict[str, Any]:
    """Buttons that run a VS Code command (for example the Analyze Project command)."""
    return {"type": "commandButtons", "buttons": buttons}


async def _actions_block(
    project_path: str, session_id: str, kind: str, payload: Dict[str, Any],
    title: str, buttons: List[Dict[str, str]], text: str = "",
) -> Dict[str, Any]:
    action_id = await pending.create(project_path, session_id, kind, payload)
    return {"type": "actions", "actionId": action_id, "kind": kind, "title": title, "text": text,
            "buttons": buttons, "resolved": None}


class Reply:
    """What the agent says: markdown text plus rich blocks."""

    def __init__(self, text: str = "", blocks: Optional[List[Dict[str, Any]]] = None, error: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.text = text
        self.blocks = blocks or []
        self.error = error
        self.meta = meta or {}


# ---------------------------------------------------------------------------
# Permission-aware execution
# ---------------------------------------------------------------------------
class AgentContext:
    def __init__(self, project_path: str, root: Optional[Path], session_id: str,
                 policy: perms.PermissionPolicy, approved: Set[str]):
        self.project_path = project_path
        self.root = root
        self.session_id = session_id
        self.policy = policy
        self.approved = approved

    def need(self, action: str, detail: str = "") -> None:
        """Raise :class:`PermissionRequired` unless allowed by policy or approved for this step."""
        perms.enforce(self.policy, action, approved=action in self.approved, detail=detail)

    def store(self) -> ChangeStore:
        if self.root is None:
            raise ChangeStoreError("WORKSPACE_REQUIRED", "Open a project folder first.", 422)
        return ChangeStore(self.root)


async def _permission_reply(ctx: AgentContext, error: perms.PermissionRequired, descriptor: Dict[str, Any],
                            leading: Optional[Reply] = None) -> Reply:
    """Ask the user to allow the blocked action (Allow / Deny)."""
    descriptor = {**descriptor, "approved": sorted(ctx.approved), "needs": error.action}
    block = await _actions_block(
        ctx.project_path, ctx.session_id, "permission", descriptor,
        title="Permission needed",
        text=error.message,
        buttons=[{"label": "Allow", "choice": "allow", "style": "primary"},
                 {"label": "Deny", "choice": "deny", "style": "secondary"}],
    )
    blocks = (leading.blocks if leading else []) + [block]
    text = leading.text if leading and leading.text else "I need your permission before I continue."
    return Reply(text, blocks)


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------
async def _do_set_setting(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    ctx.need(perms.READ)
    if ctx.root is None:
        return Reply("Open a project folder first, then I can change its settings.")
    concept = editor.concept_by_key(d["concept"])
    if concept is None:
        return Reply("I do not know that setting.")
    change = editor.SettingChange(concept, d.get("op", "set"), d["operand"], d.get("old"), d.get("fileHint"))
    only = tuple(d["only"]) if d.get("only") else None
    scripts = [s["path"] for s in inspect_project(str(ctx.root)).to_dict().get("training_scripts", [])]
    plan = editor.plan_edit(ctx.root, change, training_scripts=scripts, only=only)  # type: ignore[arg-type]

    if plan.status == "ambiguous":
        buttons = [{"label": f"{c.where()} ({c.literal})", "choice": f"site:{i}", "style": "primary" if i == 0 else "secondary"}
                   for i, c in enumerate(plan.candidates)]
        buttons.append({"label": "Cancel", "choice": "cancel", "style": "secondary"})
        block = await _actions_block(
            ctx.project_path, ctx.session_id, "choose_site",
            {"intent": intents.SET_SETTING, **d, "sites": [[c.file, c.line] for c in plan.candidates],
             "approved": sorted(ctx.approved)},
            "Which one?", buttons, plan.message)
        return Reply(plan.message, [block])
    if not plan.ready:
        return Reply(plan.message)

    store = ctx.store()
    reason = f"Set {concept.label}: {plan.old_literal} → {plan.new_literal} ({plan.site.where()})"
    try:
        record = store.propose(plan.site.file, plan.new_text, reason=reason, source="agent")
    except ChangeStoreError as error:
        return Reply(f"I could not prepare that change: {error.message}")
    return await _apply_proposal(ctx, record["changeId"], reason)


async def _apply_proposal(ctx: AgentContext, change_id: str, reason: str) -> Reply:
    """Apply a proposed change if writing is permitted, otherwise ask (showing the diff first)."""
    store = ctx.store()
    descriptor = {"intent": "apply_change", "changeId": change_id, "reason": reason}
    try:
        ctx.need(perms.WRITE)
    except perms.PermissionRequired as error:
        view = store.view(change_id)
        lead = Reply(f"Proposed change - **{reason}**", [diff_block(view, "Not applied yet")])
        return await _permission_reply(ctx, error, descriptor, lead)
    try:
        store.approve(change_id)
        store.apply(change_id)
    except ChangeStoreError as error:
        if error.code == "FILE_CONFLICT":
            return Reply("The file changed after I prepared this edit, so I did not write it. Ask me again.")
        return Reply(f"I could not apply the change: {error.message}")
    view = store.view(change_id)
    revert = await _actions_block(
        ctx.project_path, ctx.session_id, "revert_offer", {"intent": intents.REVERT, "changeId": change_id},
        "", [{"label": "Revert", "choice": "revert", "style": "secondary"}])
    return Reply(f"Done - **{reason}**. The file was re-parsed after the edit and still validates.",
                 [diff_block(view, "Applied"), revert])


async def _do_apply_change(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    return await _apply_proposal(ctx, d["changeId"], d.get("reason", "Change"))


async def _do_revert(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    if ctx.root is None:
        return Reply("Open a project folder first.")
    store = ctx.store()
    change_id = d.get("changeId")
    if not change_id:
        record = _find_revertible(store, d.get("concept"), d.get("file"))
        if record is None:
            return Reply("I have no applied change to revert" +
                         (" matching that." if d.get("concept") or d.get("file") else "."))
        change_id = record["changeId"]
    try:
        ctx.need(perms.WRITE)
    except perms.PermissionRequired as error:
        return await _permission_reply(ctx, error, {"intent": intents.REVERT, "changeId": change_id})
    record = store.get(change_id)
    if record is None:
        return Reply("That change no longer exists.")
    try:
        store.revert(change_id)
    except ChangeStoreError as error:
        details = (error.details or [])
        extra = ""
        if isinstance(details, list) and details:
            extra = " Edited again: " + ", ".join(f"line {x.get('line')}" for x in details[:3]) + "."
        return Reply(f"I did not revert it. {error.message}{extra}")
    # Show what the revert did (the reverse of the original edit).
    view = build_hunks(record.get("proposedContent") or "", record.get("originalContent") or "")
    shown = {"changeId": change_id, "filePath": record["filePath"], "operation": record.get("operation"),
             "status": "REVERTED", **view}
    return Reply(f"Reverted: {record.get('reason') or record['filePath']}. Your other edits to the file were kept.",
                 [diff_block(shown, "Reverted")])


def _find_revertible(store: ChangeStore, concept: Optional[str], file: Optional[str]) -> Optional[Dict[str, Any]]:
    label = editor.concept_by_key(concept).label if concept and editor.concept_by_key(concept) else None
    records = [r for r in store._all_records() if r.get("status") == "APPLIED" and r.get("source") == "agent"]
    records.sort(key=lambda r: r.get("appliedAt", ""), reverse=True)
    for record in records:
        if label and label.lower() not in (record.get("reason") or "").lower():
            continue
        if file and not (record["filePath"] == file or record["filePath"].endswith("/" + file)):
            continue
        return record
    return None


async def _do_show_changes(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    if ctx.root is None:
        return Reply("Open a project folder first.")
    store = ctx.store()
    records = [r for r in store.list() if r.get("status") in ("APPLIED", "PROPOSED", "APPROVED", "REVERTED")][:5]
    if not records:
        return Reply("There are no changes yet. Ask me to change a setting (for example \"set epochs to 5\") "
                     "and the edit shows up here as a red/green diff.")
    blocks = []
    for record in records:
        view = store.view(record["changeId"])
        blocks.append(diff_block(view, record.get("status", "").title()))
    return Reply(f"Latest {len(blocks)} change(s):", blocks)


async def _do_run_command(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    if ctx.root is None:
        return Reply("Open a project folder first, then I can run commands in it.")
    command = d["command"]
    verdict = classify_command(command, ctx.root)
    if verdict.tier == "blocked":
        return Reply(f"I will not run that: {verdict.reason}. This command is blocked regardless of "
                     "permissions because it could destroy data.")
    try:
        ctx.need(verdict.action, detail=f"(`{scrub(command)[:120]}`)")
    except perms.PermissionRequired as error:
        return await _permission_reply(ctx, error, {"intent": intents.RUN_COMMAND, "command": command},
                                       Reply(f"I would like to run `{scrub(command)[:200]}` ({verdict.reason})."))
    try:
        result = run_command(command, ctx.root, ctx.policy, approved=verdict.action in ctx.approved)
    except perms.PermissionRequired as error:  # defensive: classification changed between calls
        return await _permission_reply(ctx, error, {"intent": intents.RUN_COMMAND, "command": command})
    except perms.PermissionBlocked as error:
        return Reply(error.message)
    status = "timed out" if result["timedOut"] else f"exit code {result['exitCode']}"
    return Reply(f"Ran `{result['command'][:200]}` - {status}.", [command_block(result)])


async def _do_hardware(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    profile = d["profile"]
    from analysis.project_findings import HardwareProfile
    if isinstance(profile, dict):
        profile = HardwareProfile.from_dict(profile)
    if not await hardware_store.save_profile(ctx.project_path, profile):
        return Reply("I could not save that hardware setup. It will not be used for the next analysis.")
    bits = []
    if profile.mode == "cloud":
        bits.append("cloud GPU" + (f" ({profile.provider})" if profile.provider else ""))
    elif profile.mode == "local":
        bits.append("local GPU")
    if profile.gpu_name:
        bits.append(profile.gpu_name)
    if profile.hourly_rate_usd:
        bits.append(f"${profile.hourly_rate_usd:g}/hour")
    text = "Noted: " + ", ".join(bits) + ". I will use this for training time" + \
           (" and cost" if profile.mode == "cloud" or profile.hourly_rate_usd else "") + \
           " the next time you analyze the project."
    if profile.mode == "local":
        text += " No rental cost is shown for a local GPU."
    return Reply(text, [buttons_block([{"label": "Analyze Project", "command": "llmTrainingAgent.analyzeProject"}])])


async def _do_analyze(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    return Reply("Running the analysis needs the backend checks the Analyze Project command performs. "
                 "Use the button below.",
                 [buttons_block([{"label": "Analyze Project", "command": "llmTrainingAgent.analyzeProject"}])])


async def _do_permissions(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    p = ctx.policy

    def mark(flag: bool) -> str:
        return "on" if flag else "off"

    return Reply(
        "Current permissions for this project:\n\n"
        f"- Read project files: **{mark(p.read_files)}**\n"
        f"- Write project files: **{mark(p.write_files)}**\n"
        f"- Run safe terminal commands: **{mark(p.safe_commands)}**\n"
        f"- Run unrestricted terminal commands: **{mark(p.unrestricted_commands)}**\n\n"
        "Change them in the Settings view → Permissions. When something is off I ask you with Allow / Deny.")


# --- cleaning --------------------------------------------------------------------
def _cleaning_context(ctx: AgentContext):
    from scanner.context_builder import ContextBuilder
    from scanner.scanner import ProjectScanner
    return ContextBuilder().build(ProjectScanner(str(ctx.root)).scan())


async def _do_clean(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    """Cleaning wizard: method → where to write → run. The output question is always asked."""
    if ctx.root is None:
        return Reply("Open a project folder first.")
    ctx.need(perms.READ)
    method, mode = d.get("method"), d.get("outputMode")
    if method is None:
        context = _cleaning_context(ctx)
        cleaner = DatasetCleaner()
        files, _ = cleaner.discover(context)
        store_dir = (ctx.root / ".llm-training-agent").resolve()
        files = [f for f in files if store_dir not in f.resolve().parents]
        if not files:
            return Reply("I found no dataset files (.jsonl, .json, .csv, .tsv, .txt) in this project to clean.")
        names = ", ".join(f"`{f.resolve().relative_to(ctx.root.resolve()).as_posix()}`" for f in files[:4])
        more = f" and {len(files) - 4} more" if len(files) > 4 else ""
        ai_note = "" if cleaner.llm_available else " (no AI provider is configured, so AI-based cleaning would fall back to script-based)"
        block = await _actions_block(
            ctx.project_path, ctx.session_id, "clean_method", {"intent": intents.CLEAN, "approved": sorted(ctx.approved)},
            "How should I clean it?",
            [{"label": "AI-based cleaning", "choice": "ai", "style": "primary"},
             {"label": "Script-based cleaning", "choice": "script", "style": "secondary"},
             {"label": "Cancel", "choice": "cancel", "style": "secondary"}],
            "AI-based sends the records to your AI provider in small chunks to fix formatting and obvious errors. "
            f"Script-based applies fixed rules (whitespace, control characters) locally.{ai_note}")
        return Reply(f"I found {len(files)} dataset file(s): {names}{more}.", [block])
    if mode is None:
        block = await _actions_block(
            ctx.project_path, ctx.session_id, "clean_output",
            {"intent": intents.CLEAN, "method": method, "approved": sorted(ctx.approved)},
            "Where should the cleaned data go?",
            [{"label": "Create separate output files", "choice": "separate", "style": "primary"},
             {"label": "Modify existing files", "choice": "in_place", "style": "secondary"},
             {"label": "Cancel", "choice": "cancel", "style": "secondary"}],
            "Separate files keep your originals untouched; you can ask me later to merge them into the originals. "
            "Modifying existing files replaces your data files right away (a backup is kept).")
        return Reply("", [block])
    try:
        ctx.need(perms.WRITE)
    except perms.PermissionRequired as error:
        return await _permission_reply(ctx, error, {"intent": intents.CLEAN, "method": method, "outputMode": mode})
    context = _cleaning_context(ctx)
    try:
        job = await CleaningJobs(ctx.root).run(context, DatasetCleaner(), method, mode)
    except CleaningJobError as error:
        return Reply(error.message)
    except Exception as error:  # noqa: BLE001
        logger.error("Cleaning failed: %s", error, exc_info=True)
        return Reply("Cleaning failed unexpectedly. Your files were not changed. Check the backend log for details.")
    lines = [job["summary"]]
    if job["outputMode"] == "separate":
        lines += [f"\nCleaned copies are in `{job['outputDirectory']}`. Your original files are unchanged.",
                  "Tell me **\"merge the cleaned data into the original files\"** whenever you want - that still works "
                  "after you restart VS Code or open another chat."]
    else:
        report = job.get("mergeReport", {})
        lines += [f"\nYour data files were updated ({len(report.get('merged', []))} file(s)); "
                  f"backups are in `{report.get('backupDirectory')}`."]
        for skipped in report.get("skipped", []):
            lines.append(f"- Skipped `{skipped['file']}`: {skipped['reason']}")
    failures = job.get("failureSummary")
    if failures:
        lines.append(f"\n**Skipped by the AI:** {failures['chunksFailed']} chunk(s), "
                     f"{failures['recordsKeptOriginal']} record(s) kept unchanged.")
    other = [w for w in job.get("warnings", []) if "chunk(s)" not in w][:3]
    lines += [f"- {w}" for w in other]
    return Reply("\n".join(lines))


async def _do_merge(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    if ctx.root is None:
        return Reply("Open a project folder first.")
    jobs = CleaningJobs(ctx.root)
    job_id = d.get("jobId")
    if not job_id:
        pending_jobs = jobs.pending_merges()
        if not pending_jobs:
            return Reply("There is no cleaned data waiting to be merged in this project.")
        if len(pending_jobs) > 1:
            buttons = [{"label": f"{j['createdAt'][:16].replace('T', ' ')} - {len(j['files'])} file(s)",
                        "choice": f"job:{j['jobId']}", "style": "primary" if i == 0 else "secondary"}
                       for i, j in enumerate(pending_jobs[:4])]
            buttons.append({"label": "Cancel", "choice": "cancel", "style": "secondary"})
            block = await _actions_block(ctx.project_path, ctx.session_id, "choose_job",
                                         {"intent": intents.MERGE, "approved": sorted(ctx.approved)},
                                         "Which cleaning run?", buttons,
                                         "More than one cleaned dataset is waiting to be merged.")
            return Reply("", [block])
        job_id = pending_jobs[0]["jobId"]
    try:
        ctx.need(perms.WRITE)
    except perms.PermissionRequired as error:
        return await _permission_reply(ctx, error, {"intent": intents.MERGE, "jobId": job_id})
    try:
        report = jobs.merge(job_id)
    except CleaningJobError as error:
        return Reply(error.message)
    lines = []
    if report["merged"]:
        lines.append(f"Merged {len(report['merged'])} file(s) into the originals: "
                     + ", ".join(f"`{f}`" for f in report["merged"][:6]) + ". Each file was re-read and verified.")
        lines.append(f"Backups of the originals: `{report['backupDirectory']}`.")
    if report["removedTemp"]:
        lines.append("The temporary cleaned-data folder was removed.")
    for skipped in report["skipped"]:
        lines.append(f"- Not merged `{skipped['file']}`: {skipped['reason']}.")
    if report["skipped"]:
        lines.append("The remaining cleaned copies were kept so you can resolve this and ask again.")
    return Reply("\n".join(lines) or "Nothing was merged.")


async def _do_restore(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    if ctx.root is None:
        return Reply("Open a project folder first.")
    jobs = CleaningJobs(ctx.root)
    merged = [j for j in jobs.list() if j.get("status") in ("merged", "partially_merged")]
    if not merged:
        return Reply("There is no merged cleaning run to restore.")
    try:
        ctx.need(perms.WRITE)
    except perms.PermissionRequired as error:
        return await _permission_reply(ctx, error, {"intent": intents.RESTORE})
    try:
        report = jobs.restore(merged[0]["jobId"])
    except CleaningJobError as error:
        return Reply(error.message)
    lines = [f"Restored {len(report['restored'])} file(s) from the backup." if report["restored"]
             else "Nothing was restored."]
    lines += [f"- Left `{s['file']}` alone: {s['reason']}." for s in report["skipped"]]
    return Reply("\n".join(lines))


async def _do_discard(ctx: AgentContext, d: Dict[str, Any]) -> Reply:
    if ctx.root is None:
        return Reply("Open a project folder first.")
    jobs = CleaningJobs(ctx.root)
    pending_jobs = jobs.pending_merges()
    if not pending_jobs:
        return Reply("There are no cleaned copies waiting to be merged.")
    try:
        ctx.need(perms.WRITE)
    except perms.PermissionRequired as error:
        return await _permission_reply(ctx, error, {"intent": intents.DISCARD})
    for job in pending_jobs:
        jobs.discard(job["jobId"])
    return Reply(f"Removed {len(pending_jobs)} cleaned copy set(s). Your original files were not touched.")


HANDLERS = {
    intents.SET_SETTING: _do_set_setting,
    "apply_change": _do_apply_change,
    intents.REVERT: _do_revert,
    intents.SHOW_CHANGES: _do_show_changes,
    intents.RUN_COMMAND: _do_run_command,
    intents.HARDWARE: _do_hardware,
    intents.ANALYZE: _do_analyze,
    intents.SHOW_PERMISSIONS: _do_permissions,
    intents.CLEAN: _do_clean,
    intents.MERGE: _do_merge,
    intents.RESTORE: _do_restore,
    intents.DISCARD: _do_discard,
}


async def _dispatch(ctx: AgentContext, descriptor: Dict[str, Any]) -> Reply:
    handler = HANDLERS.get(descriptor.get("intent"))
    if handler is None:
        return Reply("I do not know how to do that.")
    try:
        return await handler(ctx, descriptor)
    except perms.PermissionRequired as error:
        return await _permission_reply(ctx, error, descriptor)
    except perms.PermissionBlocked as error:
        return Reply(error.message)
    except ChangeStoreError as error:
        return Reply(error.message)
    except Exception as error:  # noqa: BLE001
        logger.error("Agent action failed: %s", error, exc_info=True)
        return Reply("That did not work and nothing was changed. Check the backend log for details.")


def _descriptor_for(intent: intents.Intent) -> Optional[Dict[str, Any]]:
    if intent.kind == intents.SET_SETTING:
        change: editor.SettingChange = intent.data["change"]
        return {"intent": intents.SET_SETTING, "concept": change.concept.key, "op": change.op,
                "operand": change.operand, "old": change.old, "fileHint": change.file_hint}
    if intent.kind == intents.HARDWARE:
        return {"intent": intents.HARDWARE, "profile": intent.data["profile"].to_dict()}
    if intent.kind in HANDLERS:
        return {"intent": intent.kind, **{k: v for k, v in intent.data.items()}}
    return None


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
async def _llm_reply(project_path: str, session_id: str, text: str, provider: Optional[str]) -> Reply:
    from ai.providers import get_provider  # noqa: F401  (kept lazy: provider import is heavy)
    from chat.engine import ChatEngine
    from chat.store import store as analysis_store
    from core.errors import AppError
    from api.routes import _snapshot_models
    from models.schemas import ChatMessage

    history = [ChatMessage(role=t["role"], content=t["content"])
               for t in (await sessions.history_for_llm(project_path, session_id))[:-1]]
    snapshot = analysis_store.get_analysis(project_path)
    context, report, recommendations = _snapshot_models(snapshot, project_path)
    engine = ChatEngine(provider_name=provider)
    try:
        response = await engine.ask(context, report, recommendations, text, chat_history=history, snapshot=snapshot)
    except ValueError as error:
        return Reply("I cannot answer because no AI provider is configured. "
                     f"Reason: {scrub(str(error))}. Go to Settings (Command Palette: 'AI Provider: Configure').",
                     error="PROVIDER_NOT_CONFIGURED")
    except AppError as error:
        logger.warning("Chat provider error (%s): %s", error.error_code, error.details)
        return Reply(error.message, error=error.error_code)
    return Reply(response["assistantResponse"], meta={
        "references": response.get("references"), "confidence": response.get("confidence"),
        "model": response.get("model")})


async def _prepare(project_path: str, session_id: str, approved: Optional[Set[str]] = None) -> AgentContext:
    root: Optional[Path] = None
    if project_path:
        try:
            root = resolve_workspace_root(project_path)
        except Exception:  # noqa: BLE001 - unauthorised/missing folder: chat only, no file access
            root = None
    return AgentContext(project_path, root, session_id, await load_policy(project_path), approved or set())


async def handle_message(project_path: str, session_id: Optional[str], text: str,
                         provider: Optional[str] = None) -> Dict[str, Any]:
    """Process one user message and return the persisted session + assistant message."""
    text = (text or "").strip()[:MAX_MESSAGE_CHARS]
    if not text:
        raise ValueError("message must not be empty")
    if session_id:
        if await sessions.get_session_messages(project_path, session_id) is None:
            raise LookupError("chat session not found")
    else:
        session_id = (await sessions.create_session(project_path))["sessionId"]

    user_message = await sessions.add_message(project_path, session_id, "user", text)
    intent = intents.classify(text)

    current_title = await sessions.get_title(project_path, session_id)
    if not current_title:
        title = intents.title_for(intent, text) or sessions.generate_title(text)
        if title:
            await sessions.set_title(project_path, session_id, title)

    ctx = await _prepare(project_path, session_id)
    descriptor = _descriptor_for(intent)
    if descriptor is not None:
        reply = await _dispatch(ctx, descriptor)
    else:
        reply = await _llm_reply(project_path, session_id, text, provider)

    assistant = await sessions.add_message(project_path, session_id, "assistant", reply.text, reply.blocks)
    for block in reply.blocks:
        if block.get("type") == "actions":
            await pending.attach_message(block["actionId"], assistant["messageId"])
    session = await sessions.get_session_messages(project_path, session_id)
    return {
        "sessionId": session_id,
        "title": session["title"] if session else sessions.DEFAULT_TITLE,
        "userMessage": user_message,
        "message": assistant,
        "error": reply.error,
        "meta": reply.meta,
    }


def _valid_choice(kind: str, payload: Dict[str, Any], choice: str) -> bool:
    """Whether *choice* is one of the answers this card actually offered."""
    if kind == "permission":
        return choice in ("allow", "deny")
    if choice == "cancel":
        return kind in ("clean_method", "clean_output", "choose_site", "choose_job")
    if kind == "clean_method":
        return choice in ("ai", "script")
    if kind == "clean_output":
        return choice in ("separate", "in_place")
    if kind == "choose_site":
        tail = choice.split(":", 1)[1] if choice.startswith("site:") else ""
        return tail.isdigit() and int(tail) < len(payload.get("sites") or [])
    if kind == "choose_job":
        return choice.startswith("job:") and bool(choice.split(":", 1)[1])
    if kind == "revert_offer":
        return choice == "revert"
    return False


async def respond_to_action(project_path: str, session_id: str, action_id: str, choice: str) -> Dict[str, Any]:
    """Apply the user's answer to a pending card. The stored payload alone decides what happens."""
    choice = (choice or "").strip()[:80]
    waiting = await pending.get(project_path, action_id)
    if waiting is None or waiting["status"] != "pending":
        raise LookupError("This question was already answered or has expired.")
    if waiting["sessionId"] != session_id:
        raise LookupError("This question belongs to a different chat.")
    if not _valid_choice(waiting["kind"], waiting["payload"], choice):
        raise LookupError("That option is not valid for this question.")
    action = await pending.claim(project_path, action_id, choice)  # atomic: only one answer wins
    if action is None:
        raise LookupError("This question was already answered or has expired.")
    payload = dict(action["payload"])
    kind = action["kind"]
    approved = set(payload.get("approved") or [])
    ctx = await _prepare(project_path, session_id, approved)

    resolved_label = None
    if kind == "permission":
        if choice == "allow":
            ctx.approved.add(payload.get("needs", ""))
            resolved_label = "Allowed"
            reply = await _dispatch(ctx, payload)
        else:
            resolved_label = "Denied"
            reply = await _denied(ctx, payload)
    elif choice == "cancel":
        resolved_label = "Cancelled"
        reply = Reply("Okay, I cancelled that. Nothing was changed.")
    elif kind == "clean_method" and choice in ("ai", "script"):
        resolved_label = "AI-based cleaning" if choice == "ai" else "Script-based cleaning"
        reply = await _dispatch(ctx, {**payload, "method": choice})
    elif kind == "clean_output" and choice in ("separate", "in_place"):
        resolved_label = "Separate output files" if choice == "separate" else "Modify existing files"
        reply = await _dispatch(ctx, {**payload, "outputMode": choice})
    elif kind == "choose_site" and choice.startswith("site:"):
        index = int(choice.split(":", 1)[1]) if choice.split(":", 1)[1].isdigit() else -1
        sites = payload.get("sites") or []
        if not 0 <= index < len(sites):
            raise LookupError("That option is not valid.")
        resolved_label = f"{sites[index][0]}:{sites[index][1]}"
        reply = await _dispatch(ctx, {**payload, "only": sites[index]})
    elif kind == "choose_job" and choice.startswith("job:"):
        resolved_label = "Selected"
        reply = await _dispatch(ctx, {**payload, "jobId": choice.split(":", 1)[1]})
    elif kind == "revert_offer" and choice == "revert":
        resolved_label = "Reverted"
        reply = await _dispatch(ctx, payload)
    else:
        raise LookupError("That option is not valid for this question.")

    # Mark the original card as answered so a restored chat shows it resolved.
    updated_blocks = None
    if action.get("messageId"):
        original = await sessions.get_message(project_path, session_id, action["messageId"])
        if original:
            updated_blocks = []
            for block in original["blocks"]:
                if block.get("type") == "actions" and block.get("actionId") == action_id:
                    block = {**block, "resolved": {"choice": choice, "label": resolved_label or choice}}
                updated_blocks.append(block)
            await sessions.update_message_blocks(project_path, session_id, action["messageId"], updated_blocks)

    assistant = None
    if reply.text or reply.blocks:
        assistant = await sessions.add_message(project_path, session_id, "assistant", reply.text, reply.blocks)
        for block in reply.blocks:
            if block.get("type") == "actions":
                await pending.attach_message(block["actionId"], assistant["messageId"])
    return {"sessionId": session_id, "resolvedMessageId": action.get("messageId"),
            "resolvedBlocks": updated_blocks, "message": assistant}


async def _denied(ctx: AgentContext, payload: Dict[str, Any]) -> Reply:
    """The user said no: say so, and retire any proposal that was waiting for permission."""
    if payload.get("intent") == "apply_change" and ctx.root is not None:
        try:
            ChangeStore(ctx.root).reject(payload["changeId"])
        except ChangeStoreError:
            pass
        return Reply("Okay, I did not change the file. The proposal was discarded.")
    return Reply("Okay, I did not do that.")


async def show_changes(project_path: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Post the latest changes as an inline viewer (used by the View Changes command)."""
    if not session_id:
        session_id = (await sessions.create_session(project_path))["sessionId"]
    ctx = await _prepare(project_path, session_id)
    reply = await _dispatch(ctx, {"intent": intents.SHOW_CHANGES})
    assistant = await sessions.add_message(project_path, session_id, "assistant", reply.text, reply.blocks)
    if not await sessions.get_title(project_path, session_id):
        await sessions.set_title(project_path, session_id, "Review changes")
    return {"sessionId": session_id, "message": assistant}


__all__ = ["handle_message", "respond_to_action", "show_changes"]
