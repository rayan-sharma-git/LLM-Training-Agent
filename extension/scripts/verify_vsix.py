"""
Audit a built .vsix against the "install once, use in every project" contract.

Run:  python scripts/verify_vsix.py [path-to-vsix]

Exits non-zero if any requirement is violated, so it can gate a release.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

#: Backend modules required for `uvicorn main:app` to import at all.
REQUIRED_BACKEND_MODULES = [
    "main.py",
    "requirements.txt",
    "api/routes.py",
    "api/result_mapping.py",
    "ai/llm.py",
    "chat/engine.py",
    "chat/store.py",
    "cleaning/dataset_cleaner.py",
    "cleaning/dataset_io.py",
    "core/config.py",
    "core/redaction.py",
    "core/runtime_config.py",
    "core/security.py",
    "experiments/service.py",
    "models/schemas.py",
    "prediction/engine.py",
    "scanner/scanner.py",
    "storage/analysis_store.py",
    "storage/database.py",
    "storage/migrations.py",
]

#: Compiled extension modules required at activation.
REQUIRED_OUT_MODULES = [
    "out/extension.js",
    "out/commands/analyzerCommands.js",
    "out/commands/changeCommands.js",
    "out/commands/chatCommands.js",
    "out/services/apiClient.js",
    "out/services/backendManager.js",
    "out/services/dependencyManager.js",
    "out/services/extensionPaths.js",
    "out/services/projectRoot.js",
    "out/services/settings.js",
    "out/views/chatWebviewProvider.js",
    "out/views/settingsWebviewProvider.js",
    "out/views/simpleTreeView.js",
    "out/views/viewIds.js",
]

#: Nothing matching these may ship.
FORBIDDEN_SUBSTRINGS = [
    ".env",
    "__pycache__",
    ".pyc",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".log",
    ".vsix",
    "llm_training_agent.db",
    "out_backup.zip",
    "out-test",
    ".vscode-test",
]

#: Regexes that suggest a hardcoded credential leaked into the package.
#: These are anchored so ordinary prose ("a task-specific fine-tune") and
#: credential *rejection patterns* cannot trigger them: a match must look like
#: a real key, not like the word "sk-" followed by English.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
]

#: Modules that legitimately *mention* credential shapes in order to redact
#: them. Excluded from the secret scan for that reason.
SECRET_SCAN_EXEMPT = frozenset({"backend/core/redaction.py"})

#: Suffixes scanned for secrets and development-path assumptions.
TEXT_SUFFIXES = (".py", ".ts", ".js", ".json", ".md", ".txt", ".svg")


def audit(vsix_path: Path) -> list[str]:
    """Return the list of violations; an empty list means the VSIX is compliant."""
    problems: list[str] = []

    with zipfile.ZipFile(vsix_path) as archive:
        prefix = "extension/"
        ext = {
            n[len(prefix):].replace("\\", "/")
            for n in archive.namelist()
            if n.startswith(prefix) and not n.endswith("/")
        }

        manifest = json.loads(archive.read(f"{prefix}package.json"))

        def read_text(name: str) -> str:
            return archive.read(f"{prefix}{name}").decode("utf-8", "ignore")

        # --- Runtime dependencies (activation requires axios) ---------------
        if "node_modules/axios/package.json" not in ext:
            problems.append("missing runtime dependency: node_modules/axios")
        if not any(n.startswith("node_modules/") for n in ext):
            problems.append("no node_modules packaged: axios would be missing at runtime")

        # --- Compiled extension code ----------------------------------------
        for module in REQUIRED_OUT_MODULES:
            if module not in ext:
                problems.append(f"missing compiled extension module: {module}")

        # --- Bundled backend must be complete and importable -----------------
        for module in REQUIRED_BACKEND_MODULES:
            if f"backend/{module}" not in ext:
                problems.append(f"missing bundled backend module: backend/{module}")

        prompts = [n for n in ext if n.startswith("backend/ai/prompts/") and n.endswith(".md")]
        if len(prompts) < 5:
            problems.append(f"bundled prompts look incomplete ({len(prompts)} found)")

        if "resources/icon.svg" not in ext:
            problems.append("missing resources/icon.svg")

        # --- Nothing forbidden ----------------------------------------------
        for name in sorted(ext):
            for pattern in FORBIDDEN_SUBSTRINGS:
                if pattern in name:
                    problems.append(f"forbidden file packaged: {name}")
                    break
        if any("/tests/" in n for n in ext):
            problems.append("test files packaged")

        # --- No secrets, no development-path assumptions ---------------------
        # Third-party dependency data (mime-db and friends) legitimately contains
        # long "sk-" style strings, so only first-party code is scanned.
        for name in sorted(ext):
            if name.startswith("node_modules/"):
                continue
            if not name.endswith(TEXT_SUFFIXES):
                continue
            body = read_text(name)
            if name not in SECRET_SCAN_EXEMPT:
                for pattern in SECRET_PATTERNS:
                    if pattern.search(body):
                        problems.append(f"possible hardcoded secret in {name}")
                        break
            if name.startswith("out/") and name.endswith(".js"):
                if "'..', '..', '..'" in body or '"..", "..", ".."' in body:
                    problems.append(f"repository-relative path assumption in {name}")
                if "process.cwd()" in body:
                    problems.append(f"process.cwd() used as extension root in {name}")

        # --- Manifest behaves like a normal VS Code extension ----------------
        if manifest.get("main") != "./out/extension.js":
            problems.append("manifest main is not ./out/extension.js")
        if not manifest.get("publisher"):
            problems.append("manifest has no publisher")
        events = manifest.get("activationEvents", [])
        if "onStartupFinished" in events:
            problems.append("activationEvents still contains onStartupFinished")
        for event in events:
            if event.startswith(("workspaceContains", "workspaceFileContains", "onLanguage:")):
                problems.append(f"project-scoped activation event present: {event}")
        for entry in manifest.get("contributes", {}).get("menus", {}).get("commandPalette", []):
            if entry.get("when"):
                problems.append(f"command hidden by a when-clause: {entry.get('command')}")
        properties = manifest["contributes"]["configuration"]["properties"]
        for key in ("llmTrainingAgent.provider", "llmTrainingAgent.model"):
            if properties[key].get("scope") != "application":
                problems.append(f"{key} must be user-level (application)")

    return problems


def main() -> int:
    vsix = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("llm-training-agent-1.0.0.vsix")
    if not vsix.exists():
        print(f"VSIX not found: {vsix}")
        return 1

    problems = audit(vsix)
    if problems:
        print(f"FAIL - {vsix.name} violates {len(problems)} requirement(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"PASS - {vsix.name} satisfies all packaging requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
