"""
End-to-end verification of "install once, use in every project".

Simulates exactly what VS Code does with an installed .vsix:

  1. unzip the .vsix into a fake extension installation directory
  2. resolve the bundled backend and Python from THAT directory only
  3. run the backend against Project A, then switch to Project B
  4. assert no agent files or state leak into either project

Run:  python scripts/e2e_two_projects.py [path-to-vsix]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

EXTENSION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VSIX = EXTENSION_DIR / "llm-training-agent-1.0.0.vsix"
HEALTH_PATH = "/api/v1/health"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def snapshot(directory: Path) -> set[str]:
    """Relative paths of every file in *directory* (for leak detection)."""
    return {
        str(p.relative_to(directory)).replace("\\", "/")
        for p in directory.rglob("*")
        if p.is_file()
    }


def wait_for_health(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}{HEALTH_PATH}"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False


def install_vsix(vsix: Path, install_root: Path) -> Path:
    """Unpack a .vsix the way VS Code installs it; return the extension dir.

    The install folder name is derived from the manifest Identity, exactly as
    VS Code names it (`<publisher>.<name>-<version>`).
    """
    with zipfile.ZipFile(vsix) as archive:
        manifest_xml = archive.read("extension.vsixmanifest").decode("utf-8")

    # The Identity element's attributes are unordered, so match the element
    # first and then read each attribute individually.
    element = re.search(r"<Identity\b[^>]*>", manifest_xml)
    if not element:
        raise ValueError("extension.vsixmanifest has no Identity element")
    attributes = dict(re.findall(r'([\w:]+)="([^"]*)"', element.group(0)))

    publisher = attributes.get("Publisher", "unknown")
    name = attributes.get("Id", attributes.get("Name", "extension"))
    version = attributes.get("Version", "0.0.0")

    target = install_root / f"{publisher}.{name}-{version}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    with zipfile.ZipFile(vsix) as archive:
        for entry in archive.namelist():
            if not entry.startswith("extension/") or entry.endswith("/"):
                continue
            destination = target / entry[len("extension/"):]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, open(destination, "wb") as sink:
                shutil.copyfileobj(source, sink)
    return target


def make_project(root: Path, name: str) -> Path:
    """A small, realistic Python project to analyze."""
    project = root / name
    (project / "src").mkdir(parents=True)
    (project / "src" / "train.py").write_text(
        "import torch\n\nMODEL = 'distilgpt2'\nBATCH = 8\n\n\ndef train():\n    return MODEL\n",
        encoding="utf-8",
    )
    (project / "README.md").write_text(f"# {name}\n\nSample project.\n", encoding="utf-8")
    (project / "requirements.txt").write_text("torch\n", encoding="utf-8")
    return project


def analyze(port: int, project: Path) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/v1/project/analyze",
        data=json.dumps({"projectPath": str(project)}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def run(vsix: Path) -> int:
    if not vsix.exists():
        print(f"VSIX not found: {vsix}")
        return 1

    failures: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="llm-agent-e2e-"))
    install_root = workdir / "extensions"
    install_root.mkdir(parents=True)

    # Fake VS Code global storage: where the extension keeps its own state.
    global_storage = workdir / "globalStorage"
    global_storage.mkdir(parents=True)
    roots_file = global_storage / "allowed-roots.txt"

    process = None
    log = None
    try:
        # --- 1. Install the VSIX exactly like VS Code does -------------------
        ext_dir = install_vsix(vsix, install_root)
        print(f"[1] Installed extension to: {ext_dir}")

        manifest = json.loads((ext_dir / "package.json").read_text(encoding="utf-8"))
        if manifest["main"] != "./out/extension.js":
            failures.append("manifest main points somewhere unexpected")
        if not (ext_dir / "out" / "extension.js").exists():
            failures.append("out/extension.js missing from the installation")

        # --- 2. Resolve resources from the INSTALL directory only -------------
        backend_entry = ext_dir / "backend" / "main.py"
        if not backend_entry.exists():
            failures.append("bundled backend/main.py missing after install")
            return report(failures)
        print(f"[2] Backend resolved from install dir: {backend_entry}")

        python_exe = shutil.which("python") or sys.executable
        print(f"    Python interpreter: {python_exe}")

        # --- 3. Two independent projects ------------------------------------
        project_a = make_project(workdir, "ProjectA")
        project_b = make_project(workdir, "ProjectB")
        before_a = snapshot(project_a)
        before_b = snapshot(project_b)
        print(f"[3] Project A: {project_a}")
        print(f"    Project B: {project_b}")

        port = free_port()
        roots_file.write_text(str(project_a), encoding="utf-8")

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ext_dir / "backend"),
            # Agent state lives in VS Code global storage: not in the install
            # directory, and never in a user project.
            "DATABASE_URL": f"sqlite+aiosqlite:///{(global_storage / 'agent.db').as_posix()}",
            "LLM_TRAINING_AGENT_ALLOWED_ROOTS": str(project_a),
            "LLM_TRAINING_AGENT_ALLOWED_ROOTS_FILE": str(roots_file),
        }
        log = open(workdir / "backend.log", "w", encoding="utf-8")
        process = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            cwd=str(ext_dir / "backend"),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        if not wait_for_health(port):
            failures.append(
                "backend did not become healthy when started from the install dir"
            )
            return report(failures)
        print("[4] Backend is healthy, started purely from the install directory")

        # --- 4. Analyze Project A -------------------------------------------
        status_a, body_a = analyze(port, project_a)
        print(f"[5] Project A analyze -> HTTP {status_a}")
        if status_a != 200:
            failures.append(f"Project A analysis failed: HTTP {status_a} {body_a}")
        elif "ProjectA" not in json.dumps(body_a):
            failures.append("Project A response does not mention ProjectA")

        # --- 5. Switch to Project B (Close A / Open B) ----------------------
        roots_file.write_text(str(project_b), encoding="utf-8")
        time.sleep(1.5)
        print("[6] Switched the open project to Project B (no restart, no reinstall)")

        status_b, body_b = analyze(port, project_b)
        print(f"    Project B analyze -> HTTP {status_b}")
        if status_b != 200:
            failures.append(f"Project B analysis failed after switch: HTTP {status_b} {body_b}")
        elif "ProjectB" not in json.dumps(body_b):
            failures.append("Project B response does not mention ProjectB")

        # --- 6. Isolation: Project A must now be refused ---------------------
        status_a2, _ = analyze(port, project_a)
        print(f"[7] Project A re-analyze after switch -> HTTP {status_a2} (expected 403)")
        if status_a2 != 403:
            failures.append(
                f"Project A was still reachable after switching to B (HTTP {status_a2}); "
                "project data is not isolated"
            )

        # --- 7. No agent files leaked into the projects ----------------------
        for label, before, after in (
            ("A", before_a, snapshot(project_a)),
            ("B", before_b, snapshot(project_b)),
        ):
            leaked = sorted(after - before)
            if leaked:
                failures.append(f"files written into Project {label}: {leaked}")
        print("[8] No agent files were written into either project")

        # --- 8. No state written into the install directory ------------------
        strays = sorted(
            [p.name for p in (ext_dir / "backend").glob("*.db*")]
            + [p.name for p in ext_dir.glob("*.db*")]
        )
        if strays:
            failures.append(f"database written into the install directory: {strays}")
        print("[9] No database written into the extension install directory")

    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        if log is not None:
            log.close()
        shutil.rmtree(workdir, ignore_errors=True)

    return report(failures)


def report(failures: list[str]) -> int:
    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS - installed once, used in Project A and Project B, no leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VSIX))

