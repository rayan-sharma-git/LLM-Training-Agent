"""Tests for check_requirements.py, the dependency check the extension runs.

These guard the defects of the previous import-based check:
  * a marker-excluded package (tomli on Python >= 3.11) was reported missing
    forever, so the "install dependencies" step could never succeed;
  * an installed-but-too-old package (SQLAlchemy < 2.0.30 on new Pythons)
    counted as available because it still imported.
"""
from __future__ import annotations

import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import check_requirements as cr

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND_DIR / "check_requirements.py"


def _fake_installed(monkeypatch, installed):
    def fake_version(name):
        key = name.lower().replace("_", "-")
        if key not in installed:
            raise metadata.PackageNotFoundError(name)
        return installed[key]

    monkeypatch.setattr(cr.metadata, "version", fake_version)


def _fake_python(monkeypatch, major, minor, micro=0):
    monkeypatch.setattr(cr.sys, "version_info", SimpleNamespace(major=major, minor=minor, micro=micro))


# --- version comparison -------------------------------------------------


def test_parse_version_reads_the_numeric_release():
    assert cr.parse_version("2.0.30") == (2, 0, 30)
    assert cr.parse_version("2.0.30rc1") == (2, 0, 30)
    assert cr.parse_version("1.2.3.post4") == (1, 2, 3)
    assert cr.parse_version("not-a-version") == (0,)


def test_floor_is_enforced_numerically_not_lexically():
    assert cr.version_satisfies("2.0.23", ">=2.0.30") is False
    assert cr.version_satisfies("2.0.30", ">=2.0.30") is True
    assert cr.version_satisfies("2.10.0", ">=2.9") is True  # "2.10" > "2.9", not lexical
    assert cr.version_satisfies("2.10.0", "<2.9") is False


def test_combined_bounds_and_trailing_zeros():
    assert cr.version_satisfies("5.10.1", ">=5.9.0,<5.11.0") is True
    assert cr.version_satisfies("5.11.0", ">=5.9.0,<5.11.0") is False
    assert cr.version_satisfies("1.0", "==1.0.0") is True
    assert cr.version_satisfies("2.0.7", "==2.0.*") is True
    assert cr.version_satisfies("2.1.0", "==2.0.*") is False
    assert cr.version_satisfies("2.5.1", "~=2.5") is True
    assert cr.version_satisfies("3.0.0", "~=2.5") is False


def test_unparseable_specifier_is_unknown_not_failure():
    assert cr.version_satisfies("1.0", "banana") is None


# --- markers ------------------------------------------------------------


def test_python_version_marker_follows_the_running_interpreter(monkeypatch):
    marker = 'python_version < "3.11"'
    _fake_python(monkeypatch, 3, 13)
    assert cr.marker_applies(marker) is False
    _fake_python(monkeypatch, 3, 10)
    assert cr.marker_applies(marker) is True
    _fake_python(monkeypatch, 3, 11)
    assert cr.marker_applies(marker) is False


def test_and_or_markers(monkeypatch):
    _fake_python(monkeypatch, 3, 12)
    monkeypatch.setattr(cr.sys, "platform", "linux")
    assert cr.marker_applies('sys_platform == "win32" or python_version >= "3.12"') is True
    assert cr.marker_applies('sys_platform == "win32" and python_version >= "3.12"') is False


def test_unsupported_marker_syntax_is_unknown():
    assert cr.marker_applies('extra == "gpu"') is None
    assert cr.marker_applies("garbage") is None


# --- whole-file check ---------------------------------------------------


def test_marker_excluded_package_is_not_reported_missing(monkeypatch):
    # The original bug: tomli is required only below Python 3.11, so on 3.13
    # pip never installs it and the extension must not demand it.
    _fake_python(monkeypatch, 3, 13, 1)
    _fake_installed(monkeypatch, {"fastapi": "0.104.1"})
    result = cr.check('fastapi==0.104.1\ntomli>=1.2.3; python_version < "3.11"\n')
    assert result["ok"] is True
    assert result["missing"] == []
    assert result["skipped"] == ["tomli"]


def test_marker_included_package_is_required(monkeypatch):
    _fake_python(monkeypatch, 3, 10, 6)
    _fake_installed(monkeypatch, {})
    result = cr.check('tomli>=1.2.3; python_version < "3.11"\n')
    assert result["missing"] == ["tomli"]
    assert result["ok"] is False


def test_too_old_package_is_reported_even_though_it_imports(monkeypatch):
    _fake_installed(monkeypatch, {"sqlalchemy": "2.0.23", "greenlet": "3.0.1"})
    result = cr.check("sqlalchemy>=2.0.30\ngreenlet>=3.0.0\n")
    assert result["ok"] is False
    assert result["outdated"] == [{"name": "sqlalchemy", "installed": "2.0.23", "required": ">=2.0.30"}]
    assert result["missing"] == []


def test_missing_package_is_reported(monkeypatch):
    _fake_installed(monkeypatch, {"fastapi": "0.104.1"})
    result = cr.check("fastapi==0.104.1\nhttpx==0.25.2\n")
    assert result["missing"] == ["httpx"]


def test_names_are_matched_by_distribution_not_import_name(monkeypatch):
    # pip name != import name for these; metadata lookup must not care.
    _fake_installed(monkeypatch, {"python-dotenv": "1.0.0", "pyyaml": "6.0.3"})
    result = cr.check("python-dotenv==1.0.0\nPyYAML==6.0.3\n")
    assert result["ok"] is True


def test_extras_comments_blanks_and_options_are_handled(monkeypatch):
    _fake_installed(monkeypatch, {"uvicorn": "0.24.0"})
    text = "# a comment\n\n-r other.txt\n--index-url https://example.invalid\nuvicorn[standard]==0.24.0  # trailing\n"
    result = cr.check(text)
    assert result["checked"] == 1
    assert result["ok"] is True


def test_exact_pin_accepts_newer_but_upper_bounds_still_apply(monkeypatch):
    _fake_installed(monkeypatch, {"uvicorn": "0.54.0", "cohere": "5.12.0"})
    # A newer-than-pinned version is fine: we ask "can it run", not "is it identical".
    assert cr.check("uvicorn==0.24.0\n")["ok"] is True
    # ...but an older one is not.
    _fake_installed(monkeypatch, {"uvicorn": "0.20.0"})
    assert cr.check("uvicorn==0.24.0\n")["outdated"][0]["required"] == ">=0.24.0"
    # Explicit upper bounds are honoured.
    _fake_installed(monkeypatch, {"cohere": "5.12.0"})
    assert cr.check("cohere>=5.9.0,<5.11.0\n")["ok"] is False
    # Strict mode keeps the exact pin.
    _fake_installed(monkeypatch, {"uvicorn": "0.54.0"})
    assert cr.check("uvicorn==0.24.0\n", exact_pins_as_minimum=False)["ok"] is False


def test_shipped_backend_requirements_parse_and_demand_no_unused_sdks():
    text = (BACKEND_DIR / "requirements.txt").read_text(encoding="utf-8")
    names = [p[0].lower() for p in map(cr.parse_requirement_line, text.splitlines()) if p]
    assert "fastapi" in names and "sqlalchemy" in names and "httpx" in names
    # The backend calls every provider over REST; these SDKs are never imported.
    assert "google-generativeai" not in names
    assert "cohere" not in names
    # No development tooling in the runtime file.
    for dev_only in ("pytest", "pytest-asyncio", "ruff", "black", "mypy"):
        assert dev_only not in names


# --- command line -------------------------------------------------------


def test_cli_prints_json_and_exits_zero(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely-not-installed-xyz>=1.0\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(requirements)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["missing"] == ["definitely-not-installed-xyz"]
    assert payload["ok"] is False


def test_cli_fails_cleanly_for_an_unreadable_file(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope.txt")], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2
    assert "cannot read requirements file" in proc.stderr
