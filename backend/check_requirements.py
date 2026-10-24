"""Report which entries of a requirements file are missing or too old.

Used by the VS Code extension to decide whether the interpreter it found can
run the bundled backend, *before* trying to start it. Standard library only,
so it runs in an interpreter that has none of the backend dependencies.

Usage:
    python check_requirements.py path/to/requirements.txt

Prints one JSON object on stdout:
    {"python": "3.12.1", "ok": false, "checked": 9,
     "missing": ["fastapi"],
     "outdated": [{"name": "sqlalchemy", "installed": "2.0.23", "required": ">=2.0.30"}],
     "skipped": ["tomli"]}

Why this exists instead of "try to import each package":
  * environment markers (``tomli; python_version < "3.11"``) must be honoured,
    otherwise a package pip rightly skips is reported missing forever;
  * version floors/pins must be honoured, otherwise an old SQLAlchemy that
    imports fine but crashes the backend counts as "installed";
  * pip names are not always import names (PyYAML -> yaml, python-dotenv -> dotenv).
Checking installed distribution metadata sidesteps all three.
"""
from __future__ import annotations

import json
import operator
import re
import sys
from importlib import metadata
from typing import Callable, Dict, List, Optional, Tuple

_REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;]*?)\s*(?:;\s*(.*))?$")
_SPEC_RE = re.compile(r"^(===|==|!=|~=|<=|>=|<|>)\s*([A-Za-z0-9.*+!_-]+)$")
_MARKER_TERM_RE = re.compile(r"""^\s*([a-z_]+)\s*(==|!=|<=|>=|<|>|~=)\s*(?:'([^']*)'|"([^"]*)")\s*$""")

_OPS: Dict[str, Callable[[Tuple[int, ...], Tuple[int, ...]], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def parse_version(text: str) -> Tuple[int, ...]:
    """Numeric release tuple of a version string ('2.0.30rc1' -> (2, 0, 30))."""
    release: List[int] = []
    for piece in re.split(r"[.+\-_]", text.strip().lstrip("vV")):
        match = re.match(r"\d+", piece)
        if not match:
            break
        release.append(int(match.group()))
        if match.end() != len(piece):
            break  # pre/post/dev suffix: the numeric release ends here
    return tuple(release) or (0,)


def _pad(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    size = max(len(a), len(b))
    return a + (0,) * (size - len(a)), b + (0,) * (size - len(b))


def version_satisfies(installed: str, specifier: str) -> Optional[bool]:
    """True/False for a comma-separated specifier set, None when unparseable."""
    have = parse_version(installed)
    for raw in filter(None, (part.strip() for part in specifier.split(","))):
        match = _SPEC_RE.match(raw)
        if not match:
            return None
        op, wanted = match.group(1), match.group(2)
        if op == "===":
            op = "=="
        if wanted.endswith(".*"):
            prefix = parse_version(wanted[:-2])
            head = have[: len(prefix)] + (0,) * max(0, len(prefix) - len(have))
            equal = head == prefix
            if op == "==" and not equal:
                return False
            if op == "!=" and equal:
                return False
            if op not in ("==", "!="):
                return None
            continue
        want = parse_version(wanted)
        if op == "~=":
            # ~=X.Y.Z means >=X.Y.Z and ==X.Y.*
            lower_a, lower_b = _pad(have, want)
            if lower_a < lower_b:
                return False
            keep = max(1, len(want) - 1)
            if have[:keep] != want[:keep]:
                return False
            continue
        a, b = _pad(have, want)
        if not _OPS[op](a, b):
            return False
    return True


def _marker_value(name: str) -> Optional[str]:
    info = sys.version_info
    values = {
        "python_version": f"{info.major}.{info.minor}",
        "python_full_version": f"{info.major}.{info.minor}.{info.micro}",
        "sys_platform": sys.platform,
        "os_name": __import__("os").name,
        "platform_system": {"win32": "Windows", "darwin": "Darwin"}.get(sys.platform, "Linux"),
    }
    return values.get(name)


def marker_applies(marker: str) -> Optional[bool]:
    """Evaluate a PEP 508 marker; None when it uses syntax we do not support."""
    result: Optional[bool] = None
    for or_clause in re.split(r"\s+or\s+", marker.strip()):
        clause_ok: Optional[bool] = True
        for term in re.split(r"\s+and\s+", or_clause):
            match = _MARKER_TERM_RE.match(term)
            if not match:
                return None
            name, op = match.group(1), match.group(2)
            literal = match.group(3) if match.group(3) is not None else match.group(4)
            actual = _marker_value(name)
            if actual is None:
                return None
            if name.startswith("python_"):
                passed = version_satisfies(actual, f"{op}{literal}")
            elif op == "==":
                passed = actual == literal
            elif op == "!=":
                passed = actual != literal
            else:
                return None
            if passed is None:
                return None
            clause_ok = clause_ok and passed
        result = bool(result) or bool(clause_ok)
    return result


def parse_requirement_line(line: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """(name, specifier, marker) for one requirements line, None for non-requirements."""
    text = line.split(" #", 1)[0].strip()
    if not text or text.startswith(("#", "-")):
        return None
    match = _REQ_RE.match(text)
    if not match:
        return None
    return match.group(1), match.group(3).strip(), match.group(4)


def _as_minimum(specifier: str) -> str:
    """Treat exact pins (``==1.2.3``) as minimums (``>=1.2.3``).

    This check answers "can this interpreter run the backend?", not "is the
    environment byte-identical to the tested one". Without this, a user whose
    environment has a *newer* uvicorn than the pinned one would be told it is
    "outdated" and be asked to downgrade it. Upper bounds (``<``, ``<=``),
    exclusions (``!=``) and wildcard pins (``==1.2.*``) are kept as written.
    """
    parts = []
    for raw in (part.strip() for part in specifier.split(",")):
        match = re.match(r"^==\s*([A-Za-z0-9.+!_-]+)$", raw)
        if match and not match.group(1).endswith(".*"):
            parts.append(f">={match.group(1)}")
        else:
            parts.append(raw)
    return ",".join(part for part in parts if part)


def check(requirements_text: str, exact_pins_as_minimum: bool = True) -> Dict[str, object]:
    missing: List[str] = []
    outdated: List[Dict[str, str]] = []
    skipped: List[str] = []
    checked = 0

    for line in requirements_text.splitlines():
        parsed = parse_requirement_line(line)
        if parsed is None:
            continue
        name, specifier, marker = parsed

        if marker is not None:
            applies = marker_applies(marker)
            if applies is not True:
                # Not applicable here, or a marker we cannot evaluate. Either way
                # do not demand it: pip decides at install time, and demanding a
                # package pip will skip would loop forever.
                skipped.append(name)
                continue

        checked += 1
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        effective = _as_minimum(specifier) if exact_pins_as_minimum else specifier
        if effective and version_satisfies(installed, effective) is False:
            outdated.append({"name": name, "installed": installed, "required": effective})

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "ok": not missing and not outdated,
        "checked": checked,
        "missing": missing,
        "outdated": outdated,
        "skipped": skipped,
    }


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: check_requirements.py <requirements.txt>", file=sys.stderr)
        return 2
    try:
        with open(argv[1], "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        print(f"cannot read requirements file: {error}", file=sys.stderr)
        return 2
    print(json.dumps(check(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
