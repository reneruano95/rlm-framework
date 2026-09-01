"""The acceptance test for "distributed by copying the folder", as one command.

    .venv/Scripts/python.exe tests/verify_distribution.py

Not collected by pytest (its name does not match `test_*.py`), because it builds
environments and takes a minute. It is the check that would live in CI if this repo
had CI -- it does not, and that absence is the single largest gap against the
comparison standard, which builds a wheel and imports it on five Python versions on
every push. Until that exists, this script is the difference between a criterion and
a procedure someone remembers.

Five checks, in the order they can fail:

  1. COPY      the package directory into an empty tree, with nothing else, and import it
  2. CHEAP     a bare `import rlm` loads ONE module and no HTTP client -- the property
               `sandbox/manager.py`'s AppContainer staging depends on, and the one the
               copy test itself cannot see, because the sandbox does not run off Windows
  3. DATA      the shipped config validates from the copy, with every prompt sha matching
  4. SUITE     `pytest --pyargs rlm` passes with zero edits, zero errors, and no skip
               attributable to a missing repo file
  5. WHEEL     the same, from a BUILT WHEEL installed in a venv that never saw the repo

Each prints what it measured, not just a tick. A green line with no number is how a
check stops being one.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "rlm"
DEPS = ["pydantic>=2.9", "pyyaml>=6.0", "duckdb>=1.1", "httpx>=0.27",
        "pytest>=8.3", "pytest-asyncio>=0.24", "hypothesis>=6.112", "pytest-timeout>=2.3"]

failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {detail}")
    if not ok:
        failures.append(name)


def run(py: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(py), *args], capture_output=True, text=True, cwd=cwd)


def make_venv(root: Path, extra: list[str] | None = None) -> Path:
    venv.EnvBuilder(with_pip=True).create(root)
    py = root / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python")
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                    *DEPS, *(extra or [])], check=True, capture_output=True)
    return py


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("Building a clean environment with only the declared dependencies...")
        py = make_venv(tmp / "venv")

        # 1. COPY
        proj = tmp / "consumer"
        proj.mkdir()
        shutil.copytree(PKG, proj / "rlm",
                        ignore=shutil.ignore_patterns("__pycache__"))
        n_py = len(list((proj / "rlm").rglob("*.py")))
        r = run(py, "-c", "import rlm; print(len(rlm.__all__))", cwd=proj)
        check("copy + import", r.returncode == 0,
              f"{n_py} .py files copied, {r.stdout.strip() or r.stderr.strip()[:40]} public names")

        # 2. CHEAP -- the AppContainer property
        r = run(py, "-c",
                "import sys, rlm; "
                "print(len([m for m in sys.modules if m.startswith('rlm')]), 'httpx' in sys.modules)",
                cwd=proj)
        parts = r.stdout.split()
        cheap = r.returncode == 0 and parts[:2] == ["1", "False"]
        check("bare import stays cheap", cheap,
              f"{parts[0] if parts else '?'} module(s) loaded, httpx present = {parts[1] if len(parts) > 1 else '?'}")

        # 3. DATA
        r = run(py, "-c",
                "import yaml; from rlm.config import Config, default_config_path; "
                "p = default_config_path(); "
                "c = Config.model_validate(yaml.safe_load(p.read_text(encoding='utf-8'))); "
                "print(sum(1 for _ in c._prompt_refs()))",
                cwd=proj)
        check("shipped config validates", r.returncode == 0,
              f"{r.stdout.strip() or r.stderr.strip()[:60]} prompt refs resolved, shas matched")

        # 4. SUITE
        r = run(py, "-m", "pytest", "--pyargs", "rlm", "-q", "--no-header",
                "-p", "no:cacheprovider", cwd=proj)
        tail = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
        summary = tail[-1].strip() if tail else r.stdout.strip()[-80:]
        check("suite runs with zero edits", r.returncode == 0 and "failed" not in summary, summary)
        check("no skip hides a missing file", " skipped" not in summary,
              "a skip here would mean a repo file is being reached for" if " skipped" in summary
              else "zero skips")

        # 5. WHEEL
        print("Building a wheel and installing it where the repo has never been...")
        wpy = make_venv(tmp / "wvenv", extra=["build"])
        r = run(wpy, "-m", "build", "--wheel", "--outdir", str(tmp / "dist"), cwd=REPO)
        whl = next((tmp / "dist").glob("*.whl"), None) if r.returncode == 0 else None
        if whl is None:
            check("wheel builds", False, r.stderr.strip()[-90:])
        else:
            data = [n for n in _names(whl) if not n.endswith(".py") and ".dist-info" not in n]
            ipy = make_venv(tmp / "ivenv", extra=[str(whl)])
            r = run(ipy, "-m", "pytest", "--pyargs", "rlm", "-q", "--no-header",
                    "-p", "no:cacheprovider", cwd=tmp)
            tail = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
            check("wheel ships its data", len(data) >= 20,
                  f"{len(data)} non-.py files: config, {sum('prompts/' in d for d in data)} prompts, schema.sql")
            check("wheel suite runs", r.returncode == 0,
                  tail[-1].strip() if tail else r.stdout.strip()[-80:])

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All distribution checks passed.")
    return 0


def _names(whl: Path) -> list[str]:
    import zipfile
    with zipfile.ZipFile(whl) as z:
        return z.namelist()


if __name__ == "__main__":
    raise SystemExit(main())
