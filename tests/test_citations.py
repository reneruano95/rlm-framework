"""Every `milestones/...` path cited in the tree must be retrievable from git.

`milestones/` was deleted from the working tree on 2026-08-22. The ~176 citations
to it in `ARCHITECTURE.md`, `CHANGELOG.md`, `config.yaml` comments and `src/rlm`
docstrings were NOT stripped, and that is deliberate: they are the evidence trail
behind every gate verdict, and a spec that stops saying where its numbers came
from is worse than one whose citations need a `git show`.

So the invariant changed rather than disappeared. It used to be "this path exists
on disk". It is now:

    every cited milestones/... path must exist in commit ARCHIVE_COMMIT

which keeps the trail checkable instead of letting it rot silently. To read one:

    git show 4e75b53:milestones/s2/R13.md

WHY THIS STILL MATTERS. Gate verdicts cite evidence by path: ARCHITECTURE.md §9,
DIRECTION.md's customer-facing R13 bound, and ~30 `config.yaml` comments that
justify a shipped number by pointing at the measurement behind it --
`config.yaml:542`'s "LIVE BLOCKER ... THIS VALUE IS NOT SAFE" is an interlock
whose entire evidence is a citation. A dangling citation is not a broken link. It
is a claim that can no longer be checked, in a project whose argument for its own
numbers is that they can be.

The original version of this test caught a real defect the day it was written:
`run_thinking_ab.py` had been deleted with three surviving citations and nothing
noticed.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The last commit in which `milestones/` was present in the working tree.
#: Everything cited must be retrievable from here. If history is ever rewritten
#: such that this sha is unreachable, this test fails loudly -- which is correct:
#: at that point the citations really have become unresolvable.
ARCHIVE_COMMIT = "4e75b531a5e1fadaac31c28f7ce2ba6ac89f8243"

SKIP_DIRS = {".git", ".venv", "traces", "tools", "__pycache__", "sandbox_bootstrap", "runs"}
EXTS = {".md", ".py", ".yaml", ".toml"}

TOKEN = re.compile(r"\bmilestones/[A-Za-z0-9_][A-Za-z0-9_./-]*")

#: Paths a script names as its own `--out` destination. They never existed on a
#: clean checkout and do not exist in the archive commit either.
WRITE_TARGETS = {
    "milestones/s2/results/r13.jsonl",
    "milestones/s2/results/r13_repro.jsonl",
    "milestones/s2/audit/_refute_chunk_index.json",
}

#: Deleted before the archive commit, and cited by documents that are accurate
#: about the past. Listed rather than silently tolerated.
DELETED = {
    "milestones/s1/run_thinking_ab.py",
}


def _archive_paths() -> set[str]:
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ARCHIVE_COMMIT, "milestones/"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.fail(f"archive commit {ARCHIVE_COMMIT[:7]} is unreachable: "
                    f"{out.stderr.strip()}. The evidence trail cannot be checked.")
    return set(out.stdout.split())


def _cited_paths() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.name == "test_citations.py":
            continue
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for match in TOKEN.finditer(line):
                token = match.group(0).rstrip(".,;:)`'\"")
                found.setdefault(token, []).append(f"{rel.as_posix()}:{lineno}")
    return found


def _resolves(token: str, archive: set[str]) -> bool:
    if "*" in token or "<" in token or "{" in token:
        return True                                  # a pattern, not a path
    if token.endswith(("-", "_")):
        return True                                  # truncated before a variable
    if token in archive or token + ".py" in archive:
        return True                                  # prose may name a module
    if any(p.startswith(token + "/") for p in archive):
        return True                                  # a directory
    if token in WRITE_TARGETS or token in DELETED:
        return True
    return not Path(token).suffix                    # extensionless prose


def test_every_cited_milestone_path_is_retrievable_from_the_archive_commit():
    archive = _archive_paths()
    assert archive, f"{ARCHIVE_COMMIT[:7]} holds no milestones/ tree"
    cited = _cited_paths()
    assert cited, "found no milestone citations -- the scanner is broken"
    dangling = {t: s for t, s in cited.items() if not _resolves(t, archive)}
    assert not dangling, (
        f"cited but absent from {ARCHIVE_COMMIT[:7]} (read one with "
        f"`git show {ARCHIVE_COMMIT[:7]}:<path>`):\n" + "\n".join(
            f"  {t}\n      " + "\n      ".join(sites[:4])
            for t, sites in sorted(dangling.items())))


def test_the_archive_commit_is_reachable():
    """The whole trail hangs off one sha. Say so loudly if it ever goes away."""
    out = subprocess.run(["git", "cat-file", "-e", f"{ARCHIVE_COMMIT}^{{commit}}"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, (
        f"{ARCHIVE_COMMIT} is unreachable. Every milestones/ citation in the tree "
        "now points at nothing retrievable -- restore the commit, or strip the "
        "citations and admit the evidence is gone.")


def test_deleted_list_does_not_rot():
    """A path listed as DELETED must really be absent from the archive commit."""
    archive = _archive_paths()
    present = sorted(p for p in DELETED if p in archive)
    assert not present, (
        f"listed as deleted but present in the archive commit: {present}. "
        "Remove them from DELETED -- they resolve on their own.")
