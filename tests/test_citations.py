"""Every `milestones/...` path cited in the tree must resolve.

WHY THIS EXISTS. Gate verdicts cite their evidence by path: ARCHITECTURE.md §9,
DIRECTION.md's customer-facing R13 bound, and ~30 comments in `config.yaml`
that justify a shipped number by pointing at the measurement behind it
(`config.yaml:542`'s "LIVE BLOCKER ... THIS VALUE IS NOT SAFE" is an interlock
whose whole evidence is a citation). A dangling citation is therefore not a
broken link -- it is a claim that can no longer be checked, in a project whose
argument for its own numbers is that they can be.

The repo had no mechanism for this and it cost something real: `run_thinking_ab.py`
was deleted on 2026-08-22 and three surviving citations were not noticed by
anything. The deletion was recorded by hand in a fourth file. This test is that
mechanism.

WHAT IT DELIBERATELY DOES NOT FLAG, because a checker that cries wolf gets
deleted:

  * prose naming a MODULE rather than a file -- "milestones/s1/make_fixtures";
    resolved by trying `+ ".py"`.
  * a token truncated before a template variable or glob -- `r13_mit_`,
    `fixtures-refusal-640-s`, `logs/arch-`. These end at the character where an
    f-string or `*` began, so they are prefixes, not paths.
  * paths with no extension that name neither a directory nor a module: prose.
  * declared OUTPUT paths (`WRITE_TARGETS`) -- a script's default `--out` names
    a file that will not exist until it runs, and demanding otherwise would
    make the test fail on a clean checkout.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".git", ".venv", "traces", "tools", "__pycache__", "sandbox_bootstrap"}
EXTS = {".md", ".py", ".yaml", ".toml"}

TOKEN = re.compile(r"\bmilestones/[A-Za-z0-9_][A-Za-z0-9_./-]*")

#: Default `--out` destinations. They name where a run WILL write, so they are
#: absent on a clean checkout by construction and their absence is not a defect.
WRITE_TARGETS = {
    "milestones/s2/results/r13.jsonl",              # r13_repro.py:71
    "milestones/s2/results/r13_repro.jsonl",        # r13_repro.py:993 (--out default)
    "milestones/s2/audit/_refute_chunk_index.json",  # audit_refute_corpus_index.py:45, generated
}

#: Files that existed when they were cited and were later deleted on purpose.
#: The citation stays because the RECORD is accurate -- the run happened, the
#: script existed. Listing it here is the honest form: the test states the fact
#: instead of the document implying the file is still there.
DELETED = {
    # Retired 2026-08-22 with `config-thinkon.yaml`; the harness could no longer
    # answer its own question honestly. Closure: milestones/s2/ROOT-THINKING.md.
    "milestones/s1/run_thinking_ab.py",
}


def _cited_paths() -> dict[str, list[str]]:
    """Every milestone path token in the tree -> where it is cited."""
    found: dict[str, list[str]] = {}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.name == "test_citations.py":
            continue                                     # this file names them on purpose
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in TOKEN.finditer(line):
                token = match.group(0).rstrip(".,;:)`'\"")
                found.setdefault(token, []).append(f"{rel.as_posix()}:{lineno}")
    return found


def _resolves(token: str) -> bool:
    if "*" in token or "<" in token or "{" in token:
        return True                                      # a pattern, not a path
    if token.endswith(("-", "_")):
        return True                                      # truncated before a variable
    if (REPO_ROOT / token).exists():
        return True
    if (REPO_ROOT / (token + ".py")).exists():
        return True                                      # prose naming a module
    if token in WRITE_TARGETS or token in DELETED:
        return True
    # No extension and nothing on disk: prose, not a citation.
    return not Path(token).suffix


def test_every_cited_milestone_path_resolves():
    cited = _cited_paths()
    assert cited, "found no milestone citations at all -- the scanner is broken"
    dangling = {t: s for t, s in cited.items() if not _resolves(t)}
    assert not dangling, "cited but missing:\n" + "\n".join(
        f"  {t}\n      " + "\n      ".join(sites[:4])
        for t, sites in sorted(dangling.items()))


def test_deleted_list_does_not_rot():
    """A path listed as DELETED must actually be gone.

    Without this, the escape hatch silently becomes a way to keep a stale entry
    forever -- and worse, to suppress a genuine dangling citation if the file is
    ever restored under the same name.
    """
    resurrected = [p for p in DELETED if (REPO_ROOT / p).exists()]
    assert not resurrected, (
        f"listed as deleted but present: {resurrected}. Remove them from DELETED "
        "-- the citations resolve on their own now.")


def test_write_targets_are_named_by_something():
    """A WRITE_TARGETS entry nothing cites is a stale suppression."""
    cited = set(_cited_paths())
    orphans = [p for p in WRITE_TARGETS if p not in cited]
    assert not orphans, (
        f"WRITE_TARGETS entries nothing references: {orphans}. Drop them.")
