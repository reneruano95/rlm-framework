"""Fixture suite for the gate's screens (plan step 3 verification).

Every positive fixture is material already committed to this repo -- the memories
the local root actually wrote during the spike -- so the suite tests the screens
against the thing they exist for rather than against inventions.

Run: python bench/test_screens.py
"""

from __future__ import annotations

import json
import pathlib
import sys

# See screens.py: walk up, do not count. This file moved with it.
from rlm.config import find_repo_root  # noqa: E402
REPO = find_repo_root(pathlib.Path(__file__))
sys.path.insert(0, str(REPO))

from bench.screens import Screens  # noqa: E402

SPIKE = REPO / "docs" / "research" / "2026-08-26-prime-agent-spike" / "results" / "phase-b" / "harness"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def entry(**kw) -> dict:
    base = {"id": "cand", "kind": "prompt", "title": "t", "content": "", "path": "00-gate/prompt/00"}
    base.update(kw)
    return base


def main() -> int:
    s = Screens.build(REPO)
    held = [r["task_id"] for r in s.held_out]
    print(f"screens built over {len(held)} held-out tasks: {', '.join(held)}\n")

    # ---- S-kind --------------------------------------------------------------
    check("S-kind accepts prompt", s.s_kind(entry(kind="prompt")).passed)
    check("S-kind accepts skill", s.s_kind(entry(kind="skill")).passed)
    check("S-kind rejects memory", not s.s_kind(entry(kind="memory")).passed)
    check("S-kind rejects subagent (a route, per I1)", not s.s_kind(entry(kind="subagent")).passed)

    # ---- S-I1 ----------------------------------------------------------------
    blunt = entry(content="If the run is slow, raise max_wall_clock_s to 3000 and retry.")
    r = s.s_i1(blunt)
    check("S-I1 rejects a named budget key", not r.passed, r.detail)
    check("S-I1 rejects a named route key", not s.s_i1(entry(content="set slot_policy to reuse")).passed)
    soft = entry(content="Allow yourself more time before giving up on a hard count.")
    check("S-I1 PASSES a paraphrase (the documented blind spot)", s.s_i1(soft).passed,
          "pinned so it can never be mistaken for coverage")
    check("S-I1 accepts ordinary procedure text",
          s.s_i1(entry(content="Cross-check the count with a second, independently written pattern.")).passed)
    check("S-I1 does not fire on a file path that happens to be a config word",
          s.s_i1(entry(content="Exactly one `def in_flight` is in file rlm/dispatcher.py.")).passed,
          "regression: bare `dispatcher` used to trip on the code-QA corpus")

    # ---- S-answer ------------------------------------------------------------
    # Answers are read from the SPLIT, never hardcoded: an earlier version of this
    # suite named codeqa-06 as held-out, the split was redrawn under it, and the test
    # then asserted something false. The split is the source of truth.
    def held_answer(prefix: str) -> tuple[str, str]:
        for row in s.held_out:
            if row["task_id"].startswith(prefix):
                t = json.loads((REPO / "bench" / "tasks" / f"{row['task_id']}.json").read_text(encoding="utf-8"))
                return row["task_id"], t["answer"]
        raise AssertionError(f"no held-out task starting with {prefix}")

    t_code, a_code = held_answer("codeqa")
    t_needle, a_needle05 = held_answer("needle")
    t_agg, a_agg07 = held_answer("agg")

    r = s.s_answer(entry(content=f"The file that defines it is {a_code}."))
    check(f"S-answer rejects a held-out code-QA answer ({t_code})", not r.passed, r.detail)
    r = s.s_answer(entry(content=f"The custody key is {a_needle05}."))
    check("S-answer rejects a held-out needle answer", not r.passed, r.detail)
    r = s.s_answer(entry(content=f"WITHHELD comes to {a_agg07} on that register."))
    check("S-answer rejects a held-out aggregation answer (an integer)", not r.passed, r.detail)
    check("S-answer accepts procedure with no answers",
          s.s_answer(entry(content="Count with re.finditer over the whole text, then verify.")).passed)
    check("S-answer does not fire on a substring of a larger number",
          s.s_answer(entry(content=f"There were {a_agg07}00 records scanned in total.")).passed,
          "token-boundary test")

    # ---- S-corpus ------------------------------------------------------------
    r = s.s_corpus(entry(content=f"Look for the key {a_needle05} in the register."))
    check("S-corpus rejects a held-out corpus identifier (UUID)", not r.passed, r.detail)
    r = s.s_corpus(entry(content=f"WITHHELD comes to {a_agg07} on that register."))
    check("S-corpus PASSES an integer answer (the documented blind spot)", r.passed,
          "integers are not identifier-shaped — this is why S-answer exists")
    check("S-corpus accepts procedure text",
          s.s_corpus(entry(content="Split on blank lines and match each record's header.")).passed)

    # ---- the real spike artifacts -------------------------------------------
    print()
    for name in ("archived-global-before-C-1787841505.json", "archived-global-before-codeqa-1787840797.json"):
        path = SPIKE / name
        if not path.exists():
            check(f"fixture present: {name}", False, "missing")
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        verdicts = s.check_state(state)
        n = len(verdicts)
        kind_fails = [v for v in verdicts if "S-kind" in v.failed]
        answer_fails = [v for v in verdicts if "S-answer" in v.failed]
        corpus_fails = [v for v in verdicts if "S-corpus" in v.failed]
        print(f"  {name}: {n} entries")
        check(f"  every entry trips S-kind ({name[:22]}…)", len(kind_fails) == n,
              f"{len(kind_fails)}/{n} — all are `memory`, which v0 does not learn")
        # PROVENANCE IS RELATIVE TO THE SPLIT, and this fixture proves it. These
        # memories were derived from the SPIKE's train tasks and were legal there.
        # Under v0's split, codeqa-04 moved to the held-out side, so the entry that
        # records `def subcalls_used ... rlm/budget.py` now names a held-out answer and
        # S-answer rejects it. That is the screen working, not a false positive, and it
        # is why the split is frozen and sha-pinned before any artifact exists.
        for v in answer_fails:
            print(f"       S-answer: {v.candidate_id}")
        named = {tid for v in answer_fails for r in v.results
                 if r.name == "S-answer" and not r.passed
                 for tid in held if tid in r.detail}
        check(f"  S-answer fires only on v0 held-out tasks ({name[:22]}…)",
              bool(answer_fails) and named <= set(held),
              f"{len(answer_fails)}/{n} tripped, naming {sorted(named)}")
        check(f"  none trips S-corpus ({name[:22]}…)", not corpus_fails,
              f"{len(corpus_fails)} tripped" if corpus_fails else "no held-out identifiers")

    print()
    print("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
