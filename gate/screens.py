"""The gate's mechanical screens (spec §3.4, plan step 3).

A candidate artifact that fails ANY screen is rejected before it costs a single
held-out episode. Every screen is a TEST, not a judgement: deterministic, zero
model calls, and each one states the thing it cannot see.

WHAT THESE ARE AND ARE NOT FOR. They are cheap structural filters, and they are
not evidence that an artifact is any good. A report that cites a screen pass as
evidence of artifact quality has misread this file: what says an artifact helps is
the gate, and the spike measured the local root's own memories making a held-out
task worse while passing every provenance test that applied to them at the time.

PROVENANCE IS RELATIVE TO A SPLIT, and the fixture suite demonstrates it. The
memories committed under docs/research/2026-08-26-prime-agent-spike/ were derived
from the SPIKE's train tasks and were legal there. Screened against v0's split,
two of them trip S-answer -- one names `rlm/budget.py` (codeqa-04) and one names
`589` (agg-06), both of which v0 moved to the held-out side. Neither entry
changed; the split did. That is why the split is frozen and sha-pinned before any
artifact exists, and why re-drawing a split invalidates every artifact derived
under the old one.

  S-kind    kind in {prompt, skill}.  `subagent` says WHEN TO INVOKE a delegation,
            which is a route, and I1 says routes are never writable by an artifact.
            `memory` is where the spike's answer-memorisation lived and is deferred,
            not forbidden (spec §3.2 / D-S2).

  S-I1      no budget / cap / route / termination key name appears in the content.
            The key set is GENERATED from src/rlm/config.py's own schema, so a new
            budget key cannot silently escape it.

  S-answer  no held-out task's expected answer appears in the content, normalised by
            that task's own checker.

  S-corpus  no identifier-shaped token from a HELD-OUT corpus appears in the content.
            Reuses R13's detector (rlm.serve.leakcheck) with the held-out corpora as
            the index.

Usage:
    from gate.screens import Screens
    s = Screens.build(repo, split)          # indexes the held-out side once
    verdict = s.check(candidate)            # ScreenVerdict
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import sys
from typing import Any, Iterable, Sequence

REPO_DEFAULT = pathlib.Path(__file__).resolve().parent.parent

ALLOWED_KINDS = frozenset({"prompt", "skill"})

#: Models in `src/rlm/config.py` whose fields are budgets, caps, routes or
#: termination rules -- the surface I1 says no artifact may write. Field names are
#: read from the source rather than typed here, so adding a budget to the schema
#: extends this screen automatically.
_I1_MODELS = ("Budgets", "ChunkCfg")
#: Fields on ScaffoldCfg that are caps/routes rather than sub-configs.
_I1_SCAFFOLD_FIELDS = frozenset({
    "truncation_cap_chars", "root_window_kill_fraction", "dispatch_concurrency",
    "dispatcher", "retries",
})
#: Names that are budgets/routes but live outside those models, or are the words a
#: config-shaped instruction would use. Kept short and explicit; the blind spot
#: below is the honest limit, not this list's length.
_I1_EXTRA = frozenset({
    "slot_policy", "never_reuse", "max_predict", "per_call_timeout_s",
    "n_parallel", "max_turns", "max_continuations", "autonomous_max_tokens",
    "rlm_max_depth", "max_identical_turns",
})


def _i1_keywords(repo: pathlib.Path) -> frozenset[str]:
    """Budget/cap/route/termination identifiers, generated from the config schema."""
    src = (repo / "src" / "rlm" / "config.py").read_text(encoding="utf-8")
    names: set[str] = set(_I1_EXTRA) | set(_I1_SCAFFOLD_FIELDS)
    for cls in _I1_MODELS:
        m = re.search(rf"class {cls}\b.*?(?=\nclass |\Z)", src, re.S)
        if not m:
            raise RuntimeError(
                f"gate.screens: class {cls} not found in config.py -- the I1 key set is "
                "generated from the schema and cannot silently fall back to a stale literal."
            )
        names.update(re.findall(r"^\s{4}([a-z_][a-z0-9_]*)\s*:", m.group(0), re.M))
    # Only multi-word identifiers survive. Single words are too generic: `dispatcher`
    # fired on `rlm/dispatcher.py`, an ordinary file path in the code-QA corpus, on
    # this screen's first run against real material. A contamination screen should err
    # toward rejecting; a CONFIG screen that gates real work should not, because every
    # false positive discards an artifact for naming a file. The blunt case this gives
    # up ("set dispatcher to mock") is the price, and `dispatch_concurrency`,
    # `max_wall_clock_s` and the rest of the generated set still fire.
    return frozenset(n for n in names if "_" in n)


@dataclasses.dataclass(frozen=True)
class ScreenResult:
    name: str
    passed: bool
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class ScreenVerdict:
    candidate_id: str
    passed: bool
    results: tuple[ScreenResult, ...]

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.results if not r.passed)

    def as_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "passed": self.passed,
                "failed": list(self.failed),
                "results": [dataclasses.asdict(r) for r in self.results]}


class Screens:
    """The four screens, with the held-out side indexed once."""

    def __init__(self, *, repo: pathlib.Path, held_out: Sequence[dict],
                 answers: dict[str, tuple[str, str]], corpus_index: Any,
                 i1_keywords: frozenset[str], corpus_chunk_count: int) -> None:
        self.repo = repo
        self.held_out = list(held_out)
        self._answers = answers            # task_id -> (checker, answer)
        self._index = corpus_index         # leakcheck.ChunkIndex over held-out corpora
        self._i1 = i1_keywords
        self._corpus_chunk_count = corpus_chunk_count

    # -- construction ----------------------------------------------------------
    @classmethod
    def build(cls, repo: pathlib.Path | str = REPO_DEFAULT,
              split_path: pathlib.Path | str | None = None) -> "Screens":
        repo = pathlib.Path(repo)
        split_path = pathlib.Path(split_path or repo / "bench" / "splits" / "s6lite-v0.json")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        held = split["held_out"]

        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))
        from rlm.serve.leakcheck import ChunkIndex  # noqa: E402

        answers: dict[str, tuple[str, str]] = {}
        corpora: dict[str, str] = {}
        for row in held:
            tid = row["task_id"]
            task = json.loads((repo / "bench" / "tasks" / f"{tid}.json").read_text(encoding="utf-8"))
            answers[tid] = (task["checker"], task["answer"])
            corpus = (repo / "bench" / "tasks" / task["context_path"]).resolve()
            corpora[tid] = corpus.read_text(encoding="utf-8", errors="replace")

        index = ChunkIndex.from_chunks(corpora)
        return cls(repo=repo, held_out=held, answers=answers, corpus_index=index,
                   i1_keywords=_i1_keywords(repo), corpus_chunk_count=len(corpora))

    # -- the screens -----------------------------------------------------------
    def s_kind(self, cand: dict) -> ScreenResult:
        """kind in {prompt, skill}.

        BLIND SPOT: none -- this one is total. It is also the only screen that
        rejects on shape rather than content, so it is the one a candidate set
        converted from prime-agent's own output will trip first.
        """
        kind = cand.get("kind")
        ok = kind in ALLOWED_KINDS
        return ScreenResult("S-kind", ok,
                            "" if ok else f"kind={kind!r} not in {sorted(ALLOWED_KINDS)}")

    def s_i1(self, cand: dict) -> ScreenResult:
        """No budget/cap/route/termination identifier in the content.

        BLIND SPOT, and it is a property of the method rather than a gap to close
        later: this matches IDENTIFIERS, so a paraphrase names nothing and passes.
        "Allow yourself more time before giving up" is not caught and cannot be.
        The real enforcement is elsewhere -- an artifact is prompt text and budgets
        live in config.yaml and scaffold code (I1's actual mechanism), so a
        persuasive artifact can ask the model to behave differently but cannot raise
        a cap. This screen catches the blunt case and says so.
        """
        text = f"{cand.get('title', '')}\n{cand.get('content', '')}".lower()
        hits = sorted(k for k in self._i1 if k in text)
        return ScreenResult("S-I1", not hits,
                            "" if not hits else f"names config identifiers: {hits}")

    def s_answer(self, cand: dict) -> ScreenResult:
        """No held-out task's expected answer in the content.

        Normalisation is each task's OWN checker, so "514" and "the count is 514"
        are the same claim, exactly as scoring treats them.

        BLIND SPOT: a TRAIN answer passes, and correctly so -- artifacts are
        train-derived by design. This screen protects the evaluation set, not the
        artifact's usefulness.
        """
        if str(self.repo / "src") not in sys.path:
            sys.path.insert(0, str(self.repo / "src"))
        from rlm.measure.checkers import check  # noqa: E402

        text = f"{cand.get('title', '')}\n{cand.get('content', '')}"
        hits: list[str] = []
        for tid, (checker, answer) in self._answers.items():
            # The checker decides equality; feed it the candidate text as the "got".
            # A checker that demands a UNIQUE match (uuid_exact, int_exact) will not
            # fire on prose that merely contains other numbers, so also test the bare
            # answer as a substring, normalised the same way.
            try:
                if check(checker, text, answer):
                    hits.append(tid)
                    continue
            except Exception:
                pass
            if _normalised_contains(text, answer):
                hits.append(tid)
        return ScreenResult("S-answer", not hits,
                            "" if not hits else f"contains held-out answers for: {sorted(hits)}")

    def s_corpus(self, cand: dict) -> ScreenResult:
        """No identifier-shaped token from a held-out corpus in the content.

        Reuses R13's detector, whose two documented limits carry over unchanged --
        paraphrased content is invisible, and a contaminated refusal is invisible.

        AND ONE MORE THAT MATTERS HERE: INTEGERS ARE NOT IDENTIFIER-SHAPED. The
        detector's pattern set is UUIDs, ENT-##### codes, hex runs and long mixed
        alphanumeric tokens -- by design, so ordinary English stays out. So the
        aggregation answers `514` and `589` pass this screen. That is precisely why
        S-answer exists as a separate test, and why neither screen replaces the gate.
        """
        text = f"{cand.get('title', '')}\n{cand.get('content', '')}"
        verdict = self._index.foreign(text, sent="")
        if verdict.detected is None:
            return ScreenResult("S-corpus", False,
                                "NOT CHECKED: no held-out corpus indexed -- refusing to "
                                "report a clean screen that never ran")
        return ScreenResult("S-corpus", not verdict.detected, verdict.detail or "")

    # -- the whole set ---------------------------------------------------------
    def check(self, cand: dict) -> ScreenVerdict:
        results = (self.s_kind(cand), self.s_i1(cand), self.s_answer(cand), self.s_corpus(cand))
        return ScreenVerdict(candidate_id=str(cand.get("id", "<unnamed>")),
                             passed=all(r.passed for r in results), results=results)

    def check_state(self, state: dict) -> list[ScreenVerdict]:
        """Screen every entry of a HarnessState-shaped candidate set."""
        out: list[ScreenVerdict] = []
        for kind, entries in (state.get("entries") or {}).items():
            for entry in (entries or {}).values():
                cand = dict(entry)
                cand.setdefault("kind", kind)
                out.append(self.check(cand))
        return out


def _normalised_contains(text: str, answer: str) -> bool:
    """Whitespace-collapsed, case-folded substring test, with token-like answers
    required to sit on a boundary so `514` does not match inside `1514`.

    The boundaries are ASYMMETRIC on purpose. Leading: `.` and `/` count, so a path
    or number embedded in a longer one does not match. Trailing: they do NOT, because
    an answer at the end of a sentence is followed by a full stop -- `rlm/budget.py.`
    must still match. Measured 2026-08-27: a symmetric lookahead silently passed a
    candidate that named a held-out answer, which for a contamination screen is the
    wrong direction to err in.
    """
    t = re.sub(r"\s+", " ", text).casefold()
    a = re.sub(r"\s+", " ", answer).strip().casefold()
    if not a:
        return False
    if re.fullmatch(r"[\w./-]+", a):
        return re.search(rf"(?<![\w./-]){re.escape(a)}(?![\w-])", t) is not None
    return a in t


def _iter_entries(state: dict) -> Iterable[dict]:
    for kind, entries in (state.get("entries") or {}).items():
        for entry in (entries or {}).values():
            cand = dict(entry)
            cand.setdefault("kind", kind)
            yield cand


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Screen a candidate artifact set.")
    ap.add_argument("state", help="a harness_state.json-shaped candidate set")
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--split", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    screens = Screens.build(args.repo, args.split)
    state = json.loads(pathlib.Path(args.state).read_text(encoding="utf-8"))
    verdicts = screens.check_state(state)
    if args.json:
        print(json.dumps([v.as_dict() for v in verdicts], indent=2))
    else:
        for v in verdicts:
            print(f"{'PASS' if v.passed else 'FAIL'}  {v.candidate_id}")
            for r in v.results:
                if not r.passed:
                    print(f"       {r.name}: {r.detail}")
    return 0 if all(v.passed for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
