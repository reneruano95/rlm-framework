"""The frozen benchmark's manifest: everything §8 demands that a task file
cannot carry.

WHY A SIDECAR AND NOT A RICHER TASK FILE. Three options were open: put the
authoring metadata in the task file (and loosen `Task.from_file`), build Task
objects in code from a manifest (what s1 does), or keep task files in the
existing key set and put the metadata beside them. The spec already chose:
§6 defines `task_hash` as "sha256 of the instruction text (corpus documents
hashed separately, **in the benchmark manifest referenced by
`benchmark_version`**)". So the manifest is not an invention here, it is the
artifact §6 already refers to.

The practical consequences all point the same way:

  * `rlm run bench/tasks/agg-01.json` keeps working, so a single benchmark task
    can be debugged with the shipped CLI and no special runner.
  * `Task.from_file`'s closed key set stays closed. It refuses unknown keys on
    purpose -- a task file is a portable artifact and a typo'd key silently
    ignored is a task that measures something other than what it says.
  * The freeze hashes one file, not thirty, and §8's preconditions
    (closed-book probe, near-miss suites, corpus dating, window counts) live
    where they can be checked as a set rather than task by task.

VALIDATION IS THE POINT. `validate()` encodes §8's authoring rules as
assertions, so "the benchmark satisfies §8" is a thing the repo can prove on
demand instead of a claim in a report.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rlm.measure.checkers import CHECKERS, near_miss_suite

# §8's pre-registered split at N=30 (ARCHITECTURE.md, 2026-08-15).
CATEGORY_SPLIT = {"needle": 8, "aggregation": 7, "synthesis": 8, "code_qa": 7}
N_TASKS = sum(CATEGORY_SPLIT.values())
MIN_ADVERSARIAL = 1          # §8 floor; this benchmark carries 2
MIN_NEAR_MISSES = 3          # §8: ">=3 authored plausible-but-wrong answers"
MAX_AGG_SUBCALLS = 926       # config.yaml max_subcalls


@dataclass
class TaskEntry:
    """One benchmark task's authoring record. The RUNNABLE half of a task lives
    in its own JSON file in `Task.from_file`'s key set; this is everything §8
    requires that would not fit there."""
    task_id: str
    category: str
    task_file: str                  # repo-relative, the runnable artifact
    corpus_path: str
    corpus_sha256: str
    corpus_tokens: int
    corpus_date: str                # §8 "corpus dating" precondition
    checker: str
    question_sha256: str            # §6 task_hash = sha256 of instruction text
    adversarial: bool = False
    # Aggregation only. §8: "State each aggregation task's window count in the
    # benchmark manifest so the affordability claim is checkable rather than
    # assumed."
    windows: int | None = None
    subcalls: int | None = None
    regex_solvable: bool | None = None
    regex_at_chance: dict[str, float] | None = None
    # §8 precondition 1, filled by the closed-book probe (needs a GPU).
    closed_book: dict[str, object] | None = None


@dataclass
class BenchmarkManifest:
    benchmark_version: str
    built_at: str
    token_counter: str              # "approx-offline" or "leaf:/tokenize"
    assumed_training_cutoff: str
    tasks: list[TaskEntry] = field(default_factory=list)

    # ----------------------------------------------------------------- io --
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "BenchmarkManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tasks = [TaskEntry(**t) for t in raw.pop("tasks", [])]
        return cls(tasks=tasks, **raw)

    @property
    def sha256(self) -> str:
        """The freeze hash. One number that changes if any task, corpus hash,
        checker or precondition result changes."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    # --------------------------------------------------------- validation --
    def validate(self, *, require_closed_book: bool = False) -> None:
        """§8's authoring rules, as assertions. Raises on the first violation.

        `require_closed_book` is off while authoring and ON at freeze: the probe
        needs both models live, and §8 makes it a precondition that "must pass
        before freeze" rather than during it.
        """
        errs: list[str] = []

        if len(self.tasks) != N_TASKS:
            errs.append(f"{len(self.tasks)} tasks, pre-registered N is {N_TASKS}")

        ids = [t.task_id for t in self.tasks]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            errs.append(f"duplicate task_ids: {dupes}")

        counts: dict[str, int] = {}
        for t in self.tasks:
            counts[t.category] = counts.get(t.category, 0) + 1
        if counts != CATEGORY_SPLIT:
            errs.append(f"category split {counts} != pre-registered {CATEGORY_SPLIT}")

        n_adv = sum(1 for t in self.tasks if t.adversarial)
        if n_adv < MIN_ADVERSARIAL:
            errs.append(f"{n_adv} adversarial-context tasks, §8 requires "
                        f">= {MIN_ADVERSARIAL}")

        for t in self.tasks:
            if t.checker not in CHECKERS:
                errs.append(f"{t.task_id}: unknown checker {t.checker!r}")
                continue
            if len(near_miss_suite(t.checker)) < MIN_NEAR_MISSES:
                errs.append(f"{t.task_id}: checker {t.checker!r} ships fewer "
                            f"than {MIN_NEAR_MISSES} near-misses")

        agg = [t for t in self.tasks if t.category == "aggregation"]
        for t in agg:
            if t.windows is None or t.subcalls is None:
                errs.append(f"{t.task_id}: aggregation task must state its "
                            f"window count (§8)")
            elif t.subcalls > MAX_AGG_SUBCALLS:
                errs.append(f"{t.task_id}: {t.subcalls} sub-calls for full "
                            f"coverage, over max_subcalls {MAX_AGG_SUBCALLS}")
        # §8's two named aggregation rules.
        if not any(t.regex_solvable is True for t in agg):
            errs.append("no aggregation task is regex-solvable (§8 requires >=1)")
        defeating = [t for t in agg if t.regex_solvable is False]
        if not defeating:
            errs.append("no aggregation task defeats string matching "
                        "(§8 requires >=1)")
        for t in defeating:
            scores = dict(t.regex_at_chance or {})
            chance = scores.pop("__chance__", None)
            if chance is None or not scores:
                errs.append(f"{t.task_id}: claims to defeat string matching but "
                            f"ships no at-chance demonstration (§8 requires it "
                            f"be 'verified at authoring')")
                continue
            beaten = {k: v for k, v in scores.items() if v > chance + 0.02}
            if beaten:
                errs.append(f"{t.task_id}: regex {sorted(beaten)} beat chance "
                            f"{chance:.3f} -- not regex-defeating")

        if require_closed_book:
            missing = [t.task_id for t in self.tasks if not t.closed_book]
            if missing:
                errs.append(f"closed-book probe not run for: {missing}")
            for t in self.tasks:
                cb = t.closed_book or {}
                if cb.get("passed_without_corpus", 0):
                    errs.append(
                        f"{t.task_id}: answered correctly without the corpus in "
                        f"{cb['passed_without_corpus']}/3 seeds -- §8 requires "
                        f"it be rewritten or replaced")

        if errs:
            raise AssertionError("benchmark manifest violates §8:\n  - "
                                 + "\n  - ".join(errs))


__all__ = ["BenchmarkManifest", "TaskEntry", "CATEGORY_SPLIT", "N_TASKS"]
