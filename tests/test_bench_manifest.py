"""The built benchmark, checked against §8 as a whole artifact.

`bench/manifest.py::validate` encodes §8's authoring rules; these tests run it
against the benchmark actually on disk and check the properties a manifest
cannot check about itself -- that every task file loads through the shipped
loader, and that every checker accepts its own task's answer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bench.manifest import CATEGORY_SPLIT, N_TASKS, BenchmarkManifest
from rlm.episode import Task

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "bench" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="benchmark not built yet (python -m bench.build)")


@pytest.fixture(scope="module")
def manifest() -> BenchmarkManifest:
    return BenchmarkManifest.load(MANIFEST)


def test_the_manifest_validates_against_section_8(manifest):
    """Everything §8 requires of the SET of tasks: the pre-registered count and
    per-category split, >=1 adversarial, a known checker with >=3 near-misses
    per task, a window count on every aggregation task inside max_subcalls, and
    both aggregation rules (>=1 regex-solvable, >=1 verified at chance)."""
    manifest.validate(require_closed_book=False)


def test_every_task_file_loads_through_the_shipped_loader(manifest):
    """The reason task files stay in `Task.from_file`'s key set: any one of the
    30 can be run with `rlm run bench/tasks/<id>.json` and no special harness.
    A task the shipped loader refuses is a task S4 cannot run."""
    for e in manifest.tasks:
        t = Task.from_file(REPO / e.task_file)
        assert t.task_id == e.task_id
        assert t.category == e.category
        assert t.checker == e.checker


def test_every_checker_accepts_its_own_tasks_answer(manifest):
    """A checker that rejects the ground truth would fail every arm on that
    task and the per-category table would report a category-wide zero that
    means nothing about the roots."""
    for e in manifest.tasks:
        t = Task.from_file(REPO / e.task_file)
        assert t.check(t.answer), (
            f"{e.task_id}: checker {e.checker!r} rejects its own answer "
            f"{t.answer!r}")


def test_the_corpora_on_disk_match_their_recorded_hashes(manifest):
    """The freeze rests on these hashes. A corpus edited after the manifest was
    written would silently change what a task measures."""
    import hashlib
    for e in manifest.tasks:
        text = (REPO / e.corpus_path).read_text(encoding="utf-8")
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert got == e.corpus_sha256, f"{e.task_id}: {e.corpus_path} changed"


def test_the_split_is_the_pre_registered_one(manifest):
    counts: dict[str, int] = {}
    for e in manifest.tasks:
        counts[e.category] = counts.get(e.category, 0) + 1
    assert counts == CATEGORY_SPLIT
    assert len(manifest.tasks) == N_TASKS


def test_a_broken_manifest_is_rejected_rather_than_reported(manifest):
    """validate() must FAIL loudly. Checked by breaking a copy: silent tolerance
    here would let a benchmark that violates §8 be frozen and run."""
    import copy
    m = copy.deepcopy(manifest)
    m.tasks = m.tasks[:-1]
    with pytest.raises(AssertionError, match="tasks, pre-registered N"):
        m.validate()

    m2 = copy.deepcopy(manifest)
    for t in m2.tasks:
        if t.category == "aggregation" and t.regex_solvable is False:
            t.regex_at_chance = {"__chance__": 0.55, "trivial": 0.99}
            break
    with pytest.raises(AssertionError, match="not regex-defeating"):
        m2.validate()
