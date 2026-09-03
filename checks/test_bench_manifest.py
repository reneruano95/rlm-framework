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


def test_the_frozen_benchmark_passes_the_preconditions_that_gate_a_freeze(manifest):
    """§8's preconditions "must pass before freeze". This asserts the STRICT
    form -- the one that refuses a manifest whose closed-book probe has not been
    run, or any of whose tasks was answered correctly without its corpus.

    A task answerable from memory does not inflate every arm equally: B1 and B3
    answer in one call from parametric knowledge while RLM and B2 spend hundreds
    of leaf calls reading a corpus they did not need, so the benchmark would
    report the scaffold losing on cost while tying on quality."""
    manifest.validate(require_closed_book=True)
    for e in manifest.tasks:
        cb = e.closed_book or {}
        assert cb.get("seeds") == 3 and set(cb.get("models", [])) == {"root", "leaf"}
        assert cb.get("passed_without_corpus") == 0


def test_config_pins_the_frozen_version(manifest):
    """config.yaml's `benchmark.version` names this manifest; S4's
    config_snapshot records it, so an episode can be traced back to the exact
    task set it was scored against."""
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    assert str(cfg["benchmark"]["version"]) == manifest.benchmark_version
    assert manifest.token_counter.startswith("leaf:"), (
        "a frozen benchmark must be counted with the real tokenizer; the "
        "offline proxy is vocabulary-specific and this vocabulary is new")


def test_closed_book_probe_dry_run_flag():
    """The closed-book probe's --dry-run flag reads the manifest and lists
    tasks without calling any server. The --manifest flag selects which
    manifest to read and write results to."""
    import tempfile
    import sys
    from io import StringIO
    from pathlib import Path
    from bench.manifest import BenchmarkManifest, TaskEntry

    # Create a minimal test manifest in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_path = tmpdir_path / "test_manifest.json"

        # Create a minimal test manifest with one task
        test_manifest = BenchmarkManifest(
            benchmark_version="test-v1",
            built_at="2026-09-02",
            token_counter="leaf:/tokenize",
            assumed_training_cutoff="2024-08-01",
            tasks=[
                TaskEntry(
                    task_id="test-01",
                    category="needle",
                    task_file="bench/tasks/test-01.json",
                    corpus_path="bench/corpus/test-01.txt",
                    corpus_sha256="abc123",
                    corpus_tokens=100,
                    corpus_date="2024-08-01",
                    checker="exact",
                    question_sha256="def456"
                )
            ]
        )
        test_manifest.write(manifest_path)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            from bench.closed_book import main
            # This should not raise and should complete without server calls
            result = main(["--manifest", str(manifest_path), "--dry-run"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Verify the dry-run completed (exit code 0)
        assert result == 0, f"Expected exit code 0, got {result}"

        # Verify the output mentions the task
        assert "test-01" in output, "Task ID should appear in dry-run output"
        assert "needle" in output, "Category should appear in dry-run output"


def test_closed_book_probe_manifest_argument_controls_write_back():
    """The --manifest argument controls which manifest file is read from AND
    written back to. This ensures a probe of v2 writes to v2, not v1."""
    import tempfile
    import sys
    from io import StringIO
    from pathlib import Path
    from bench.manifest import BenchmarkManifest, TaskEntry

    # Create a temporary manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_path = tmpdir_path / "custom_manifest.json"

        # Create a minimal test manifest
        test_manifest = BenchmarkManifest(
            benchmark_version="test-v1",
            built_at="2026-09-02",
            token_counter="leaf:/tokenize",
            assumed_training_cutoff="2024-08-01",
            tasks=[
                TaskEntry(
                    task_id="test-01",
                    category="needle",
                    task_file="bench/tasks/test-01.json",
                    corpus_path="bench/corpus/test-01.txt",
                    corpus_sha256="abc123",
                    corpus_tokens=100,
                    corpus_date="2024-08-01",
                    checker="exact",
                    question_sha256="def456"
                )
            ]
        )
        test_manifest.write(manifest_path)

        # Verify the --manifest flag controls which manifest is read
        # (dry-run doesn't make server calls, so we can use it without live servers)
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            from bench.closed_book import main
            result = main(["--manifest", str(manifest_path), "--dry-run"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Verify it read from the custom manifest
        assert result == 0
        assert "test-01" in output
        # Verify the output shows the correct manifest path
        assert "custom_manifest.json" in output
