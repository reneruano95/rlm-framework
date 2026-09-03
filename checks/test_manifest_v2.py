import json
from pathlib import Path

import pytest

from bench.manifest import BenchmarkManifest, TaskEntry

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "bench" / "manifest.json"
V1_PIN = "571918d24bd848b8cb4122b7226882d65b929aa4dff0cb80a578d2eb04603c91"

pytestmark = pytest.mark.skipif(not V1.exists(), reason="v1 manifest not built")


def test_v2_fields_leave_the_v1_sha_byte_identical():
    """Adding optional fields must not move the frozen v1 pin: None-valued v2
    fields are omitted from the serialized form, so v1's bytes are unchanged."""
    m = BenchmarkManifest.load(V1)
    assert m.sha256 == V1_PIN
    assert m.to_json() == V1.read_text(encoding="utf-8")


def test_v2_fields_round_trip_when_set(tmp_path):
    t = TaskEntry(task_id="ls-01-train", category="linear_semantic",
                  task_file="bench/tasks/v2/ls-01-train.json",
                  corpus_path="bench/corpora/v2/ls-01-train.txt",
                  corpus_sha256="0" * 64, corpus_tokens=60_000,
                  corpus_date="2026-09-02", checker="int_exact",
                  question_sha256="1" * 64, stream="train", shape_id="ls-01",
                  min_windows=139, label_source="CogComp/trec@sha256:abc")
    m = BenchmarkManifest(benchmark_version="v2", built_at="2026-09-02",
                          token_counter="leaf:/tokenize",
                          assumed_training_cutoff="2025-12-31", tasks=[t],
                          rules={"margin": 2})
    p = tmp_path / "m.json"
    m.write(p)
    back = BenchmarkManifest.load(p)
    assert back.tasks[0].stream == "train"
    assert back.tasks[0].min_windows == 139
    assert back.rules == {"margin": 2}
    assert json.loads(p.read_text())["tasks"][0]["shape_id"] == "ls-01"
    assert "interactive" not in json.loads(p.read_text())["tasks"][0]


def test_scored_tasks_is_the_scored_stream_or_everything():
    train = TaskEntry(task_id="a", category="c", task_file="", corpus_path="",
                      corpus_sha256="", corpus_tokens=1, corpus_date="",
                      checker="int_exact", question_sha256="", stream="train")
    held = TaskEntry(**{**train.__dict__, "task_id": "b", "stream": "held_out"})
    m = BenchmarkManifest(benchmark_version="v2", built_at="", token_counter="",
                          assumed_training_cutoff="", tasks=[train, held],
                          rules={"scored_stream": "train"})
    assert [t.task_id for t in m.scored_tasks()] == ["a"]
    v1like = BenchmarkManifest(benchmark_version="v1", built_at="", token_counter="",
                               assumed_training_cutoff="", tasks=[train, held])
    assert [t.task_id for t in v1like.scored_tasks()] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Task 18: build_v2 emits the whole v2 artifact.
# --------------------------------------------------------------------------- #


def test_build_v2_emits_sixteen_shapes_in_two_frozen_streams(tmp_path, monkeypatch):
    from bench import build_v2
    monkeypatch.setattr(build_v2, "TASKS", tmp_path / "tasks")
    monkeypatch.setattr(build_v2, "CORPORA", tmp_path / "corpora")
    monkeypatch.setattr(build_v2, "MANIFEST", tmp_path / "manifest.v2.json")
    rc = build_v2.main(["--stream", "both", "--small"])      # --small: 6K/20K tokens for tests
    assert rc == 0
    m = BenchmarkManifest.load(tmp_path / "manifest.v2.json")
    assert len(m.tasks) == 32 and len(m.scored_tasks()) == 16
    assert {t.shape_id for t in m.tasks if t.stream == "train"} == {t.shape_id for t in m.tasks if t.stream == "held_out"}
    assert m.rules["margin"] == 2 and m.rules["abstentions"] == {"b2": ["interactive"]}
    m.validate(require_closed_book=False)


def test_build_v2_refuses_a_task_a_parser_can_solve(tmp_path, monkeypatch):
    from bench import build_v2
    monkeypatch.setattr(build_v2, "PARAPHRASE", False)     # disable the defence
    monkeypatch.setattr(build_v2, "TASKS", tmp_path / "tasks")
    monkeypatch.setattr(build_v2, "CORPORA", tmp_path / "corpora")
    monkeypatch.setattr(build_v2, "MANIFEST", tmp_path / "manifest.v2.json")
    with pytest.raises(SystemExit) as e:
        build_v2.main(["--stream", "train", "--small"])
    assert "parser adversary" in str(e.value)


def test_practice_stream_writes_outside_bench_and_no_manifest(tmp_path):
    from bench import build_v2
    rc = build_v2.main(["--practice", "--seed", "7", "--out", str(tmp_path / "p"), "--small"])
    assert rc == 0 and (tmp_path / "p" / "tasks" / "ls-01-practice.json").exists()
    assert not (tmp_path / "p" / "manifest.v2.json").exists()
