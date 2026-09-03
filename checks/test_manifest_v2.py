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
