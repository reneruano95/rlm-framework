import io
import json

from rlm.trace.lifecycle import ALLOWED_KINDS, Lifecycle


def test_writes_jsonl_to_file_and_stream(tmp_path):
    buf = io.StringIO()
    lc = Lifecycle(tmp_path / "lc.jsonl", stream=buf)
    lc.event("server_health", server="leaf", state="up")
    lc.close()
    line = (tmp_path / "lc.jsonl").read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["kind"] == "server_health" and rec["server"] == "leaf"
    assert "ts" in rec
    assert json.loads(buf.getvalue().strip())["kind"] == "server_health"


def test_rejects_kinds_outside_the_narrow_allowlist():
    lc = Lifecycle(None, stream=io.StringIO())
    for kind in ("step", "episode", "observation", "llm_call"):
        assert kind not in ALLOWED_KINDS
    try:
        lc.event("step", foo=1)
    except ValueError as exc:
        assert "not a lifecycle kind" in str(exc)
    else:
        raise AssertionError("episode data must be refused (spec §5, I4)")


def test_surrogates_do_not_crash_the_logger(tmp_path):
    lc = Lifecycle(tmp_path / "lc.jsonl", stream=io.StringIO())
    lc.event("trace_write_failure", detail="bad" + chr(0xDCFF))
    lc.close()
    json.loads((tmp_path / "lc.jsonl").read_text(encoding="utf-8").strip())


def test_never_raises_into_the_caller(tmp_path):
    lc = Lifecycle(tmp_path / "nonexistent-dir" / "lc.jsonl", stream=io.StringIO())
    lc.event("quiesce_wait", seconds=1)  # must degrade, not explode
    lc.close()


def test_root_history_is_a_lifecycle_kind():
    """v0.3.16: the history-divergence monitor is a lifecycle event, and
    Lifecycle.event refuses unknown kinds with ValueError."""
    from rlm.trace.lifecycle import ALLOWED_KINDS
    assert "root_history" in ALLOWED_KINDS
