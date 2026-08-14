"""The envelope's two wirings: the prompt registry, and C4 (spec §5).

The registry half is what makes the S2 A/B a 2x2 rather than two arms: the
envelope's format instructions live in their OWN sha256-pinned file, APPENDED to
whichever leaf prefix is pinned, so `leaf-prefix.v1.md` — the control arm behind
every measurement recorded so far — never has to be edited to carry them.

The C4 half is the production path §5 specifies: parsed and validated
SCAFFOLD-SIDE in `llm_query`, one retry on parse failure through C4's existing
retry machinery, then a structured error the root can branch on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rlm.config import ConfigError, PromptRegistry
from rlm.errors import EnvelopeParseError, StepStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = REPO_ROOT / "prompts"


# --------------------------------------------------------------------------- #
# the registry: prefix and envelope block vary independently
# --------------------------------------------------------------------------- #


def _registry(tmp_path: Path, *, envelope: bool) -> PromptRegistry:
    (tmp_path / "root.md").write_text("ROOT", encoding="utf-8")
    (tmp_path / "leaf.md").write_text("LEAF PREFIX", encoding="utf-8")
    (tmp_path / "env.md").write_text("ENVELOPE BLOCK", encoding="utf-8")
    (tmp_path / "strat.md").write_text("STRAT", encoding="utf-8")
    return PromptRegistry.from_files(
        root_path=tmp_path / "root.md",
        leaf_prefix_path=tmp_path / "leaf.md",
        leaf_envelope_path=(tmp_path / "env.md") if envelope else None,
        strategy_paths={"needle": tmp_path / "strat.md"},
    )


def test_render_leaf_is_the_prefix_alone_when_the_envelope_is_off(tmp_path):
    registry = _registry(tmp_path, envelope=False).load()
    assert registry.render_leaf(envelope=False) == "LEAF PREFIX"
    assert registry.render_leaf(envelope=False) == registry.leaf_prefix()


def test_render_leaf_appends_the_envelope_block(tmp_path):
    registry = _registry(tmp_path, envelope=True).load()
    assert registry.render_leaf(envelope=True) == "LEAF PREFIX\n\nENVELOPE BLOCK"
    # ... and the prefix arm is byte-identical to the no-envelope registry's,
    # which is what makes v1-with-envelope a true second arm of the same 2x2.
    assert registry.render_leaf(envelope=False) == "LEAF PREFIX"


def test_asking_for_an_envelope_that_was_never_declared_is_an_error(tmp_path):
    registry = _registry(tmp_path, envelope=False).load()
    with pytest.raises(ConfigError, match="no leaf envelope"):
        registry.render_leaf(envelope=True)


def test_the_envelope_block_is_hashed_like_every_other_prompt(tmp_path):
    registry = _registry(tmp_path, envelope=True).load()
    hashes = registry.hashes()
    assert "leaf_envelope.file" in hashes
    assert "leaf_envelope.body" in hashes


def test_a_pinned_envelope_block_that_drifted_is_refused(tmp_path):
    registry = _registry(tmp_path, envelope=True)
    registry.leaf_envelope_sha256 = "0" * 64
    with pytest.raises(ConfigError, match="sha256 mismatch"):
        registry.load()


# --------------------------------------------------------------------------- #
# the shipped prompt files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["leaf-prefix.v1.md", "leaf-prefix.v2.md",
                                  "leaf-envelope.v1.md", "leaf-envelope.v2.md"])
def test_the_leaf_prompt_files_carry_no_volatile_token(name):
    """§4's byte-identical head: no timestamps, ids or counters may enter the
    prefix. The changelog header carries dates and is stripped before render,
    so the check is against the BODY."""
    from rlm.config import _strip_changelog

    body = _strip_changelog((PROMPTS / name).read_text(encoding="utf-8"))
    assert "2026" not in body
    if not name.startswith("leaf-envelope"):
        assert "{" not in body          # only the envelope's JSON examples
    for volatile in ("run_id", "episode_id", "task_id", "timestamp", "chunk_index"):
        assert volatile not in body


def test_the_envelope_block_ships_at_v2_and_v1_stays_on_disk_unedited():
    """v1's override clause pointed the wrong way — it said it outranked the
    instructions "above", while §4 puts the QUESTION last, so the question's
    "reply with the key itself and nothing else" won and a smoke run returned 12
    bare answers and zero JSON. v2 names the question explicitly. v1 is kept,
    unmodified and unpinned, because the registry rule is monotonic versions and
    because the reason it was superseded is part of the record."""
    from rlm.config import _strip_changelog

    v1 = _strip_changelog((PROMPTS / "leaf-envelope.v1.md").read_text(encoding="utf-8"))
    v2 = _strip_changelog((PROMPTS / "leaf-envelope.v2.md").read_text(encoding="utf-8"))
    assert "overrides any formatting instruction above" in v1
    assert "including the question's own" in v2
    # The contract itself did not move: same three fields, same abstain rule, so
    # the correction is about WHERE the format wins, not about what it asks for.
    for clause in ('`"answer"`', '`"evidence"`', '`"abstain"`',
                   'Set `"abstain": true` whenever the excerpt does not answer'):
        assert clause in v1 and clause in v2


def test_v1_is_untouched_by_this_work():
    """v1 is the pinned prefix behind every measurement recorded so far and the
    A/B's control arm. Its sha256 is asserted, not merely intended: an edit here
    would silently move the 95% baseline the whole experiment is measured
    against."""
    import hashlib

    digest = hashlib.sha256((PROMPTS / "leaf-prefix.v1.md").read_bytes()).hexdigest()
    assert digest == "dfaddb37a24594724709d36dbb117fc1b12bfa436229322d4d5ebf46625d0f76"


def test_v2_offers_the_same_refusal_TOKEN_as_v1():
    """The two prefixes must be scoreable by ONE classifier: if v2 renamed the
    refusal token the arms would differ in what a refusal even looks like, and
    the comparison would be of the scorer, not of the prompt."""
    from rlm.config import _strip_changelog

    for name in ("leaf-prefix.v1.md", "leaf-prefix.v2.md"):
        body = _strip_changelog((PROMPTS / name).read_text(encoding="utf-8"))
        assert "`NONE`" in body


# --------------------------------------------------------------------------- #
# C4: parse scaffold-side, retry through the EXISTING machinery
#
# Against `mock_server` -- a real loopback HTTP server speaking the leaf's
# /tokenize + /apply-template + /completion (SSE) shapes -- so what runs is the
# whole dispatch path, streaming included, not a stubbed client.
# --------------------------------------------------------------------------- #

GOOD = '{"answer": "ENT-1", "evidence": ["key ENT-1"], "abstain": false}'
ABSTAIN = '{"answer": "", "evidence": [], "abstain": true}'
CHUNK = "a document in which key ENT-1 appears"


async def test_envelope_off_returns_the_raw_string_unchanged(mock_server):
    """Every existing caller and every measurement recorded so far depends on
    this: the envelope is opt-in and its default is off."""
    mock_server.answer = "NONE"
    d = mock_server.dispatcher()
    assert await d.query("q", role="leaf", call_id="c1", chunk=CHUNK) == "NONE"


async def test_envelope_on_returns_a_structured_result(mock_server):
    mock_server.answer = GOOD
    d = mock_server.dispatcher(envelope=True)
    out = await d.query("q", role="leaf", call_id="c1", chunk=CHUNK)
    assert out["answer"] == "ENT-1"
    assert out["abstain"] is False
    assert out["evidence"] == ["key ENT-1"]
    assert out["evidence_verified"] == [True]
    assert out["evidence_ok"] is True
    assert out["raw"] == GOOD


async def test_evidence_is_verified_against_the_chunk_not_the_answer(mock_server):
    """A span the leaf invented comes back False even though the envelope parsed
    perfectly -- that is the entire content of the check."""
    mock_server.answer = GOOD
    d = mock_server.dispatcher(envelope=True)
    out = await d.query("q", role="leaf", call_id="c1",
                        chunk="a document mentioning no such key at all")
    assert out["evidence_verified"] == [False]
    assert out["evidence_ok"] is False


async def test_evidence_is_NOT_CHECKED_without_a_chunk(mock_server):
    """The single-string call form: C4 cannot see where the document ended, so
    it has checked nothing. None, never False (`rlm.leakcheck`'s discipline)."""
    mock_server.answer = GOOD
    d = mock_server.dispatcher(envelope=True)
    out = await d.query("q", role="leaf", call_id="c1")
    assert out["evidence_verified"] == [None]
    assert out["evidence_ok"] is None


async def test_an_abstention_checks_nothing_rather_than_failing_the_check(mock_server):
    """An abstention quotes nothing, so there is nothing to verify. Reporting
    `evidence_ok=False` there would make every refusal look like a caught lie."""
    mock_server.answer = ABSTAIN
    d = mock_server.dispatcher(envelope=True)
    out = await d.query("q", role="leaf", call_id="c1", chunk=CHUNK)
    assert out["abstain"] is True
    assert out["evidence_ok"] is None


async def test_a_malformed_envelope_is_retried_and_then_succeeds(mock_server):
    """The retry rides C4's EXISTING machinery: same call_id, incrementing
    retry_idx, one logged step per attempt -- so the call still counts ONCE
    against max_subcalls while every attempt's tokens count (§5 C4)."""
    mock_server.answer_script = ["sorry, I cannot produce JSON", GOOD]
    d = mock_server.dispatcher(envelope=True)
    out = await d.query("q", role="leaf", call_id="c1", chunk=CHUNK)
    assert out["answer"] == "ENT-1"
    assert [s["retry_idx"] for s in d.steps] == [0, 1]
    assert d.steps[0]["status"] == StepStatus.ERROR
    assert "envelope" in d.steps[0]["error_detail"]
    assert d.steps[1]["status"] == StepStatus.OK
    assert {s["call_id"] for s in d.steps} == {"c1"}


async def test_a_persistently_malformed_envelope_raises_a_structured_error(mock_server):
    """`EnvelopeParseError` must be distinguishable from a dead server -- both
    are DispatchErrors, and they have opposite remedies."""
    mock_server.answer = "nope, plain text forever"
    d = mock_server.dispatcher(envelope=True)
    with pytest.raises(EnvelopeParseError) as excinfo:
        await d.query("q", role="leaf", call_id="c1", chunk=CHUNK)
    assert excinfo.value.raw == "nope, plain text forever"
    assert len(d.steps) == 3                        # max_attempts, all recorded
    assert all(s["status"] == StepStatus.ERROR for s in d.steps)


async def test_the_failed_attempts_record_what_could_not_be_parsed(mock_server):
    """Otherwise a run reports a count of envelope failures with no way to see
    what the leaf emitted, and the A/B's MALFORMED column is unauditable."""
    mock_server.answer = "I cannot comply"
    d = mock_server.dispatcher(envelope=True)
    with pytest.raises(EnvelopeParseError):
        await d.query("q", role="leaf", call_id="c1", chunk=CHUNK)
    assert all(s["response_text"] == "I cannot comply" for s in d.steps)


async def test_an_abstention_is_a_normal_answer_not_an_error(mock_server):
    """The point of the field: "not here" must be a cheap, legitimate,
    non-error reply, or the root learns to avoid it exactly as the model does."""
    mock_server.answer = ABSTAIN
    d = mock_server.dispatcher(envelope=True)
    out = await d.query("q", role="leaf", call_id="c1", chunk=CHUNK)
    assert out["abstain"] is True
    assert len(d.steps) == 1
    assert d.steps[0]["status"] == StepStatus.OK


async def test_the_leak_detector_still_runs_on_an_envelope_answer(mock_server):
    """R13's foreign-string detector runs on EVERY leaf answer (§5 C4), and it
    must see the whole raw reply: a leaked identifier that lands in `evidence`
    rather than in `answer` is still a leak."""
    mock_server.answer = ('{"answer": "x", "evidence": ["ENT-99999"], '
                          '"abstain": false}')
    d = mock_server.dispatcher(envelope=True)
    d.set_corpus({"other": "ENT-99999 lives in a different chunk"})
    await d.query("q", role="leaf", call_id="c1", chunk="a clean chunk")
    assert d.steps[0]["leak_detected"] is True
    assert "ENT-99999" in d.steps[0]["leak_detail"]


async def test_the_envelope_block_actually_reaches_the_model(mock_server):
    """The format instructions ride in the SYSTEM PREFIX (§4), not in a per-call
    instruction: a per-call one would be a volatile byte in the head and would
    break the byte-identical contract R3's drift detector rests on."""
    registry = PromptRegistry.from_files(
        root_path=PROMPTS / "root.v3.md",
        leaf_prefix_path=PROMPTS / "leaf-prefix.v1.md",
        leaf_envelope_path=PROMPTS / "leaf-envelope.v1.md",
        strategy_paths={},
    ).load()
    d = mock_server.dispatcher(envelope=True,
                               system_prefix=registry.render_leaf(envelope=True))
    mock_server.answer = GOOD
    await d.query("q", role="leaf", call_id="c1", chunk=CHUNK)
    system = mock_server.template_bodies[0]["messages"][0]
    assert system["role"] == "system"
    assert '"abstain"' in system["content"]
    assert system["content"].startswith(registry.leaf_prefix())


async def test_the_rendered_head_is_byte_identical_across_calls(mock_server):
    """§4's contract, asserted on the wire rather than assumed: two calls, two
    different chunks, one identical system message."""
    mock_server.answer = ABSTAIN
    d = mock_server.dispatcher(envelope=True, slot_pool=4)
    await d.query("q1", role="leaf", call_id="c1", chunk="chunk one")
    await d.query("q2", role="leaf", call_id="c2", chunk="chunk two")
    heads = [b["messages"][0]["content"] for b in mock_server.template_bodies]
    assert len(heads) == 2
    assert heads[0] == heads[1]


