"""S2 chunk-size sweep harness: fixture determinism, the classifier, the
token-target search, and both runners end to end against `MockLlamaServer`.

Nothing here touches a real server. The point of the suite is that the parts
which decide what the sweep MEASURES — how a chunk is sized, where a needle
lands, and which of the five labels an answer gets — are pinned before any GPU
time is spent, because a classifier bug is indistinguishable from a model
finding once the run is over.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from s1.make_fixtures import approx_tokens, leaf_counter
from s2 import make_sweep_fixtures as mk
from s2 import run_sweep as rs
from s2 import run_ub_experiment as ub
from s2.leafcall import PinnedLeafCaller

# --------------------------------------------------------------------------- #
# fixtures: determinism and the claims build_cell asserts
# --------------------------------------------------------------------------- #


def test_cells_are_reproducible_from_the_seed():
    a = mk.build_cell(approx_tokens, size_tokens=1024, position=0.5, seed=1)
    b = mk.build_cell(approx_tokens, size_tokens=1024, position=0.5, seed=1)
    assert a["text"] == b["text"]
    assert a["sha256"] == b["sha256"]
    assert a["questions"] == b["questions"]


def test_a_different_seed_or_size_plants_different_facts():
    base = mk.build_cell(approx_tokens, size_tokens=1024, seed=1)
    other_seed = mk.build_cell(approx_tokens, size_tokens=1024, seed=2)
    other_size = mk.build_cell(approx_tokens, size_tokens=2048, seed=1)
    keys = {c["questions"]["literal"]["expected"]
            for c in (base, other_seed, other_size)}
    assert len(keys) == 3, "every cell must plant its own key"


def test_planted_facts_hold_the_three_properties_the_sweep_scores():
    cell = mk.build_cell(approx_tokens, size_tokens=2048, seed=1)
    text = cell["text"]
    key = cell["questions"]["literal"]["expected"]
    name = cell["questions"]["paraphrase"]["expected"]
    absent_entity = cell["questions"]["absent"]["entity"]

    assert text.count(key) == 1                        # LITERAL: verbatim, once
    assert mk.UUID_RE.findall(text) == [key]           # …and the only UUID
    assert name not in text                            # PARAPHRASE: regex-proof
    assert all(part in text for part in name.split())  # …but both halves present
    assert absent_entity.lower() not in text.lower()   # ABSENT: really absent
    assert cell["questions"]["absent"]["expected"] is None


def test_the_absent_question_differs_from_the_literal_one_only_by_entity():
    cell = mk.build_cell(approx_tokens, size_tokens=1024, seed=1)
    literal = cell["questions"]["literal"]
    absent = cell["questions"]["absent"]
    assert literal["question"].replace(literal["entity"], "<ORG>") == \
        absent["question"].replace(absent["entity"], "<ORG>")


@pytest.mark.parametrize("size", [1024, 2048, 4096])
def test_cells_hit_their_token_target(size):
    cell = mk.build_cell(approx_tokens, size_tokens=size, seed=1)
    slack = max(8, int(size * mk.SIZE_TOLERANCE))
    assert abs(cell["measured_tokens"] - size) <= slack


@pytest.mark.parametrize("position", [0.1, 0.5, 0.9])
def test_the_needle_lands_at_the_requested_depth(position):
    cell = mk.build_cell(approx_tokens, size_tokens=8192, position=position, seed=1)
    depth = cell["needles"]["literal"]["token_depth"]
    assert abs(depth - position) <= 0.05, f"needle at {depth}, wanted {position}"


def test_an_unplaceable_depth_refuses_rather_than_lying():
    """A 1K cell cannot hold a needle whose START is at 90%: the needles are
    ~16% of it. The generator must say so, not quietly record 0.84."""
    with pytest.raises(AssertionError, match="cannot be placed that deep"):
        mk.build_cell(approx_tokens, size_tokens=1024, position=0.9, seed=1)


def test_fit_to_tokens_is_the_longest_whole_word_prefix_under_the_target():
    text = mk.make_filler(mk._cell_rng(1, 1024, 0.5), 400)
    cut = mk.fit_to_tokens(text, 200, approx_tokens)
    assert approx_tokens(cut) <= 200
    assert cut == text[:len(cut)]                     # a prefix, not a rewrite
    assert not text[len(cut):len(cut) + 1].strip() or cut.split() == \
        text[:len(cut)].split()                       # never cuts mid-word
    nxt = text[len(cut):].split()
    if nxt:                                           # one more word overshoots
        assert approx_tokens(cut + " " + nxt[0]) > 200 or len(cut) == len(text)


def test_boundary_search_is_monotonic_in_the_token_target():
    text = mk.make_filler(mk._cell_rng(1, 4096, 0.5), 800)
    offsets = [mk.boundary_at_token_target(text, t, approx_tokens)
               for t in (0, 100, 400, 800)]
    assert offsets == sorted(offsets)
    assert approx_tokens(text[:offsets[2]]) >= 400


# --------------------------------------------------------------------------- #
# THE CLASSIFIER — the deliverable
# --------------------------------------------------------------------------- #

KEY = "6f1a2b3c-4d5e-4f60-8192-a3b4c5d6e7f8"
NEAR = "6f1a2b3c-4d5e-4f60-8192-a3b4c5d6e7f9"   # last character altered
NAME = "Aldmorawick Fenngate"


def _lit(raw):
    return rs.classify(raw, question_type="literal", expected=KEY,
                       expected_kind="uuid")


def _par(raw):
    return rs.classify(raw, question_type="paraphrase", expected=NAME,
                       expected_kind="name")


def _abs(raw):
    return rs.classify(raw, question_type="absent", expected=None,
                       expected_kind="uuid")


@pytest.mark.parametrize("raw,label", [
    (KEY, rs.CORRECT),
    (f"  {KEY}\n", rs.CORRECT),
    (f"The key is {KEY}.", rs.CORRECT),
    (f"Key: {KEY}", rs.CORRECT),
    (f"```\n{KEY}\n```", rs.CORRECT),
    (f"`{KEY}`", rs.CORRECT),
    (f"<think>\n\n</think>\n\n{KEY}", rs.CORRECT),
    (KEY.upper(), rs.CORRECT),
])
def test_correct_survives_formatting_but_not_content_changes(raw, label):
    assert _lit(raw)["label"] == label


def test_the_motivating_failure_is_a_confabulation_at_edit_distance_one():
    """§7 #2's whole reason for existing: `…feb83` for `…feb89`."""
    out = _lit(NEAR)
    assert out["label"] == rs.CONFABULATION
    assert out["uuid_edit_distance"] == 1


def test_a_wholly_invented_key_is_also_a_confabulation_but_further_away():
    out = _lit("00000000-0000-4000-8000-000000000000")
    assert out["label"] == rs.CONFABULATION
    assert out["uuid_edit_distance"] > 1


@pytest.mark.parametrize("raw", [
    "NONE", "none", "None.", "not present", "NOT PRESENT.",
    "The excerpt does not contain that key", "There is no such key here",
    "<think>\n\n</think>\n\nNONE",
])
def test_a_refusal_to_a_present_fact_is_a_MISS_never_a_confabulation(raw):
    assert _lit(raw)["label"] == rs.MISS


@pytest.mark.parametrize("raw", ["NONE", "none", "not present in this excerpt"])
def test_the_same_refusal_to_an_absent_fact_is_CORRECT(raw):
    assert _abs(raw)["label"] == rs.CORRECT


def test_any_non_refusal_answer_to_an_absent_question_is_a_false_positive():
    out = _abs(KEY)
    assert out["label"] == rs.FALSE_POSITIVE


def test_a_hedge_that_still_names_a_value_is_not_a_refusal():
    """"NONE, but the closest match is …" is a confabulation wearing a hedge;
    scoring it as a MISS would understate exactly the danger being measured."""
    assert _lit(f"None, but the closest match is {NEAR}")["label"] == rs.CONFABULATION
    assert _abs(f"None. The nearest entry is {KEY}")["label"] == rs.FALSE_POSITIVE


def test_an_answer_offering_two_keys_has_not_extracted_one():
    out = _lit(f"{KEY} or {NEAR}")
    assert out["label"] == rs.CONFABULATION
    assert out["uuid_edit_distance"] == 0  # the true value IS among them…
    assert len(out["uuid_candidates"]) == 2  # …but so is another


@pytest.mark.parametrize("raw", ["", "   ", "\n\n", "<think>\n\n</think>\n\n",
                                 "...", "-- -- --"])
def test_empty_and_contentless_output_is_MALFORMED(raw):
    assert _lit(raw)["label"] == rs.MALFORMED


def test_regurgitating_the_excerpt_is_MALFORMED_not_CORRECT():
    """The key is IN there — and the leaf was asked for the bare value. A
    containment rule with no length guard would score a dump as extraction."""
    dump = ("filler " * 200) + KEY + (" filler" * 200)
    out = _lit(dump)
    assert out["label"] == rs.MALFORMED
    assert "restated" in out["reason"]


def test_a_decode_loop_is_MALFORMED():
    assert _lit("key key key key key key key key key")["label"] == rs.MALFORMED


def test_paraphrase_scoring_is_content_exact():
    assert _par(NAME)["label"] == rs.CORRECT
    assert _par(f"{NAME.lower()}")["label"] == rs.CORRECT
    assert _par("Aldmorawick Fenngato")["label"] == rs.CONFABULATION
    assert _par("NONE")["label"] == rs.MISS


def test_every_answer_gets_exactly_one_label_from_the_taxonomy():
    for raw in (KEY, NEAR, "NONE", "", "junk", f"none, maybe {NEAR}"):
        for fn in (_lit, _par):
            assert fn(raw)["label"] in rs.LABELS


def test_the_scorer_refuses_a_mis_specified_cell():
    with pytest.raises(ValueError):
        rs.classify("x", question_type="absent", expected=KEY)
    with pytest.raises(ValueError):
        rs.classify("x", question_type="literal", expected=None)
    with pytest.raises(ValueError):
        rs.classify("x", question_type="nonsense", expected=KEY)


def test_edit_distance_is_levenshtein():
    assert rs.edit_distance("abc", "abc") == 0
    assert rs.edit_distance("abc", "abd") == 1
    assert rs.edit_distance("abc", "ab") == 1
    assert rs.edit_distance("", "abc") == 3


# --------------------------------------------------------------------------- #
# the call schedule
# --------------------------------------------------------------------------- #


def test_the_plan_pays_one_cold_prefill_per_chunk():
    plan = rs.plan_calls([{"cell_id": "c"}], trials=3)
    assert len(plan) == 9
    assert [c["cold"] for c in plan] == [True] + [False] * 8
    assert [c["seed"] for c in plan[:9:3]] == [1, 2, 3]


def test_each_trial_asks_all_three_question_types_in_a_rotated_order():
    plan = rs.plan_calls([{"cell_id": "c"}], trials=3)
    per_trial = [[c["question_type"] for c in plan if c["trial"] == t]
                 for t in (1, 2, 3)]
    for types in per_trial:
        assert sorted(types) == sorted(rs.QUESTION_TYPES)
    assert len({tuple(t) for t in per_trial}) == 3, "the order must rotate"


# --------------------------------------------------------------------------- #
# end to end against the loopback mock server
# --------------------------------------------------------------------------- #


def _caller(server, leaf_prefix: str) -> PinnedLeafCaller:
    from rlm.dispatcher import ServerClient

    return PinnedLeafCaller(
        client=ServerClient(server.base_url, timeout=10.0),
        system_prefix=leaf_prefix, max_predict=64, temperature=0.3, top_p=0.9,
        enable_thinking=False, slot_capacity_tokens=8192)


async def _fixtures(server, tmp_path: Path, *, size: int = 256) -> dict:
    count = leaf_counter(server.port)          # the mock's own /tokenize
    return mk.write_fixtures(count, sizes=(size,), positions=(0.5,),
                             out_dir=tmp_path, counter_name="leaf:/tokenize")


async def test_prepare_measures_the_rendered_prefix_head(mock_server, leaf_prefix):
    caller = _caller(mock_server, leaf_prefix)
    try:
        prefix_tokens = await caller.prepare()
        assert prefix_tokens and prefix_tokens > 0
        assert caller.markers  # derived from the server's own chat_template
    finally:
        await caller.aclose()


async def test_the_sweep_pins_one_slot_and_re_queries_the_resident_chunk(
        mock_server, leaf_prefix, tmp_path):
    manifest = await _fixtures(mock_server, tmp_path)
    caller = _caller(mock_server, leaf_prefix)
    out = tmp_path / "sweep.jsonl"
    try:
        await caller.prepare()
        records = await rs.sweep(caller, manifest,
                                 list(manifest["cells"].values()), slot=3,
                                 trials=3, seeds=(1, 2, 3), phase="main",
                                 out_path=out, echo=lambda *a, **k: None)
    finally:
        await caller.aclose()

    assert len(records) == 9                          # 3 trials x 3 questions
    assert [r["cold"] for r in records] == [True] + [False] * 8
    bodies = [b for b in mock_server.completion_bodies]
    assert {b["id_slot"] for b in bodies} == {3}      # ONE pinned slot
    assert [b["seed"] for b in bodies[:3]] == [1, 1, 1]
    assert {b["temperature"] for b in bodies} == {0.3}
    assert all(b["cache_prompt"] for b in bodies)


async def test_every_call_records_the_raw_output_and_the_cache_evidence(
        mock_server, leaf_prefix, tmp_path):
    manifest = await _fixtures(mock_server, tmp_path)
    caller = _caller(mock_server, leaf_prefix)
    out = tmp_path / "sweep.jsonl"
    try:
        await caller.prepare()
        await rs.sweep(caller, manifest, list(manifest["cells"].values()),
                       slot=0, trials=1, seeds=(1,), phase="main",
                       out_path=out, echo=lambda *a, **k: None)
    finally:
        await caller.aclose()

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    for rec in lines:
        for field in ("raw_output", "tokens_cached", "slot_id", "prefill_ms",
                      "decode_ms", "wall_s", "prefix_tokens", "label",
                      "question_type", "expected", "size_target",
                      "size_measured", "position", "trial", "seed"):
            assert field in rec, field
        # verbatim: the mock echoes the user segment, and the record keeps it
        # whole rather than the normalized/truncated form.
        assert rec["raw_output"].startswith("echo:")
        assert rec["status"] == "ok"


async def test_the_chunk_is_sent_before_the_question(mock_server, leaf_prefix,
                                                     tmp_path):
    """§4's layout is what makes the warm re-query possible at all: a
    question-first prompt measured cache_n 0 twice."""
    manifest = await _fixtures(mock_server, tmp_path)
    cell = next(iter(manifest["cells"].values()))
    chunk = Path(cell["chunk_path"]).read_text(encoding="utf-8")
    caller = _caller(mock_server, leaf_prefix)
    try:
        await caller.prepare()
        await rs.sweep(caller, manifest, [cell], slot=0, trials=1, seeds=(1,),
                       phase="main", out_path=tmp_path / "s.jsonl",
                       echo=lambda *a, **k: None)
    finally:
        await caller.aclose()
    # trial 1's rotation is (literal, paraphrase, absent), so the last call is
    # the ABSENT question -- and the user message is exactly chunk-then-question.
    user = mock_server.template_bodies[-1]["messages"][1]["content"]
    assert user == f"{chunk}\n\n{cell['questions']['absent']['question']}"
    system = mock_server.template_bodies[-1]["messages"][0]["content"]
    assert system == leaf_prefix       # §4's byte-identical head, untouched


async def test_a_server_error_is_recorded_rather_than_retried(
        mock_server, leaf_prefix, tmp_path):
    manifest = await _fixtures(mock_server, tmp_path)
    caller = _caller(mock_server, leaf_prefix)
    out = tmp_path / "sweep.jsonl"
    try:
        await caller.prepare()
        mock_server.fail_times(1)
        records = await rs.sweep(caller, manifest,
                                 list(manifest["cells"].values()), slot=0,
                                 trials=1, seeds=(1,), phase="main",
                                 out_path=out, echo=lambda *a, **k: None)
    finally:
        await caller.aclose()
    assert records[0]["status"] == "error" and records[0]["label"] is None
    assert mock_server.dispatch_count == 3      # 3 calls, no retry of the first
    assert records[1]["status"] == "ok"


def test_the_report_rescoring_ignores_the_label_written_at_run_time():
    stale = [{"status": "ok", "raw_output": KEY, "question_type": "literal",
              "expected": KEY, "expected_kind": "uuid", "label": rs.MISS,
              "size_target": 1024, "position": 0.5, "trial": 1, "cold": True,
              "tokens_cached": 10, "tokens_in": 20, "wall_s": 1.0,
              "slot_id": 0, "cache_hit_fraction": 0.5}]
    assert rs.rescore(stale)[0]["label"] == rs.CORRECT
    assert "CORRECT" in rs.render_report(stale)


def test_the_summary_keeps_miss_and_confabulation_apart():
    recs = [
        {"status": "ok", "raw_output": "NONE", "question_type": "literal",
         "expected": KEY, "expected_kind": "uuid", "size_target": 1024,
         "position": 0.5, "trial": 1, "cold": True, "tokens_cached": 0,
         "tokens_in": 10, "wall_s": 3.0, "slot_id": 0},
        {"status": "ok", "raw_output": NEAR, "question_type": "literal",
         "expected": KEY, "expected_kind": "uuid", "size_target": 1024,
         "position": 0.5, "trial": 2, "cold": False, "tokens_cached": 9,
         "tokens_in": 10, "wall_s": 0.5, "slot_id": 0},
    ]
    cell = rs.summarize(rs.rescore(recs))["cells"][(1024, 0.5, "literal")]
    assert cell["labels"][rs.MISS] == 1
    assert cell["labels"][rs.CONFABULATION] == 1
    assert cell["uuid_edit_distances"] == [1]


def test_the_runner_refuses_fixtures_built_with_the_offline_proxy(tmp_path):
    mk.write_fixtures(approx_tokens, sizes=(1024,), positions=(0.5,),
                      out_dir=tmp_path, counter_name="approx-offline")
    assert rs.main(["--phase", "main", "--fixtures", str(tmp_path)]) == 2


# --------------------------------------------------------------------------- #
# the ub experiment
# --------------------------------------------------------------------------- #


def test_ub_verdict_reads_the_numbers_the_spec_pre_registered():
    meetable = ub.ub_verdict(prefix_tokens=311, cache_n_b=311,
                             cache_n_a_warm=2400, ub=128)
    assert meetable["verdict"] == "GATE-A MEETABLE"
    assert meetable["reaches_prefix"] is True

    not_meetable = ub.ub_verdict(prefix_tokens=311, cache_n_b=38,
                                 cache_n_a_warm=2400, ub=512)
    assert not_meetable["verdict"] == "GATE-A NOT MEETABLE"
    assert not_meetable["reaches_prefix"] is False


def test_a_failed_warm_control_invalidates_the_ub_run():
    """cache_n(B)=0 could mean 'the prefix did not survive' or 'the pin did not
    work'. Without the control the two are indistinguishable, so the verdict
    must refuse to choose."""
    out = ub.ub_verdict(prefix_tokens=311, cache_n_b=0, cache_n_a_warm=0, ub=512)
    assert out["verdict"] == "INVALID"
    assert out["control_ok"] is False


async def test_the_ub_experiment_runs_four_calls_on_one_pinned_slot(
        mock_server, leaf_prefix):
    count = leaf_counter(mock_server.port)
    chunk_a, chunk_b = ub.build_chunks(count, chunk_tokens=128, seed=1)
    assert chunk_a != chunk_b
    assert chunk_a[:100] != chunk_b[:100]

    caller = _caller(mock_server, leaf_prefix)
    try:
        await caller.prepare()
        record = await ub.run_experiment(caller, ub=512, slot=5,
                                         chunk_a=chunk_a, chunk_b=chunk_b,
                                         echo=lambda *a, **k: None)
    finally:
        await caller.aclose()

    assert [c["step"] for c in record["calls"]] == [
        "A_cold", "A_warm", "B_after_A", "A_after_B"]
    assert {b["id_slot"] for b in mock_server.completion_bodies} == {5}
    assert record["ub"] == 512
    # The mock never reuses a prompt, so the warm control fails -- which is
    # exactly the outcome that must invalidate the run rather than be read as
    # "the prefix did not survive".
    assert record["verdict"] == "INVALID"
