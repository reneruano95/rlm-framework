# tests/test_prompts.py
import hashlib
import re
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
FILES = ["root.v1.md", "root.v2.md", "leaf-prefix.v1.md", "strat-needle.v1.md",
         "strat-aggregation.v1.md", "strat-synthesis.v1.md",
         "strat-codeqa.v1.md", "strat-default.v1.md"]


@pytest.mark.parametrize("name", FILES)
def test_every_file_exists_and_opens_with_a_changelog_header(name):
    text = (PROMPTS / name).read_text(encoding="utf-8")
    assert text.startswith("<!-- changelog"), "spec §5 requires the header"
    assert re.search(r"^v\d+ ", text.split("-->")[0], re.M)


def test_leaf_prefix_carries_no_volatile_tokens():
    """§4: byte-identical prefix — no timestamps, ids, counters, dates."""
    text = (PROMPTS / "leaf-prefix.v1.md").read_text(encoding="utf-8")
    body = text.split("-->", 1)[1]
    for pattern in (r"\d{4}-\d{2}-\d{2}", r"\bid\b\s*[:=]", r"\{[a-z_]+\}",
                    r"run[_ ]?id", r"episode"):
        assert not re.search(pattern, body, re.I), f"volatile token {pattern!r}"


def test_root_prompts_use_the_injected_api_names_exactly():
    for name in ("root.v1.md", "root.v2.md"):
        text = (PROMPTS / name).read_text(encoding="utf-8")
        assert "llm_query" in text and "final_answer" in text
        assert "chunks" in text and "context" in text
        assert "llm_query_batched" not in text  # not our API
        assert "SUBMIT(" not in text and "FINAL(" not in text


def test_the_two_ab_variants_differ_only_by_the_exemplar_block():
    """CONTROLLER RULING (brief defect, same class as task-13's st_int bug):
    the brief asserted v1's *entire* body is a raw substring of v2. But the
    probe's own generator (gen_v2.py) splices the worked-exemplar block in
    BEFORE the closing "A strategy block ... follows" sentence, not after
    it -- that sentence has to stay the LAST paragraph of the root prompt
    (the scaffold appends the selected strategy block immediately after),
    so the exemplars cannot be tacked on past it. That makes v1's tips a
    *prefix* of v2 up to the anchor sentence, and the anchor sentence itself
    a matching *suffix* -- never one contiguous substring spanning both.
    This checks the actual controlled-A/B invariant (v2 = v1 with exactly
    one contiguous insertion) instead of the naive substring check.
    """
    v1 = (PROMPTS / "root.v1.md").read_text(encoding="utf-8")
    v2 = (PROMPTS / "root.v2.md").read_text(encoding="utf-8")
    assert len(v2) > len(v1), "v2 is tips + worked exemplars"
    body_v1 = v1.split("-->", 1)[1].strip()
    body_v2 = v2.split("-->", 1)[1].strip()
    anchor = "A strategy block for this task's declared category follows."
    idx1, idx2 = body_v1.index(anchor), body_v2.index(anchor)
    assert body_v1[:idx1] == body_v2[:idx1], "tips before the insertion point must match verbatim"
    assert body_v1[idx1:] == body_v2[idx2:], "the closing strategy-block sentence must be byte-identical"
    assert body_v2[idx1:idx2].strip(), "v2 must insert something between the tips and the closing sentence"


def test_exemplars_use_the_canonical_fence_and_our_api():
    v2 = (PROMPTS / "root.v2.md").read_text(encoding="utf-8")
    assert "```repl" in v2
    assert "await llm_query(" in v2
    assert "final_answer(" in v2
    assert "asyncio.gather" in v2  # the fan-out idiom


def test_v2_exemplars_use_the_pre_registered_chunk_kwarg():
    """§4's layout is now enforced by the scaffold (`llm_query(q, chunk=...)`
    composes `[prefix][chunk][question]` scaffold-side). The exemplars are what
    the S2 gate is really testing today, so they must teach the form a gate can
    score -- a hand-composed `chunk + "\\n\\n" + q` is indistinguishable from
    any other single string once it crosses the bridge.

    root.v1.md is deliberately NOT checked here: it is the pinned S1 A/B winner
    and editing it would invalidate the recorded result.
    """
    v2 = (PROMPTS / "root.v2.md").read_text(encoding="utf-8")
    examples = v2.split("# Worked examples", 1)[1]
    assert "chunk=chunks[" in examples
    assert "llm_query(chunks[i] + " not in examples
    assert "llm_query(c + " not in examples


def test_prompt_promise_matches_the_configured_extractor():
    """D16: the file text is generated from cell_extraction; they cannot disagree."""
    from rlm.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    v1 = (PROMPTS / "root.v1.md").read_text(encoding="utf-8")
    if cfg.scaffold.cell_extraction.select == "first":
        assert "only the first runs" in v1
    else:
        assert "only the last runs" in v1


def test_extraction_shaped_strategies_carry_the_evidence_span_check():
    for name in ("strat-needle.v1.md", "strat-aggregation.v1.md",
                 "strat-synthesis.v1.md"):
        text = (PROMPTS / name).read_text(encoding="utf-8").lower()
        assert "evidence" in text, f"{name} missing the R12/R5 evidence-span check"


def test_config_pins_match_the_files_on_disk():
    from rlm.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    for path, pinned in cfg.pinned_prompt_hashes().items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert actual == pinned, f"{path} drifted from its config pin"
