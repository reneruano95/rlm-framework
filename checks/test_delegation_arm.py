"""The delegation arm: `llm_query` as the ONLY route to chunk content.

Plan: `docs/superpowers/plans/2026-08-20-delegation-arm.md`. The arm exists to
price what delegation adds, because after S1, S2 and S4 the RLM arm has made
zero leaf calls in a scored episode.

Two properties carry the arm, and both are asserted here:
  * the handicap BITES -- chunk text is not reachable from the sandbox, and
    reaching for it says so rather than failing obscurely;
  * the handicap is INERT by default -- `run_episode` serves S1 and S3 too, and
    an arm that leaked into them would invalidate gates that have already passed.
"""
import inspect
import sys

import pytest

from rlm.measure.bench import ARM_ORDER, ARM_PROFILE, RESIDENT_PROFILE
from rlm.episode import run_episode

pytestmark_win = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def test_the_restricted_arm_is_registered_next_to_rlm():
    """Adjacency is not cosmetic: both run on the RESIDENT profile, so placing
    them together is what keeps §8's two-relaunch-per-block bound intact."""
    assert ARM_ORDER == ("rlm", "rlm-restricted", "rlm-nosubcalls", "b2", "b1", "b3")
    assert ARM_PROFILE["rlm-restricted"] == RESIDENT_PROFILE
    assert ARM_PROFILE["rlm-restricted"] == ARM_PROFILE["rlm"]


def test_restriction_is_off_by_default():
    """`run_episode` is also the S1 and S3 path. If this ever defaults true,
    two passed gates start measuring a different system than they passed on."""
    sig = inspect.signature(run_episode)
    assert sig.parameters["restrict_chunks"].default is False


# --------------------------------------------------------------------------- #
# the handicap, through a real sandbox
# --------------------------------------------------------------------------- #


@pytestmark_win
async def test_opaque_chunks_carry_no_text(session):
    """The corpus never crosses the pipe: the child builds handles from a count."""
    await session.setvar("chunks", {"__opaque_chunks__": 3})
    out = await session.exec_cell("print(len(chunks)); print(repr(chunks[1]))")
    assert out.stdout.splitlines() == ["3", "<chunk 1 of 3>"]


@pytestmark_win
@pytest.mark.parametrize("expr", [
    "len(chunks[0])",
    "chunks[0][:10]",
    "'needle' in chunks[0]",
    "chunks[0].lower()",
    "chunks[0].split()",
    "list(chunks[0])",
    "chunks[0] + 'x'",
    "f'ask about {chunks[0]}'",     # the placeholder-interpolation trap
    "'{}'.format(chunks[0])",
    "str(chunks[0])",
])
async def test_reaching_for_chunk_text_is_refused_with_the_route(session, expr):
    """Every scanning habit a root has must fail loudly AND name `llm_query`.
    A bare AttributeError would read as a bug in the root's own code, and it
    would retry rather than delegate -- which is the behaviour under test."""
    await session.setvar("chunks", {"__opaque_chunks__": 2})
    out = await session.exec_cell(f"print({expr})")
    assert "TypeError" in out.traceback
    assert "llm_query" in out.traceback


@pytestmark_win
async def test_a_handle_cannot_be_interpolated_into_a_question(session):
    """The trap that makes this arm silently measure nothing.

    With `__str__` aliased to `__repr__`, f"...{chunks[i]}..." interpolates the
    PLACEHOLDER "<chunk 0 of 2>" instead of raising. The root then thinks it
    embedded the document, sends the question with no `chunk=`, and
    `window_key(None, call_id)` gives every such call its own window -- one
    never-reused slot burned per call, to ask about a placeholder. No error is
    raised anywhere and the answers get scored."""
    await session.setvar("chunks", {"__opaque_chunks__": 2})
    out = await session.exec_cell(
        "q = f'what is in {chunks[0]}?'\nprint(q)")
    assert "TypeError" in out.traceback
    assert "<chunk 0 of 2>" not in out.stdout


@pytestmark_win
async def test_repr_still_works_so_the_root_can_plan(session):
    """Denial is about content, not about knowing what you hold."""
    await session.setvar("chunks", {"__opaque_chunks__": 7})
    out = await session.exec_cell("print(repr(chunks[3])); print(len(chunks))")
    assert out.stdout.splitlines() == ["<chunk 3 of 7>", "7"]


@pytestmark_win
async def test_a_handle_still_reaches_the_sub_model(session):
    """The handle is useless for reading and sufficient for asking. The mock
    dispatcher echoes the composed prompt, so the resolved chunk is visible."""
    await session.setvar("chunks", {"__opaque_chunks__": 4})
    out = await session.exec_cell(
        "print(await llm_query('who?', chunk=chunks[2]))")
    # The parent resolves the ref; this session has no corpus behind it, so
    # what matters here is that the call CROSSED rather than raising.
    assert "MOCK:" in out.stdout


@pytestmark_win
async def test_unrestricted_chunks_are_still_plain_strings(session):
    """The default path is unchanged -- this is the `rlm` arm's behaviour and
    the S4 re-validation depends on it being byte-identical."""
    await session.setvar("chunks", ["alpha beta", "gamma"])
    out = await session.exec_cell(
        "print(len(chunks)); print(chunks[0][:5]); print('beta' in chunks[0])")
    assert out.stdout.splitlines() == ["2", "alpha", "True"]


# --------------------------------------------------------------------------- #
# parent-side resolution: the half the sandbox cannot be trusted with
# --------------------------------------------------------------------------- #


def test_a_handle_resolves_to_its_own_chunk():
    from rlm.episode import resolve_chunk_ref
    chunks = ["alpha", "beta", "gamma"]
    assert resolve_chunk_ref(2, chunks) == "gamma"
    assert resolve_chunk_ref(0, chunks) == "alpha"


@pytest.mark.parametrize("bad", [3, -1, 99])
def test_an_out_of_range_handle_is_refused_not_clamped(bad):
    """Clamping would answer about the wrong chunk and be scored as a real
    answer -- undetectable after the fact, which is why this raises."""
    from rlm.errors import RlmError
    from rlm.episode import resolve_chunk_ref
    with pytest.raises(RlmError, match="outside the corpus"):
        resolve_chunk_ref(bad, ["alpha", "beta", "gamma"])


@pytest.mark.parametrize("bad", [True, False])
def test_a_bool_handle_is_refused(bad):
    """`bool` is an `int` subclass, so `chunk=True` would silently resolve to
    chunk 1 without this."""
    from rlm.errors import RlmError
    from rlm.episode import resolve_chunk_ref
    with pytest.raises(RlmError, match="integer index"):
        resolve_chunk_ref(bad, ["alpha", "beta"])


@pytest.mark.parametrize("bad", ["1", 1.0, None.__class__])
def test_a_non_integer_handle_is_refused(bad):
    from rlm.errors import RlmError
    from rlm.episode import resolve_chunk_ref
    with pytest.raises(RlmError, match="integer index"):
        resolve_chunk_ref(bad, ["alpha", "beta"])


# --------------------------------------------------------------------------- #
# the restricted arm's own root prompt
# --------------------------------------------------------------------------- #


def test_the_shipped_config_pins_a_restricted_root_prompt(valid_cfg):
    """Without it the arm runs on root.v3, whose tip 2 tells the root to scan
    `chunks` -- which this arm makes raise. Measured consequence: 83 of 149
    steps in the smoke's codeqa-01 episode were prose, not code."""
    ref = valid_cfg.scaffold.prompts.root_restricted
    assert ref is not None
    assert ref.sha256, "an unpinned prompt is a prompt that can drift silently"
    assert ref.path != valid_cfg.scaffold.prompts.root.path


def test_the_restricted_prompt_does_not_tell_the_root_to_scan(valid_cfg):
    """The one passage that has to differ. root.v3 says a regex or keyword scan
    over `chunks` is 'free and exact'; here it raises."""
    reg = valid_cfg.prompt_registry().load()
    restricted = reg.render_root("needle", restricted=True)
    plain = reg.render_root("needle")
    assert "free and exact" in plain
    assert "free and exact" not in restricted
    assert "ChunkRef" in restricted and "ChunkRef" not in plain
    assert restricted != plain


def test_the_plain_root_prompt_is_untouched(valid_cfg):
    """root.v3 is the `rlm` arm's prompt and the S4 re-validation depends on it
    being byte-identical to what S4 ran."""
    reg = valid_cfg.prompt_registry().load()
    assert reg.hashes()["root.file"] == valid_cfg.scaffold.prompts.root.sha256


def test_asking_for_a_restricted_prompt_that_is_not_configured_is_refused(valid_cfg):
    """Silently falling back to root.v3 would run the arm on a prompt that
    lies about its environment -- prompt mismatch scored as a delegation
    result."""
    from rlm.config import PromptRegistry
    from rlm.errors import ConfigError
    p = valid_cfg.scaffold.prompts
    bare = PromptRegistry.from_files(
        root_path=p.root.path, leaf_prefix_path=p.leaf_prefix.path,
        strategy_paths={"needle": p.strategy_templates.needle.path}).load()
    with pytest.raises(ConfigError, match="root_restricted"):
        bare.render_root("needle", restricted=True)


def test_the_restricted_arm_has_its_own_wall_clock(valid_cfg):
    """§8's 1300 s is derived for 604 sub-calls x 2.78 s. This arm delegates for
    every READ, not only every question, so it sits at that ceiling on arrival:
    synth-01 was killed at 1,306 s having completed 629 leaf calls."""
    b = valid_cfg.scaffold.budgets
    assert b.restricted_max_wall_clock_s is not None
    assert b.restricted_max_wall_clock_s > b.max_wall_clock_s


def test_the_restricted_wall_clock_covers_the_sub_call_budget(valid_cfg):
    """The rule the number encodes: the wall clock must not kill an episode
    BEFORE it can spend the sub-calls §8 already allows it. At the measured
    2.08 s/call, `max_subcalls` needs ~1,926 s."""
    b = valid_cfg.scaffold.budgets
    assert b.restricted_max_wall_clock_s >= b.max_subcalls * 2.08


def test_the_shared_wall_clock_is_untouched(valid_cfg):
    """rlm/b1/b2/b3 keep the pre-registered threshold; only the new arm differs,
    which is exactly why the difference has to be stated beside its margins."""
    assert valid_cfg.scaffold.budgets.max_wall_clock_s == 1300
