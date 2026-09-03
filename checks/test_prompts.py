# checks/test_prompts.py
import hashlib
import re
from pathlib import Path

import pytest

import rlm as _rlm
PKG_PROMPTS = Path(_rlm.__file__).resolve().parent / "_data" / "prompts"


PROMPTS = PKG_PROMPTS
FILES = ["root.v1.md", "root.v2.md", "root.v3.md", "root.v4.md",
         "root-nosubcalls.v1.md", "leaf-prefix.v1.md",
         "strat-needle.v1.md", "strat-aggregation.v1.md",
         "strat-aggregation.v2.md",
         "strat-synthesis.v1.md", "strat-codeqa.v1.md", "strat-default.v1.md",
         # §8's baseline arms (S4). Listed here, and not merely pinned in
         # config.yaml, because the header shape is load-bearing rather than
         # decorative: `PromptRegistry._strip_changelog` only strips THIS form,
         # and a header it does not recognise is a header the model reads.
         "b1-single-shot.v1.md", "b2-leaf-summary.v1.md",
         "b2-root-final.v1.md", "b3-single-shot.v1.md",
         # benchmark v2 (Task 20): one strategy block per v2 category, per
         # root arm -- the `rlm` arm's three teach `llm_query`/delegation,
         # the `-nosubcalls` twins never name a sub-model (the runtime
         # refuses `llm_query` in that arm; the prompt must agree).
         "strat-linear-semantic.v1.md", "strat-interactive.v1.md",
         "strat-code-solvable.v1.md", "strat-linear-semantic-nosubcalls.v1.md",
         "strat-interactive-nosubcalls.v1.md", "strat-code-solvable-nosubcalls.v1.md"]

#: The two recorded S1 A/B arms. Their bytes ARE the published 6/6-vs-6/6
#: result: editing either one retroactively invalidates it, so the hashes are
#: pinned here as well as in milestones/s1/RESULTS.md. `root.v3.md` exists precisely so
#: that teaching the `chunk=` form never requires touching them.
S1_AB_ARM_HASHES = {
    "root.v1.md": "1c58a5813e7d62cf2721843b01106b2e59cb5734b95a38e660f81229fad6f24f",
    "root.v2.md": "6ae1a35aaf36be681eadf9e5e685b8dcd586067c53126e116af116e877ff7215",
}


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
    for name in ("root.v1.md", "root.v2.md", "root.v3.md"):
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


def test_the_recorded_s1_ab_arms_are_never_edited():
    """v1 is the pinned S1 winner and v2 its arm; both are historical record.
    A diff to either is not a prompt improvement, it is a retroactive edit to a
    published measurement -- which is what `root.v3.md` exists to avoid."""
    for name, expected in S1_AB_ARM_HASHES.items():
        actual = hashlib.sha256((PROMPTS / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} was edited; the S1 A/B record is broken"


def _child_llm_query_signature():
    """(positional, keyword-only) parameter names of the `llm_query` the
    sandbox actually injects, read from `rlm/sandbox/child.py` by AST.

    Parsed, never imported: importing that module runs the sandbox bootstrap
    (os.dup2 on fds 0/1, an event loop, an audit hook) in the test process.
    """
    import ast
    src = (Path(__file__).resolve().parents[1] / "src" / "rlm" / "sandbox" / "child.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_llm_query_template")
    return ([a.arg for a in fn.args.args], [a.arg for a in fn.args.kwonlyargs])


def test_v3_teaches_the_chunk_form_of_the_real_injected_signature():
    """§7 #2's gap: the scaffold composes `[prefix][chunk][question]` only when
    the call supplies `chunk=`, and the PINNED prompt is what the running root
    reads. v3 must therefore teach the kwarg, name it exactly as C1 injects it,
    and never demonstrate the hand-concatenated form again."""
    v3 = (PROMPTS / "root.v3.md").read_text(encoding="utf-8")
    positional, kwonly = _child_llm_query_signature()
    assert positional == ["question"] and kwonly == ["chunk", "role"], (
        "child.py's llm_query signature moved; root.v3.md teaches the old one")

    body = v3.split("-->", 1)[1]
    signature = next(ln for ln in body.splitlines() if ln.startswith("- `await llm_query("))
    for name in positional + kwonly:
        assert f"{name}" in signature, f"the taught signature omits {name!r}"
    assert "chunk: str | None = None" in signature  # optional, not a new requirement

    assert "llm_query(question, chunk=chunks[i])" in body, "v3 must SHOW the chunk= form"
    assert 'llm_query(chunks[i] + ' not in body, "the hand-composed form must not be demonstrated"
    # The single-string form still works (`chunk=None`) and v3 must say so
    # rather than present the kwarg as a breaking change.
    assert "llm_query(prompt)" in body and "may be omitted" in body


def test_v3_is_root_v1_plus_exactly_the_chunk_form():
    """v1 won the S1 A/B, so its content is the baseline: v3 may add the
    `chunk=` guidance and change nothing else. The three lines below are the
    entire change surface -- they are the lines that STATE the old single-string
    API, and leaving them in place beside the new form is what would keep the
    root emitting `chunk=None`."""
    body_v1 = (PROMPTS / "root.v1.md").read_text(encoding="utf-8").split("-->", 1)[1]
    body_v3 = (PROMPTS / "root.v3.md").read_text(encoding="utf-8").split("-->", 1)[1]

    kept = set(body_v3.splitlines())
    removed = [ln for ln in body_v1.splitlines() if ln.strip() and ln not in kept]
    assert len(removed) == 3, f"v3 dropped more of v1 than the API lines: {removed}"
    assert removed[0].startswith('- `await llm_query(prompt: str, role:')
    assert removed[1] == "Compose every sub-call prompt this way:"
    assert removed[2] == r'answer = await llm_query(chunks[i] + "\n\n" + question)'

    # v1's tips section, byte-identical, and still the END of the file: the
    # scaffold appends the selected strategy block immediately after it.
    tips = body_v1[body_v1.index("# Tips"):]
    assert tips in body_v3, "v1's tips section must be carried over verbatim"
    assert body_v3.endswith(tips), "the closing strategy-block sentence must stay last"


def test_config_pins_the_v3_root_prompt():
    """The pin is the only thing that decides what the running root reads."""
    from rlm.config import load_config, resolve_prompt_path
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.scaffold.prompts.root.path.name == "root.v3.md"
    actual = hashlib.sha256((PROMPTS / "root.v3.md").read_bytes()).hexdigest()
    assert cfg.scaffold.prompts.root.sha256 == actual


def test_prompt_promise_matches_the_configured_extractor():
    """D16: the file text is generated from cell_extraction; they cannot disagree."""
    from rlm.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    for name in ("root.v1.md", "root.v2.md", "root.v3.md"):
        text = (PROMPTS / name).read_text(encoding="utf-8")
        if cfg.scaffold.cell_extraction.select == "first":
            assert "only the first runs" in text
        else:
            assert "only the last runs" in text


def test_extraction_shaped_strategies_carry_the_evidence_span_check():
    for name in ("strat-needle.v1.md", "strat-aggregation.v1.md",
                 "strat-aggregation.v2.md", "strat-synthesis.v1.md",
                 # Task 20: linear-semantic aggregates over labelled records
                 # and carries the same evidence-span check; its nosubcalls
                 # twin keeps the check verbatim (it never names a sub-model).
                 "strat-linear-semantic.v1.md", "strat-linear-semantic-nosubcalls.v1.md"):
        text = (PROMPTS / name).read_text(encoding="utf-8").lower()
        assert "evidence" in text, f"{name} missing the R12/R5 evidence-span check"


def test_the_pinned_aggregation_template_counts_for_an_overlapping_chunker():
    """The template the root actually runs must match the chunker it is given.

    `chunks` has been OVERLAPPING windows since §7 #2 (window 1,024 / stride
    768): a third of the corpus by tokens sits in two windows, by construction.
    Two of the pinned v1 template's instructions are wrong under that geometry
    -- summing per-chunk counts double-counts every item in an overlap, and
    stitching a chunk's tail to the next chunk's head duplicates an item the
    overlap already delivers whole. §8 makes aggregation the category that
    forces coverage and punishes sampling, so this lands as a wrong answer in
    the arm being measured, not as an error anyone sees.
    """
    from rlm.config import load_config, resolve_prompt_path

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    text = resolve_prompt_path(
        Path(cfg.scaffold.prompts.strategy_templates["aggregation"].path)
    ).read_text(
        encoding="utf-8").lower()
    assert "`chunks` overlaps" in text, "the template never says the windows overlap"
    assert "never sum per-chunk counts" in text
    assert "not** stitch the tail" in text
    assert "context" in text, "no non-repeating view named for occurrence counts"


def test_config_pins_match_the_files_on_disk():
    from rlm.config import load_config, resolve_prompt_path
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    for path, pinned in cfg.pinned_prompt_hashes().items():
        # `pinned_prompt_hashes` keys are the paths as DECLARED, because that is what
        # `config_snapshot` records; the file they name now lives in the package.
        actual = hashlib.sha256(resolve_prompt_path(Path(path)).read_bytes()).hexdigest()
        assert actual == pinned, f"{path} drifted from its config pin"


# CONTROLLER RULING (Task 20, 2026-09-03): retargeted from v1's five
# categories to v2's three (linear_semantic, interactive, code_solvable).
# "needle" was Task 7's arbitrary pick to exercise the mechanism, made
# before v2's category set existed; no brief requires nosubcalls twins for
# v1 categories, and authoring five unused prompt files to satisfy a stale
# fixture would be waste. See task-20-report.md. The xfail mark is removed:
# the six files it was blocked on are landed and this passes on its merits.
def test_render_root_no_subcalls_uses_the_nosubcalls_body_and_block(nosubcalls_cfg):
    reg = nosubcalls_cfg.prompt_registry().load()
    text = reg.render_root("linear_semantic", no_subcalls=True)
    assert "llm_query" not in text and "sub-call" not in text.lower()


V2_BLOCKS = ["strat-linear-semantic.v1.md", "strat-interactive.v1.md", "strat-code-solvable.v1.md"]


@pytest.mark.parametrize("name", V2_BLOCKS)
def test_v2_blocks_have_headers_and_the_rlm_variants_teach_llm_query_or_code(name):
    assert (PROMPTS / name).read_text(encoding="utf-8").startswith("<!-- changelog")
    body = _body(name)
    assert body.lstrip().startswith("# Strategy: ")
    if "code-solvable" not in name:
        assert "llm_query" in body


@pytest.mark.parametrize("name", [n.replace(".v1.md", "-nosubcalls.v1.md") for n in V2_BLOCKS])
def test_nosubcalls_blocks_never_name_the_sub_model(name):
    body = _body(name).lower()
    for banned in ("llm_query", "sub-model", "sub-call", "asyncio.gather", "delegat"):
        assert banned not in body, (name, banned)


def test_the_interactive_blocks_teach_env():
    for name in ("strat-interactive.v1.md", "strat-interactive-nosubcalls.v1.md"):
        body = _body(name)
        assert "env.search(" in body and "env.open(" in body and "env.window(" in body


def _body(name):  # header stripped, like the loader
    from rlm.config import _strip_changelog
    return _strip_changelog((PROMPTS / name).read_text(encoding="utf-8"))


def test_v4_is_v3_with_exactly_line_37_changed():
    v3, v4 = _body("root.v3.md").splitlines(), _body("root.v4.md").splitlines()
    assert len(v3) == len(v4)
    diff = [(a, b) for a, b in zip(v3, v4) if a != b]
    assert len(diff) == 1
    old, new = diff[0]
    assert old.startswith("`llm_query` reaches a small, fast, stateless model.")
    assert "the same model as you" in new and "no REPL, no memory between calls" in new


def test_nosubcalls_body_describes_a_repl_with_no_sub_model():
    body = _body("root-nosubcalls.v1.md")
    for banned in ("llm_query", "sub-model", "sub-call", "Sub-call", "delegat", "leaf"):
        assert banned not in body, banned
    assert "final_answer(value)" in body and "`chunks: list[str]`" in body
    assert body.rstrip().endswith(
        "A strategy block for this task's declared category follows. The scaffold "
        "selected it from the task's category; you do not choose it, and where it is "
        "more specific than the tips above, it wins.")


# CONTROLLER RULING (Task 8 fix round 1): the brief's Step 3 derivation for
# root-nosubcalls.v1.md is v4 minus lines PLUS two mandated full-paragraph
# rewrites that are neither "kept verbatim" nor a renumbered tip -- the
# original two-case invariant below (kept-verbatim / renumbered-tip) was
# incomplete and could never pass for a correct derivation, because it had
# no way to admit these. Rather than leave that permanently strict-xfailed
# (which would silently retire the very guarantee this test exists to
# enforce -- that root-nosubcalls.v1.md is v4 minus lines, not a fresh
# re-authoring), the invariant gets a third, explicit case: an allowlist of
# exactly the two authorised rewrites, pinned to their exact before/after
# text. Every OTHER line must still be kept-verbatim or a correctly
# renumbered tip; an unauthorised rewrite anywhere else still fails.
_AUTHORISED_REWRITES = {
    # body line 9: drop the delegation clause (§14.3's whole point -- this
    # arm's root does not know a sub-model exists).
    "You never see the context as text. It is already loaded as Python "
    "objects in a REPL that persists across your turns. You act by writing "
    "code that inspects those objects, and by delegating chunk-level "
    "reading to a cheap sub-model. You are an orchestrator, not a reader.":
        "You never see the context as text. It is already loaded as Python "
        "objects in a REPL that persists across your turns. You act by "
        "writing code that inspects those objects. You are a programmer "
        "over the context, not a reader of it.",
    # the "# Budgets" paragraph: drop the "Sub-calls," cap and the
    # sub-call-spending sentence -- there is nothing to spend sub-calls on.
    "Sub-calls, tokens, and wall-clock are capped per episode by the "
    "scaffold. The caps are enforced, not advisory; you cannot raise them "
    "and asking for more has no effect. A breach kills the episode with no "
    "answer at all. Spend sub-calls only on text that genuinely has to be "
    "read; spend code freely.":
        "Tokens and wall-clock are capped per episode by the scaffold. The "
        "caps are enforced, not advisory; you cannot raise them and asking "
        "for more has no effect. A breach kills the episode with no answer "
        "at all.",
}


def test_nosubcalls_body_is_v4_minus_only_the_sub_call_lines():
    v4 = [l for l in _body("root.v4.md").splitlines() if l.strip()]
    ns = [l for l in _body("root-nosubcalls.v1.md").splitlines() if l.strip()]
    kept = [l for l in ns if l in v4]
    remainder = [l for l in ns if l not in v4]
    rewritten = [l for l in remainder if l in _AUTHORISED_REWRITES.values()]
    renumbered = [l for l in remainder if l not in rewritten]
    # every authorised rewrite's ORIGINAL must actually be the v4 line it
    # replaces -- pins both sides, not just the new text.
    for old, new in _AUTHORISED_REWRITES.items():
        assert old in v4, old
        assert new in ns, new
    # everything else not in v4 is a renumbered tip (same text after the "N. ")
    assert all(any(l.split(". ", 1)[1] == k.split(". ", 1)[1] for k in v4 if ". " in k)
               for l in renumbered if ". " in l), renumbered
    assert len(kept) + len(rewritten) + len(renumbered) == len(ns)
