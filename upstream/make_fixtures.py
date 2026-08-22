"""S2 chunk-size QUALITY sweep — fixture generator (spec §7 #2, v0.2.4).

WHAT THIS BUILDS AND WHY IT LOOKS LIKE THIS.

§7 #2 was promoted to the first optimization because measuring the (finally
working) leaf falsified the shipped 32,768-token chunk size on QUALITY, not on
wall-clock: against a chunk that provably CONTAINS the needle, ~550 tokens
returns the exact key, ~1K and ~10K return `NONE`, and ~30K returned the true
UUID **with its last character altered** while flatly denying a phrase
literally present in the chunk. So the sweep's fixtures have to make three
different failures separable, at six sizes, at a controlled depth:

  * **LITERAL** — the answer string is in the chunk, character for character.
    A wrong answer here is unambiguous.
  * **PARAPHRASE** — the fact is in the chunk but stated only in other words,
    so the answer string occurs NOWHERE in the document (asserted at build
    time) and no regex can find it. This is the case the `contains` checker
    cannot score for the model, and the case a lossy long-context read fails
    first.
  * **ABSENT** — the fact is not in the chunk at all and the correct answer is
    a refusal. This is the only question type that measures the FALSE-POSITIVE
    rate, which is what makes a confabulation dangerous: a root that submits a
    fluent wrong UUID has been lied to, not merely failed.

The ABSENT question is worded *identically* to the LITERAL one, changing only
the organisation named — so the two cells differ in exactly one thing (whether
the fact is there), and the wording presupposes the key exists. That
presupposition is deliberate: it is the maximum-pressure case, and a refusal
under pressure is the property the runtime actually needs.

**Non-benchmark and uncontaminatable, by construction.** Random entity->UUID
pairings and coined organisation names cannot exist in any training corpus
(§8's contamination precondition 2), the filler is generated, and none of it
goes near the frozen benchmark — which does not exist yet, and whose authoring
is a separate S2 deliverable. Re-deriving the chunk-size default BEFORE the
benchmark is authored is the opposite of test-set tuning (§7 #2's own note).

**Sizes are measured in LEAF tokens, never characters.** `chunk_size` means
"target-leaf tokens" (§5 C2), so the fit is a binary search over word
boundaries with an INJECTED counter, and `main()` injects the leaf server's own
`/tokenize`. The offline proxy (`approx_tokens`) exists so the generator is
testable with no server; a manifest built with it is stamped as such and
`milestones/s2/run_sweep.py` REFUSES it — a sweep calibrated on a 4-chars-per-token guess
would measure the guess.

Determinism: everything derives from `--seed` (default 1). Per-cell RNG is
seeded from the string `s2:{seed}:{size}:{position}` — distinct facts per cell
(so no cell can be answered from another cell's needle) but reproducible from
the seed alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `uv run milestones/s2/make_sweep_fixtures.py`
    sys.path.insert(0, str(REPO_ROOT))

# Imported, not re-implemented. `control_truncate` IS "the longest whole-word
# prefix whose token count is <= N, by binary search, deterministic for a given
# counter" -- exactly the fit this generator needs -- and `approx_tokens` is the
# repo's one stated 4-chars-per-token proxy. A third copy of either would be a
# third thing to keep honest.
# --- inlined from the project's milestones/s1/make_fixtures.py so this file stands alone ---
import urllib.request  # noqa: E402

def approx_tokens(text: str) -> int:
    """The offline token proxy: ~4 chars per token.

    Stated, not measured -- deliberately the same approximation
    `rlm.dispatcher.MockDispatcher.count_tokens` documents, so the one place
    in this repo that guesses at token counts guesses the same way twice. It
    is monotonic in prefix length and deterministic, which is all
    `control_truncate`'s binary search requires. It is NOT what the control
    document is cut with: `main()` passes the leaf server's own tokenizer.
    """
    return (len(text) + 3) // 4


def _word_ends(text: str) -> list[int]:
    """Character offsets one past each whitespace-delimited word.

    Dropped when this bundle was extracted to be standalone, which made
    `control_truncate` raise `NameError` on its first call and took the whole
    reproducer down before it generated a single fixture. Restored verbatim
    from `milestones/s1/make_fixtures.py`, the implementation every fixture in this
    project was actually built with.
    """
    return [m.end() for m in re.finditer(r"\S+", text)]


def control_truncate(text: str, n_tokens: int,
                      count: Callable[[str], int] | None = None) -> str:
    """Keep the first `n_tokens` tokens of `text`, drop the rest (§9 S1 (a)).

    "The first N tokens" is resolved to the longest WHOLE-WORD prefix whose
    token count is <= N, by binary search over word boundaries -- so the cut
    never lands mid-word and never exceeds the budget. Character offsets come
    from the original string, so newlines and paragraph structure survive
    verbatim; nothing is reflowed.

    `count` defaults to `approx_tokens` (offline, deterministic, no server).
    Pass the leaf server's tokenizer to cut at a real token budget. The rule
    is deterministic for any given counter, which is what the S1 fixture test
    pins.
    """
    counter = count or approx_tokens
    ends = _word_ends(text)
    if not ends or n_tokens <= 0:
        return ""
    lo, hi = 0, len(ends)          # number of words kept
    if counter(text) <= n_tokens:
        return text
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter(text[:ends[mid - 1]]) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:ends[lo - 1]] if lo else ""


# --------------------------------------------------------------------------- #
# writing the fixtures (needs a real token counter)
# --------------------------------------------------------------------------- #


def leaf_counter(port: int = 8081, timeout: float = 300.0) -> Callable[[str], int]:
    """`/tokenize` on the leaf server -- the only token count that means
    anything here (spec §5 C2: chunk size is measured in target-leaf tokens).
    A zero count for non-empty input is a fault, never a legitimate answer
    (recipes §serverapi: /tokenize fails silently)."""
    url = f"http://127.0.0.1:{port}/tokenize"

    def count(text: str) -> int:
        if not text:
            return 0
        req = urllib.request.Request(
            url, data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tokens = json.loads(resp.read().decode("utf-8")).get("tokens", [])
        if not tokens:
            raise RuntimeError("/tokenize returned 0 tokens for non-empty input")
        return len(tokens)

    return count

S2_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = S2_DIR / "fixtures"
MANIFEST_NAME = "manifest.json"

#: §7 #2, verbatim: "Sweep {1K, 2K, 4K, 8K, 16K, 32K} on non-benchmark
#: fixtures." Powers of two, in leaf tokens.
SIZES: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384, 32768)

#: The main sweep holds depth constant at ~50%; the position sub-study adds
#: ~10% and ~90% at the two sizes where the main sweep shows quality breaking
#: down. Which two those are is a POST-HOC choice by design -- the runner takes
#: them as arguments so the decision is visible in the command line rather than
#: hidden in a default.
MAIN_POSITION = 0.50
STUDY_POSITIONS: tuple[float, ...] = (0.10, 0.50, 0.90)

SEED = 1
QUESTION_TYPES = ("literal", "paraphrase", "absent")

#: The paraphrase needle sits just after the literal one so both share a depth
#: band without being one glued block.
PARAPHRASE_DEPTH_OFFSET = 0.03

#: A cell whose measured length misses its target by more than this is a
#: broken fixture, not a small fixture: the sweep's x-axis IS the token count.
SIZE_TOLERANCE = 0.01

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

# --------------------------------------------------------------------------- #
# Generation vocabulary. Deliberately NOT s1's pool: an S2 chunk must be
# distinguishable from an S1 fixture at a glance, and milestones/s1/tasks is a pinned,
# already-measured artifact that nothing here should be able to perturb.
# --------------------------------------------------------------------------- #

_WORDS = (
    "tidewater cistern gantry pumice lodestone brazier culm dredger fascine "
    "grommet hawser inkwell jackstay kiln limber mullion nacelle oakwood "
    "pintle quoin ratline scupper thwart underwood vane wadding yawl adze "
    "bevel chamfer dado escutcheon fillet groove haunch inlay kerf lap "
    "mitre notch ogee plinth rabbet sill tenon undercut veneer wainscot "
    "consignor drayage freightage haulage lighterage portage stowage "
    "tallage ullage wharfage cordage dunnage lashing pallet strapping "
    "manifest waybill docket tariff levy duty warrant bond writ lien "
    "surveyor assessor factor broker chandler cooper drayman ferrier "
    "gaffer hoistman keeper lampman marker netter overseer packer quayman "
    "reeve stower tallyman usher warder yardman auditor bailer carter"
).split()

_SYL_A = ("ald bry cen dov erl fask gorm hurn ilv jent korr lums merv nield "
          "orst pryl quin ravv selk tors").split()
_SYL_B = ("mora dale vint holt shaw regn brae carn dune fenn gild hask "
          "irme jost keld lorn mest").split()
_SYL_C = ("wick sted holm ridge combe thorpe gate ness field bourne").split()

_BODIES = ("Ledger Registry Bureau Consortium Syndicate Trust Repository "
           "Chapterhouse Exchange Assembly Guildhall Depository").split()

_SENTENCE_LEADS = (
    "The {a} tally was entered against the {b} account without amendment",
    "A second {a} return reached the {b} desk after the cut-off",
    "Nothing in the {a} schedule disputes the {b} valuation",
    "Bonded storage for the {a} lot remains held at the {b} shed",
    "The {a} weight was struck twice against the {b} standard",
    "Release of the {a} parcel to the {b} carrier was witnessed",
    "No surveyor initialled the {a} column of the {b} return",
    "Frost kept the {a} gang at the {b} stage until first light",
    "The {a} draft was endorsed over to the {b} holder in blank",
    "Two {a} seals were struck and one {b} seal was reissued",
)


def _coined(rng: random.Random, *, unlike: tuple[str, ...] = ()) -> str:
    """A coined, capitalised name that is not a near-twin of `unlike`.

    Near-twins are rejected on purpose: two names differing by a letter turn a
    reading test into a proof-reading test, and a failure caused by that is a
    fixture defect wearing an R5 costume.
    """
    banned_head = {n[:3].lower() for n in unlike}
    banned_tail = {n[-4:].lower() for n in unlike}
    for _ in range(400):
        name = (rng.choice(_SYL_A) + rng.choice(_SYL_B) + rng.choice(_SYL_C)).capitalize()
        if name[:3].lower() in banned_head or name[-4:].lower() in banned_tail:
            continue
        if name in unlike:
            continue
        return name
    raise AssertionError(f"could not coin a name distinct from {unlike!r}")


def _org(rng: random.Random, *, unlike: tuple[str, ...] = ()) -> tuple[str, str]:
    """`("the Fenngate Ledger", "Fenngate")` — an organisation that has never
    existed, so no model can know its archive key."""
    coined = _coined(rng, unlike=unlike)
    return f"the {coined} {rng.choice(_BODIES)}", coined


def _sentence(rng: random.Random) -> str:
    lead = rng.choice(_SENTENCE_LEADS).format(
        a=rng.choice(_WORDS), b=rng.choice(_WORDS))
    tail = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 15)))
    return f"{lead}, {tail}."


def _paragraph(rng: random.Random, index: int) -> str:
    head = f"ENT-{rng.randrange(10_000, 99_999)} / {rng.randrange(0x1000, 0xffff):04x}"
    body = " ".join(_sentence(rng) for _ in range(rng.randint(3, 6)))
    return f"[{index:05d}] {head}\n{body}"


def make_filler(rng: random.Random, n_words: int) -> str:
    """`n_words`-ish words of paragraph-structured filler.

    Deterministic in `rng`. Contains no UUID-shaped string and no coined
    organisation name, both asserted by `build_cell`.
    """
    paragraphs: list[str] = []
    words = 0
    index = 0
    while words < n_words:
        index += 1
        para = _paragraph(rng, index)
        paragraphs.append(para)
        words += len(para.split())
    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------- #
# fitting to a TOKEN target, and placing a needle at a TOKEN depth
# --------------------------------------------------------------------------- #


def _paragraph_offsets(text: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer(r"\n\n", text)]


def boundary_at_token_target(text: str, target_tokens: int,
                             count: Callable[[str], int]) -> int:
    """Char offset of the first paragraph boundary holding >= `target_tokens`.

    An ABSOLUTE token target, not a fraction of `text`: the needle's depth is
    a property of the FINISHED chunk, and the filler this searches is shorter
    than the finished chunk by exactly the needles that will be spliced into
    it. Taking a fraction of the filler instead would place a "50%" needle at
    ~42% of a 1K cell (measured while writing this) — the needles' own width,
    silently biasing the one axis the position sub-study exists to vary.

    Binary search over the paragraph-boundary list, so the real tokenizer is
    called ~log2(#paragraphs) times rather than once per paragraph: at 32K that
    is ~10 round trips instead of ~700, and the two answers are identical.
    """
    offsets = _paragraph_offsets(text)
    if len(offsets) < 2 or target_tokens <= 0:
        return 0
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if count(text[:offsets[mid]]) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    # The NEAREST boundary, not the first one past the target: a paragraph is
    # 100-170 leaf tokens, so always rounding up biases every needle deeper by
    # up to a whole paragraph, which at 4K is ~4% of the depth axis.
    if lo > 0:
        over = count(text[:offsets[lo]]) - target_tokens
        under = target_tokens - count(text[:offsets[lo - 1]])
        if under < over:
            return offsets[lo - 1]
    return offsets[lo]


def fit_to_tokens(text: str, target: int, count: Callable[[str], int]) -> str:
    """Longest whole-word prefix of `text` holding <= `target` tokens.

    This is `s1.make_fixtures.control_truncate` — the same rule, the same
    binary search, the same injected counter — named for what it does here.
    """
    return control_truncate(text, target, count)


# --------------------------------------------------------------------------- #
# one cell = one chunk + its three questions
# --------------------------------------------------------------------------- #


def cell_id(size_tokens: int, position: float) -> str:
    return f"s2-{size_tokens}-p{round(position * 100):02d}"


def _cell_rng(seed: int, size_tokens: int, position: float) -> random.Random:
    """Seeded from a STRING, so the stream is reproducible across processes and
    platforms (CPython hashes str/bytes seeds with sha512; `hash()` of a str is
    salted per-process and must never be used for this)."""
    return random.Random(f"s2:{seed}:{size_tokens}:{round(position * 1000)}")


def build_cell(count: Callable[[str], int], *, size_tokens: int,
               position: float = MAIN_POSITION, seed: int = SEED) -> dict:
    """Build ONE chunk of ~`size_tokens` leaf tokens with its three questions.

    Refuses rather than degrades. Every claim the sweep will make about this
    fixture is asserted here, against the injected counter and the built text:
    the size is within tolerance, the literal key occurs exactly once and is
    the only UUID-shaped string in the document, the paraphrase answer occurs
    nowhere, and the absent organisation is named nowhere.
    """
    rng = _cell_rng(seed, size_tokens, position)

    # -- the three facts ---------------------------------------------------- #
    lit_org, lit_coined = _org(rng)
    lit_key = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    literal_needle = (
        f"[custody note] The archive key issued to {lit_org} is {lit_key}. "
        f"It was cut once, for that holder only, and no duplicate key was ever "
        f"issued under that name."
    )

    par_org, par_coined = _org(rng, unlike=(lit_coined,))
    given = _coined(rng, unlike=(lit_coined, par_coined))
    family = _coined(rng, unlike=(lit_coined, par_coined, given))
    paraphrase_answer = f"{given} {family}"
    paraphrase_needle = (
        f"[custodial register] {par_org} enters each custodian across two "
        f"separate lines and never prints a custodian's name in full.\n"
        f"GIVEN . . . . . . {given}\n"
        f"SEAL  . . . . . . intact\n"
        f"TERM  . . . . . . open\n"
        f"FAMILY  . . . . . {family}\n"
        f"Those two entries are the only custodian of record for this trust; "
        f"no other custodian is entered anywhere in the register."
    )

    abs_org, abs_coined = _org(rng, unlike=(lit_coined, par_coined, given, family))

    # -- filler sized so that filler + needles lands ON the target ---------- #
    #
    # Tokenization is not additive across a splice, so "filler_target =
    # size - needle_tokens" is an estimate, not an identity. The assembly is
    # therefore a short fixed-point loop: assemble, measure, correct the filler
    # budget by the residual, re-assemble. It converges in one or two passes
    # (the map is ~1:1 in tokens) and REFUSES if it does not -- a cell that
    # misses its own x-axis value is not a fixture.
    literal_tokens = count(f"\n\n{literal_needle}\n\n")
    needle_tokens = count(f"\n\n{literal_needle}\n\n{paraphrase_needle}\n\n")
    slack = max(8, int(size_tokens * SIZE_TOLERANCE))
    # ~1.7 leaf tokens per word for this class of English (s1's measured
    # calibration), over-generated 40% so the fit below always has slack to
    # cut down into.
    pool = make_filler(rng, int(size_tokens / 1.7 * 1.4) + 80)
    filler_target = max(1, size_tokens - needle_tokens)

    def assemble(budget: int) -> tuple[str, int]:
        filler = fit_to_tokens(pool, budget, count)
        # Absolute token targets in the FINISHED chunk. The paraphrase target
        # subtracts the literal needle's own length, because that needle is
        # spliced in ahead of it and would otherwise push it deeper than asked.
        lit_at = boundary_at_token_target(
            filler, int(position * size_tokens), count)
        par_at = boundary_at_token_target(
            filler,
            int(min(position + PARAPHRASE_DEPTH_OFFSET, 0.99) * size_tokens)
            - literal_tokens,
            count)
        # Both offsets are computed on the SAME text and applied deepest-first,
        # so neither insertion moves the other's measured depth.
        if par_at <= lit_at:  # a tiny document can collapse the two boundaries
            par_at = lit_at
        spliced = filler[:par_at] + paraphrase_needle + "\n\n" + filler[par_at:]
        spliced = spliced[:lit_at] + literal_needle + "\n\n" + spliced[lit_at:]
        return spliced, count(spliced)

    text, measured = assemble(filler_target)
    for _ in range(4):
        if abs(measured - size_tokens) <= slack:
            break
        filler_target = max(1, filler_target + (size_tokens - measured))
        text, measured = assemble(filler_target)
    else:
        if measured > size_tokens:  # last resort: cut the tail, never the head
            text = fit_to_tokens(text, size_tokens, count)
            measured = count(text)

    # -- validity, asserted rather than promised ---------------------------- #
    if abs(measured - size_tokens) > slack:
        raise AssertionError(
            f"{cell_id(size_tokens, position)}: built {measured} leaf tokens "
            f"against a {size_tokens} target (slack {slack}); the sweep's "
            f"x-axis is the token count, so this is a broken fixture")
    if text.count(lit_key) != 1:
        raise AssertionError(f"literal key occurs {text.count(lit_key)} times, want 1")
    stray = {u for u in UUID_RE.findall(text)} - {lit_key}
    if stray:
        raise AssertionError(f"document contains UUID-shaped strings other than "
                             f"the planted key: {sorted(stray)[:3]}")
    if paraphrase_answer in text:
        raise AssertionError("the paraphrase answer occurs verbatim; a regex "
                             "would find it and the cell would measure nothing")
    if literal_needle not in text or paraphrase_needle not in text:
        raise AssertionError("a needle did not survive the final fit")
    for token in (abs_coined, abs_org):
        if token.lower() in text.lower():
            raise AssertionError(f"the ABSENT organisation {token!r} appears in "
                                 f"the document; the cell would measure a MISS, "
                                 f"not a false positive")

    lit_off = text.index(literal_needle)
    par_off = text.index(paraphrase_needle)

    def depth_of(offset: int) -> float:
        return round(count(text[:offset]) / measured, 4)

    lit_depth = depth_of(lit_off)
    # A needle has WIDTH: at 1K the two needles are ~16% of the chunk, so a
    # needle whose START is at 90% would have to end past the end. Refuse
    # rather than silently record a shallower depth than the cell claims.
    if abs(lit_depth - position) > 0.10:
        raise AssertionError(
            f"{cell_id(size_tokens, position)}: the literal needle landed at "
            f"depth {lit_depth:.2f}, not {position:.2f}. At {size_tokens} "
            f"tokens the two needles are {needle_tokens} tokens "
            f"({needle_tokens / size_tokens:.0%}) of the chunk and cannot be "
            f"placed that deep. Run the position sub-study at sizes where the "
            f"needles are a small fraction of the chunk.")

    return {
        "cell_id": cell_id(size_tokens, position),
        "size_tokens": size_tokens,
        "measured_tokens": measured,
        "position": position,
        "chars": len(text),
        "seed": seed,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "needles": {
            "literal": {"text": literal_needle, "char_offset": lit_off,
                        "token_depth": lit_depth},
            "paraphrase": {"text": paraphrase_needle, "char_offset": par_off,
                           "token_depth": depth_of(par_off)},
        },
        "questions": {
            "literal": {
                "type": "literal",
                # Same sentence as the absent question, one entity apart.
                "question": (
                    f"Exactly one custody note in this excerpt records the archive "
                    f"key issued to {lit_org}. What is that key? Reply with the key "
                    f"itself and nothing else."),
                "expected": lit_key,
                "expected_kind": "uuid",
                "entity": lit_org,
            },
            "paraphrase": {
                "type": "paraphrase",
                "question": (
                    f"{par_org[0].upper() + par_org[1:]} has exactly one custodian "
                    f"of record, but the register never prints that custodian's "
                    f"name in full. What is the custodian's full name? Reply with "
                    f"the given name followed by the family name, and nothing else."),
                "expected": paraphrase_answer,
                "expected_kind": "name",
                "entity": par_org,
            },
            "absent": {
                "type": "absent",
                "question": (
                    f"Exactly one custody note in this excerpt records the archive "
                    f"key issued to {abs_org}. What is that key? Reply with the key "
                    f"itself and nothing else."),
                # No expected answer: the correct answer is a refusal, and the
                # scorer must never be handed a string to match here.
                "expected": None,
                "expected_kind": "uuid",
                "entity": abs_org,
            },
        },
    }


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


def write_fixtures(count: Callable[[str], int], *, sizes: tuple[int, ...] = SIZES,
                   positions: tuple[float, ...] = (MAIN_POSITION,),
                   seed: int = SEED, out_dir: Path = FIXTURES_DIR,
                   counter_name: str = "leaf:/tokenize",
                   merge: bool = True) -> dict:
    """Build every (size, position) cell, write the chunks, write the manifest.

    `merge` keeps cells already in the manifest that this invocation did not
    rebuild — the position sub-study is run AFTER the main sweep, on sizes
    chosen from the main sweep's result, and must not delete the main sweep's
    own fixtures on its way in.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    cells: dict[str, dict] = {}
    if merge and manifest_path.exists():
        cells = json.loads(manifest_path.read_text(encoding="utf-8")).get("cells", {})

    for size in sizes:
        for position in positions:
            cell = build_cell(count, size_tokens=size, position=position, seed=seed)
            text = cell.pop("text")
            chunk_path = out_dir / f"{cell['cell_id']}.chunk.txt"
            chunk_path.write_text(text, encoding="utf-8", newline="\n")
            cell["chunk_path"] = str(chunk_path)
            cells[cell["cell_id"]] = cell

    manifest = {
        "generator": "milestones/s2/make_sweep_fixtures.py",
        "generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "token_counter": counter_name,
        "seed": seed,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cells": dict(sorted(cells.items())),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    return manifest


def load_manifest(out_dir: Path = FIXTURES_DIR) -> dict:
    return json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the S2 chunk-size sweep fixtures (spec §7 #2)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--positions", nargs="+", type=float,
                        default=[MAIN_POSITION],
                        help="needle depth(s); the main sweep is 0.5, the "
                             "sub-study adds 0.1 and 0.9 at two sizes")
    parser.add_argument("--leaf-port", type=int, default=8081)
    parser.add_argument("--out", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--offline", action="store_true",
                        help="build with the 4-chars-per-token proxy instead of "
                             "the leaf tokenizer. Smoke-test only: the manifest "
                             "is stamped 'approx-offline' and run_sweep.py "
                             "refuses to sweep it.")
    parser.add_argument("--no-merge", action="store_true",
                        help="drop cells not rebuilt by this invocation")
    args = parser.parse_args(argv)

    if args.offline:
        count, counter_name = approx_tokens, "approx-offline"
    else:
        count, counter_name = leaf_counter(args.leaf_port), "leaf:/tokenize"

    manifest = write_fixtures(
        count, sizes=tuple(args.sizes), positions=tuple(args.positions),
        seed=args.seed, out_dir=args.out, counter_name=counter_name,
        merge=not args.no_merge)
    for cid, cell in manifest["cells"].items():
        needles = cell["needles"]
        print(f"{cid}: target {cell['size_tokens']} -> measured "
              f"{cell['measured_tokens']} leaf tokens, {cell['chars']} chars; "
              f"literal at depth {needles['literal']['token_depth']:.2f}, "
              f"paraphrase at {needles['paraphrase']['token_depth']:.2f}; "
              f"key={cell['questions']['literal']['expected']}")
    print(f"wrote {args.out / MANIFEST_NAME} ({len(manifest['cells'])} cells, "
          f"counter={manifest['token_counter']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
