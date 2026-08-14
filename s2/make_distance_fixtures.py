"""S2 — INSTRUCTION-DECAY fixtures: distance and density, separated.

WHY THIS FILE EXISTS AT ALL, and why it is not just `--sizes 640 1024 2048`
against `s2/make_sweep_fixtures.py`.

`s2/REFUSAL-AB.md` §2 recorded an unplanned result with a bigger claim than the
experiment it came from: the SAME `leaf-prefix.v1.md`, unchanged, produces a
30/30 false-positive rate at a 1,024-token window and 0/21 at a 640-token window
(Fisher p = 8.7e-15). The natural reading is §4's INSTRUCTION DECAY paragraph:
instructions decay with distance from the point of generation exactly as facts
do (§7 #2's cliff), so `[system prefix][chunk][question]` puts the rules at the
maximum possible distance and a longer chunk pushes them out of reach.

**That reading is confounded, and the confound is the whole reason for this
generator.** A 640-token cell built by `make_sweep_fixtures.py` also holds FEWER
DISTRACTOR ENTITIES than a 1,024-token one — filler paragraphs each carry an
`ENT-#####/hex` binding, so entity count scales with size. Distance and density
moved together, and "the rules are back in reach" and "there is less material to
misattribute from" predict the same 640-vs-1024 result. An experiment that does
not separate them measures nothing, so this generator makes DENSITY an explicit
factor with two levels at every size:

  * **matched-density** — every size carries the SAME number of distractor
    entity bindings: `MATCHED_ENTITIES` = 3, the count measured in the existing
    640-token cells (`s2/fixtures-refusal-640-s*/s2-640-p50.chunk.txt`, 3
    `ENT-` codes each). Larger chunks are padded to length with NEUTRAL filler
    that carries no identifier-shaped token at all. Size then varies distance
    alone.
  * **natural-density** — a binding on every filler paragraph, exactly as
    `make_sweep_fixtures.py` builds them, so entity count scales with size as
    it did in the sweep and in the A/B.

At a fixed size the two levels differ ONLY in density; across sizes the matched
level differs ONLY in distance. If the false-positive rate tracks SIZE within
matched-density, distance is the cause; if it tracks DENSITY at fixed size, the
instruction-decay hypothesis is dead. That is the one comparison this file
exists to make possible.

WHAT A "DISTRACTOR ENTITY" IS HERE, stated so the operationalization can be
argued with rather than guessed at. It is an `[NNNNN] ENT-##### / hhhh`
paragraph header: a coined entity bound to an identifier-shaped value, which is
exactly the object the leaf misattributes (§10 R13's strongest artifact is
`ENT-#####:hex` pairs split between two documents, and the A/B's false positives
hand over the ONE planted UUID when asked about an organisation that owns no
key). Every other axis is held fixed by construction: the needles, the three
questions, the coined-name inventory (4 names, needles only) and the count of
UUID-shaped strings (exactly 1) are identical at every size and both densities.

EVERYTHING ELSE IS `make_sweep_fixtures.py`'s, IMPORTED. The vocabulary, the
needle text, the three question wordings, the token-target fit, the depth
placement and the validity assertions are that module's — reproduced here would
be a second thing to keep honest, and the questions have to be word-identical to
the ones the A/B asked or the replication arm is not a replication.

Determinism: everything derives from `--seed`. Per-cell RNG is seeded from the
string `s2d:{seed}:{size}:{density}`.
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
if str(REPO_ROOT) not in sys.path:  # `uv run s2/make_distance_fixtures.py`
    sys.path.insert(0, str(REPO_ROOT))

from rlm.leakcheck import identifier_tokens  # noqa: E402
from s1.make_fixtures import approx_tokens, leaf_counter  # noqa: E402
from s2.make_sweep_fixtures import (  # noqa: E402
    MANIFEST_NAME,
    PARAPHRASE_DEPTH_OFFSET,
    SIZE_TOLERANCE,
    UUID_RE,
    _coined,
    _org,
    _sentence,
    boundary_at_token_target,
    fit_to_tokens,
)

S2_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = S2_DIR / "fixtures-distance"

#: FACTOR 2 — the size axis, in leaf tokens. 640 and 1,024 are the two windows
#: the A/B compared; 2,048 extends the axis upward. All three fit one 2,560-
#: token slot under layouts A and C; 2,048 does NOT fit under layout B (the
#: repeated prefix costs another ~311 tokens), which `s2/run_distance.py`
#: reports as a structural exclusion rather than working around.
SIZES: tuple[int, ...] = (640, 1024, 2048)

#: FACTOR 3 — the two density levels.
DENSITIES: tuple[str, ...] = ("matched", "natural")

#: The matched-density entity budget: the count measured in the 640-token cells
#: the A/B actually ran (3 `ENT-` codes per chunk). Holding this fixed across
#: sizes is what turns the size axis into a pure distance axis.
MATCHED_ENTITIES = 3

#: The natural level's rate, calibrated ON the fixtures the A/B and the sweep
#: actually ran rather than chosen: `s2/fixtures-refusal-640-s*` carry 3 `ENT-`
#: bindings per 640-token chunk, `s2/fixtures-refusal-s*` 6 per 1,024 and
#: `s2/fixtures/s2-2048-p50.chunk.txt` 11 per 2,048 — one binding per ~186
#: tokens. `int(size / 186 + 0.5)` reproduces all three counts exactly (3, 6,
#: 11), so a natural-density cell here has the density of the cell whose result
#: this experiment is trying to explain.
NATURAL_TOKENS_PER_ENTITY = 186

#: The needles sit at 50% depth, as in the A/B's own 640-token cells.
POSITION = 0.50

SEED = 1


# --------------------------------------------------------------------------- #
# filler: entity-bearing and neutral
# --------------------------------------------------------------------------- #


def entity_header(rng: random.Random, index: int) -> str:
    """One distractor binding: a paragraph index, an `ENT-` code, a hex value.

    Byte-for-byte the header `make_sweep_fixtures._paragraph` writes, so a
    natural-density cell built here is the same object the sweep and the A/B
    measured.
    """
    return (f"[{index:05d}] ENT-{rng.randrange(10_000, 99_999)} / "
            f"{rng.randrange(0x1000, 0xffff):04x}")


def n_bindings(size_tokens: int, density: str) -> int:
    """How many distractor bindings a cell of this size carries at this level.

    `matched` is flat in size (that is the point); `natural` is the measured
    rate of the fixtures the A/B ran.
    """
    if density == "matched":
        return MATCHED_ENTITIES
    return max(1, int(size_tokens / NATURAL_TOKENS_PER_ENTITY + 0.5))


def neutral_paragraph(rng: random.Random) -> str:
    """Filler with NO entity binding: sentences only, no header line.

    The sentences come from the same pool as everything else (ordinary
    lowercase English nouns), so the padding is neutral in the sense that
    matters — it carries no identifier-shaped token and no coined name — while
    remaining the same register of prose. `build_cell` asserts the first half of
    that against `rlm.leakcheck`'s own pattern set rather than trusting it.

    2–4 sentences, where `make_sweep_fixtures._paragraph` uses 3–6: a paragraph
    is the unit the needle depth snaps to, and at 640 tokens a 190-token
    paragraph puts the nearest boundary to 50% at 64%. Shorter paragraphs make
    the depth grid fine enough that every cell in this grid holds its needle at
    the SAME depth, which is what keeps depth from becoming a fourth factor.
    """
    return " ".join(_sentence(rng) for _ in range(rng.randint(2, 4)))


def make_pool(rng: random.Random, n_words: int) -> str:
    """`n_words`-ish words of NEUTRAL, header-free filler."""
    paragraphs: list[str] = []
    words = 0
    while words < n_words:
        para = neutral_paragraph(rng)
        paragraphs.append(para)
        words += len(para.split())
    return "\n\n".join(paragraphs)


def _split_paragraphs(text: str) -> list[str]:
    return text.split("\n\n")


def place_entities(filler: str, headers: list[str]) -> str:
    """Prepend `len(headers)` entity headers to EVENLY SPACED filler paragraphs.

    Even spacing, not clustering: a fixed entity budget dropped in one place
    would make "density" covary with "where the distractors sit relative to the
    question", which is a fourth factor nobody asked for. Both density levels
    are placed by the same rule, so they differ in count and in nothing else.
    """
    paras = _split_paragraphs(filler)
    if not paras or not headers:
        return filler
    k = min(len(headers), len(paras))
    chosen = sorted({min(int((i + 0.5) * len(paras) / k), len(paras) - 1)
                     for i in range(k)})
    out = list(paras)
    for n, idx in enumerate(chosen):
        out[idx] = f"{headers[n]}\n{paras[idx]}"
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #


def cell_id(size_tokens: int, density: str) -> str:
    return f"s2d-{size_tokens}-{density}"


def _cell_rng(seed: int, size_tokens: int, density: str) -> random.Random:
    return random.Random(f"s2d:{seed}:{size_tokens}:{density}")


def build_cell(count: Callable[[str], int], *, size_tokens: int, density: str,
               seed: int = SEED, position: float = POSITION,
               matched_entities: int = MATCHED_ENTITIES) -> dict:
    """One chunk of ~`size_tokens` leaf tokens at one density, with its three
    questions. Refuses rather than degrades: every claim the runner will make
    about this fixture is asserted here.

    The FACTS are drawn from the cell RNG in the same order
    `make_sweep_fixtures.build_cell` draws them, and the needle and question
    strings are that module's verbatim — the point of the experiment is that
    only POSITION and DENSITY vary, so a reworded question would be a third
    variable.
    """
    if density not in DENSITIES:
        raise ValueError(f"unknown density {density!r}")
    rng = _cell_rng(seed, size_tokens, density)

    lit_org, lit_coined = _org(rng)
    lit_key = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    literal_needle = (
        f"[custody note] The archive key issued to {lit_org} is {lit_key}. "
        f"It was cut once, for that holder only, and no duplicate key was ever "
        f"issued under that name.")

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
        f"no other custodian is entered anywhere in the register.")

    abs_org, abs_coined = _org(rng, unlike=(lit_coined, par_coined, given, family))

    want_bindings = (matched_entities if density == "matched"
                     else n_bindings(size_tokens, density))
    headers = [entity_header(rng, i + 1) for i in range(want_bindings)]

    literal_tokens = count(f"\n\n{literal_needle}\n\n")
    needle_tokens = count(f"\n\n{literal_needle}\n\n{paraphrase_needle}\n\n")
    slack = max(8, int(size_tokens * SIZE_TOLERANCE))
    pool = make_pool(rng, int(size_tokens / 1.7 * 1.5) + 120)
    filler_target = max(1, size_tokens - needle_tokens)

    def assemble(budget: int) -> tuple[str, int]:
        filler = fit_to_tokens(pool, budget, count)
        filler = place_entities(filler, headers)
        lit_at = boundary_at_token_target(filler, int(position * size_tokens), count)
        par_at = boundary_at_token_target(
            filler,
            int(min(position + PARAPHRASE_DEPTH_OFFSET, 0.99) * size_tokens)
            - literal_tokens,
            count)
        if par_at <= lit_at:
            par_at = lit_at
        spliced = filler[:par_at] + paraphrase_needle + "\n\n" + filler[par_at:]
        spliced = spliced[:lit_at] + literal_needle + "\n\n" + spliced[lit_at:]
        return spliced, count(spliced)

    text, measured = assemble(filler_target)
    for _ in range(6):
        if abs(measured - size_tokens) <= slack:
            break
        # The header line costs tokens the budget did not know about, so the
        # correction is applied to the FILLER budget and re-assembled; this is
        # the same fixed-point loop `make_sweep_fixtures` runs, one iteration
        # longer because placing headers perturbs it a second time.
        filler_target = max(1, filler_target + (size_tokens - measured))
        text, measured = assemble(filler_target)
    else:
        if measured > size_tokens:
            text = fit_to_tokens(text, size_tokens, count)
            measured = count(text)

    # -- validity, asserted rather than promised --------------------------- #
    if abs(measured - size_tokens) > slack:
        raise AssertionError(
            f"{cell_id(size_tokens, density)}: built {measured} leaf tokens "
            f"against a {size_tokens} target (slack {slack}); the x-axis IS the "
            f"token count, so this is a broken fixture")
    if text.count(lit_key) != 1:
        raise AssertionError(f"literal key occurs {text.count(lit_key)} times, want 1")
    stray = set(UUID_RE.findall(text)) - {lit_key}
    if stray:
        raise AssertionError(f"UUID-shaped strings other than the planted key: "
                             f"{sorted(stray)[:3]}")
    if paraphrase_answer in text:
        raise AssertionError("the paraphrase answer occurs verbatim")
    if literal_needle not in text or paraphrase_needle not in text:
        raise AssertionError("a needle did not survive the final fit")
    for token in (abs_coined, abs_org):
        if token.lower() in text.lower():
            raise AssertionError(
                f"the ABSENT organisation {token!r} appears in the document; the "
                f"cell would measure a MISS, not a false positive")

    entity_codes = re.findall(r"ENT-\d{4,6}", text)
    if len(entity_codes) != want_bindings:
        raise AssertionError(
            f"{cell_id(size_tokens, density)}: {len(entity_codes)} entity "
            f"bindings, want exactly {want_bindings}. Density is a CONTROLLED "
            f"factor here — a cell that misses its own level separates nothing.")
    if len(set(entity_codes)) != len(entity_codes):
        raise AssertionError("duplicate ENT- codes: the bindings must be distinct")

    # The neutral padding must really be neutral. Checked against
    # `rlm.leakcheck`'s own pattern set (the instrument R13 is scored with), not
    # against a second, kinder regex written here.
    ids = set(identifier_tokens(text))
    expected_ids = {lit_key.lower()} | {c.lower() for c in entity_codes}
    unexpected = {i.lower() for i in ids} - expected_ids
    if unexpected:
        raise AssertionError(
            f"{cell_id(size_tokens, density)}: identifier-shaped tokens that are "
            f"neither the planted key nor a counted binding: {sorted(unexpected)[:5]}")

    lit_off = text.index(literal_needle)
    par_off = text.index(paraphrase_needle)

    def depth_of(offset: int) -> float:
        return round(count(text[:offset]) / measured, 4)

    lit_depth = depth_of(lit_off)
    if abs(lit_depth - position) > 0.10:
        raise AssertionError(
            f"{cell_id(size_tokens, density)}: the literal needle landed at "
            f"depth {lit_depth:.2f}, not {position:.2f}")

    return {
        "cell_id": cell_id(size_tokens, density),
        "size_tokens": size_tokens,
        "measured_tokens": measured,
        "density": density,
        "entity_bindings": len(entity_codes),
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
                   densities: tuple[str, ...] = DENSITIES, seed: int = SEED,
                   out_dir: Path = FIXTURES_DIR,
                   counter_name: str = "leaf:/tokenize",
                   matched_entities: int = MATCHED_ENTITIES) -> dict:
    """Every (size, density) cell for one seed, into its own directory.

    One seed per directory, mirroring `s2/fixtures-refusal-*`: the runner pools
    several directories and tags each cell with its manifest seed, which is how
    this experiment gets INDEPENDENT generated facts at one (size, density)
    rather than more trials against one fact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: dict[str, dict] = {}
    for size in sizes:
        for density in densities:
            cell = build_cell(count, size_tokens=size, density=density, seed=seed,
                              matched_entities=matched_entities)
            text = cell.pop("text")
            chunk_path = out_dir / f"{cell['cell_id']}.chunk.txt"
            chunk_path.write_text(text, encoding="utf-8", newline="\n")
            cell["chunk_path"] = str(chunk_path)
            cells[cell["cell_id"]] = cell

    manifest = {
        "generator": "s2/make_distance_fixtures.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "token_counter": counter_name,
        "seed": seed,
        "matched_entities": matched_entities,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cells": dict(sorted(cells.items())),
    }
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the S2 instruction-decay fixtures (distance x density)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="build one directory per seed: --out is suffixed "
                             "with -s<seed>")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--densities", nargs="+", default=list(DENSITIES))
    parser.add_argument("--matched-entities", type=int, default=MATCHED_ENTITIES)
    parser.add_argument("--leaf-port", type=int, default=8081)
    parser.add_argument("--out", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--offline", action="store_true",
                        help="build with the 4-chars-per-token proxy. Smoke-test "
                             "only: the manifest is stamped 'approx-offline' and "
                             "s2/run_distance.py refuses it.")
    args = parser.parse_args(argv)

    if args.offline:
        count, counter_name = approx_tokens, "approx-offline"
    else:
        count, counter_name = leaf_counter(args.leaf_port), "leaf:/tokenize"

    for seed in (args.seeds or [args.seed]):
        out = (args.out if not args.seeds
               else args.out.with_name(f"{args.out.name}-s{seed}"))
        manifest = write_fixtures(
            count, sizes=tuple(args.sizes), densities=tuple(args.densities),
            seed=seed, out_dir=out, counter_name=counter_name,
            matched_entities=args.matched_entities)
        for cid, cell in manifest["cells"].items():
            print(f"{cid}: target {cell['size_tokens']} -> measured "
                  f"{cell['measured_tokens']} leaf tokens, "
                  f"{cell['entity_bindings']} entity binding(s), "
                  f"needle depth {cell['needles']['literal']['token_depth']:.2f}, "
                  f"key={cell['questions']['literal']['expected']}")
        print(f"wrote {out / MANIFEST_NAME} ({len(manifest['cells'])} cells, "
              f"seed {seed}, counter={counter_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
