"""The benchmark's OWN coined-name space, disjoint from every fixture's.

§8 requires S2's gates to run on "dedicated non-benchmark fixtures so S2 cannot
overfit the benchmark it is authoring". That is a statement about corpora, and
it is only true if the benchmark's entity names cannot collide with a fixture's:
a benchmark answer that also appears in an s2 fixture would let a leak, a
memorisation, or a stale slot score as a pass.

The existing generators already establish the pattern -- three pairwise-disjoint
syllable triples, one per corpus family -- so this is the fourth:

    milestones/s1/make_fixtures.py        _SYL_A 23 x _SYL_B 20 x _SYL_C 10
    milestones/s2/make_sweep_fixtures.py  _SYL_A 20 x _SYL_B 17 x _SYL_C 10
    milestones/s2/make_distance_fixtures  imports the s2 sweep pool
    bench/vocab.py             THIS ONE

**Do not import a name pool from s1 or s2 here.** Importing the fit and
placement helpers is right and is done elsewhere in this package; importing
`_org`/`_coined` would put the benchmark inside the fixture namespace, which is
the precise contamination §8 forbids.

The structural check lives in `bench/build.py` as `assert_name_space_disjoint`,
and it runs on the BUILD path -- not at import of this module, where it used to
sit. Two reasons. Its own purpose is that a collision must not be discovered
late, and the late moment is a build; firing on every import by every consumer
except at the build was the wrong placement. And an AssertionError raised inside
a leaf library is something a maintainer can be tempted to route around, whereas
a build that refuses to emit a corpus is not. Moving it also means THIS module
imports nothing from `milestones/`.
"""
from __future__ import annotations

import random
import re

__all__ = ["SYL_A", "SYL_B", "SYL_C", "BODIES", "coined_name", "organisation",
           "harvest_names"]

# Chosen so that no syllable is shared with either existing pool. Checked, not
# asserted by eye: bench/build.py's assert_name_space_disjoint re-derives it
# against the real fixture modules on every build.
# `eph`, `lorn` and `ryn` were in the first draft of this list and all three
# collide with s1's pool; `keld` in SYL_B collided too. The assertion below
# found them on its first run, which is the entire reason it runs at import
# rather than living in a test somebody remembers to write.
SYL_A = ("azh brill cyne drusk emberly frayn glin hesp irm joss kyth lissom "
         "myrr neth oph prask quill rooke sturn tavv umb vosk wren xanth "
         "yeld zurn").split()
SYL_B = ("ambry blythe crowe delve emmer fallow girt hallow inch kestrel "
         "lammer marrow nether orrery pell quarrel ryesome tarn upwell verge"
         ).split()
SYL_C = ("hurst leigh mere scar tofts wardine yatt cleave dorne griff").split()

# Organisation bodies: distinct from s1's and s2's, and deliberately plain --
# the corpus is meant to be dull, so that a model cannot answer from genre.
BODIES = ("Ledgerhouse", "Assay Office", "Tithe Barn", "Muniment Room",
          "Chancery Annexe", "Bonding Yard", "Escrow Hall", "Warrant Office")

_ID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\bENT-\d{4,6}\b", re.I)


def coined_name(rng: random.Random) -> str:
    return (rng.choice(SYL_A) + rng.choice(SYL_B) + rng.choice(SYL_C)).capitalize()


def organisation(rng: random.Random) -> str:
    return f"{coined_name(rng)} {rng.choice(BODIES)}"


def harvest_names(text: str) -> set[str]:
    """Every identifier-shaped token in a corpus: UUIDs and ENT- codes.

    Used for the second, literal half of the disjointness check -- the
    structural syllable check cannot see a UUID that happens to collide, and a
    collision there would be far more damaging than a shared syllable."""
    return {m.lower() for m in _ID_RE.findall(text)}
