"""S1 fixture generator (spec §9 S1, task-17 brief).

Two tasks, both **non-benchmark and synthetic**, built deterministically from
one seed:

  * **needle** — >= 64,000 leaf tokens of generated filler carrying ONE random
    `entity -> UUID` pairing, placed in the final third of the document and
    verified (programmatically, at build time) to sit beyond the control arm's
    truncation point. A random UUID paired with a random coined name cannot
    exist in any training corpus, which is the whole point: a root that
    "answers" it without reading the document is confabulating, not recalling.
  * **paraphrase-needle** — the fact is stated ONLY in paraphrase, so the
    answer string occurs nowhere in the document and a regex for it cannot
    find it. That forces interpretation rather than a scan (spec §9 S1: it
    exercises the leaf path and pre-tests the R5 confabulation surface).

Nothing here imports `rlm`. The generator is measurement apparatus: it must be
importable, and reproducible, with no server, no config and no scaffold. The
ONE thing it cannot do offline is count leaf tokens -- so `write_fixtures`
takes an injected counter (the same idiom C2's chunker uses), and `main()`
supplies the real one by POSTing to the leaf server's `/tokenize`.

THE CONTROL-ARM RULE, STATED (spec §9 S1 (a)). `control_truncate(text,
n_tokens)` keeps the longest **whole-word prefix** of the document whose token
count is <= `n_tokens`, and drops the rest. Head-only, no sampling, no
summarisation: the point of the control is that the needle is genuinely
outside the window, not that it was cleverly summarised out of it. The counter
is injected; the default is an offline proxy (`approx_tokens`, ~4 chars per
token -- the same stated approximation `rlm.dispatcher.MockDispatcher` uses)
so the rule is deterministic and testable with no server, while the fixture
builder and the gate runner pass the REAL leaf tokenizer and therefore cut at
a real token budget. Both cut points are recorded in the task JSON, and the
build refuses unless the needle sits beyond BOTH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

S1_DIR = Path(__file__).resolve().parent
TASKS_DIR = S1_DIR / "tasks"

# Pinned generation parameters. `N_WORDS` was calibrated ONCE against the real
# leaf tokenizer (Qwen3.6-35B-A3B UD-Q4_K_M, b10375) at 1.72 tokens/word for
# this word pool: 37,600 words -> 64,735 leaf tokens (needle) / 64,607
# (paraphrase), over the >= 64,000 floor §9 states and under 2 x 32,768, so
# each corpus is exactly two chunks at the configured chunk size and one
# episode costs two chunk prefills rather than three. `write_fixtures`
# re-measures and REFUSES rather than trusting this number.
N_WORDS = 37_600
SEED = 1

# Where the needle sits, as a fraction of the document's words. Both are in
# the final third (spec §9 S1); the build asserts the char offset is past
# every control cut point regardless.
NEEDLE_AT = 0.78
PARAPHRASE_AT = 0.72

# The control arm's token budget (§9 S1 (a)): the root's 32,768-token window
# minus prompt overhead and the 90% C5 kill margin, so the request actually
# fits and the arm measures "the needle is not in the window", not "the
# request was rejected".
CONTROL_TOKENS = 28_000

_WORDS = (
    "harbor granite velvet copper meadow signal lantern orchard timber falcon "
    "marble cinder willow beacon quarry ribbon saddle thistle anchor bramble "
    "crystal dynamo ember foxglove gutter hollow ingot juniper kestrel lattice "
    "mortar nectar oakum pallet quiver rudder sextant tallow umber vellum "
    "wicket yarrow zephyr basalt cobalt drizzle estuary fathom girder haddock "
    "inlet jetty kelp lichen mallet nimbus outcrop plinth quartz rampart "
    "shingle transom updraft vector whetstone yardarm ballast capstan derrick "
    "escarpment flotsam gantry halyard isthmus keelson loam mainsail netting "
    "obelisk parapet quay reeve spandrel trellis undertow vestibule windlass "
    "consignment schedule inspection allocation clearance dispatch manifest "
    "inventory calibration provision remittance survey tender warrant yield "
    "abutment bollard culvert dowel eaves ferrule gasket hasp joist lintel"
).split()

_SYL_A = ("bran cor dal fen gar hol jar kel mor nor pav quor rath sev thar "
          "vex wyn zan brel crom dun eph gis").split()
_SYL_B = ("ta wold vane mir sk el dor nis vash ker lun mek ras ton vil gorn "
          "shel drix quen ryn").split()
_SYL_C = ("dis um ath ick ora eln ist urn ade ilm".split())

_LEDGER = ("Ledger Registry Bureau Consortium Syndicate Trust Repository "
           "Chapterhouse Exchange Assembly").split()

_SENTENCE_LEADS = (
    "The {a} crew logged the {b} without comment",
    "A revised {a} was filed against the {b} the same afternoon",
    "Nothing in the {a} record contradicts the {b} entry",
    "Storage for the {a} remains assigned to the {b} bay",
    "The {a} count was reconciled against the {b} sheet twice",
    "Transfer of the {a} to the {b} store was approved in principle",
    "No inspector signed the {a} line of the {b} form",
    "Weather held the {a} party at the {b} landing overnight",
)


# --------------------------------------------------------------------------- #
# deterministic generation
# --------------------------------------------------------------------------- #


def _coined_name(rng: random.Random, *, unlike: tuple[str, ...] = ()) -> str:
    """A coined, capitalised name. Not a real surname, not in any corpus.

    `unlike` rejects near-twins (a shared first or last syllable). Two names
    that differ by one letter -- `Nornisath` vs `Mornisath` -- would turn a
    reading test into a proof-reading test, and a failure caused by that is a
    fixture defect masquerading as an R1 finding.
    """
    banned_a = {n[:4].lower() for n in unlike}
    banned_z = {n[-4:].lower() for n in unlike}
    for _ in range(200):
        a, b, c = rng.choice(_SYL_A), rng.choice(_SYL_B), rng.choice(_SYL_C)
        name = (a + b + c).capitalize()
        if name[:4].lower() in banned_a or name[-4:].lower() in banned_z:
            continue
        if any(name == n for n in unlike):
            continue
        return name
    raise AssertionError("could not coin a name distinct from " + repr(unlike))


def _org(rng: random.Random, *, unlike: tuple[str, ...] = ()) -> tuple[str, str]:
    """`(display, coined)` -- e.g. `("the Vexdorist Ledger", "Vexdorist")`."""
    coined = _coined_name(rng, unlike=unlike)
    return f"the {coined} {rng.choice(_LEDGER)}", coined


def _sentence(rng: random.Random) -> str:
    lead = rng.choice(_SENTENCE_LEADS).format(a=rng.choice(_WORDS), b=rng.choice(_WORDS))
    tail = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 16)))
    return f"{lead}, {tail}."


def _paragraph(rng: random.Random, index: int) -> str:
    head = f"REF-{rng.randrange(10_000, 99_999)} / {rng.randrange(0x1000, 0xffff):04x}"
    body = " ".join(_sentence(rng) for _ in range(rng.randint(3, 7)))
    return f"[{index:05d}] {head}\n{body}"


def _document(rng: random.Random, n_words: int, insertions: list[tuple[float, str]]) -> tuple[str, dict[str, int]]:
    """Filler paragraphs with `insertions` placed at word-count fractions.

    Returns the document and each insertion's character offset, keyed by its
    text -- the build asserts on those offsets rather than on a promise.
    """
    marks = sorted((frac, text) for frac, text in insertions)
    paragraphs: list[str] = []
    offsets: dict[str, int] = {}
    words = 0
    index = 0
    pending = list(marks)
    while words < n_words:
        if pending and words >= pending[0][0] * n_words:
            _, text = pending.pop(0)
            offsets[text] = sum(len(p) + 2 for p in paragraphs)
            paragraphs.append(text)
            words += len(text.split())
            continue
        index += 1
        para = _paragraph(rng, index)
        paragraphs.append(para)
        words += len(para.split())
    for _, text in pending:  # insertions past the end would be a silent bug
        raise AssertionError(f"insertion {text[:40]!r} was never placed")
    return "\n\n".join(paragraphs), offsets


def build(seed: int = SEED, n_words: int = N_WORDS) -> dict[str, dict]:
    """Both S1 tasks, deterministically, with no server and no config.

    Everything a task needs except `tokenized_len` (which only the leaf's own
    tokenizer can supply) is decided here, so the same seed always produces
    the same document, the same needle and the same answer.
    """
    rng = random.Random(seed)

    # -- needle: one random entity -> UUID pairing --------------------------- #
    entity = f"{_coined_name(rng)} {rng.choice(_LEDGER)}"
    key = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    needle = (
        f"[custody note] The archive key issued to the {entity} is {key}. "
        f"It was issued once, to that holder only, and no second key was ever "
        f"cut under that name."
    )
    needle_doc, needle_offsets = _document(
        rng, n_words, [(NEEDLE_AT, needle)])

    # -- paraphrase-needle: the answer string occurs nowhere ------------------ #
    org, org_coined = _org(rng, unlike=(entity.split()[0],))
    given = _coined_name(rng, unlike=(org_coined, entity.split()[0]))
    family = _coined_name(rng, unlike=(org_coined, given, entity.split()[0]))
    passage = (
        f"[custodial register] {org} enters each custodian across two separate "
        f"lines and never prints a custodian's name in full.\n"
        f"GIVEN . . . . . . {given}\n"
        f"SEAL  . . . . . . intact\n"
        f"TERM  . . . . . . open\n"
        f"FAMILY  . . . . . {family}\n"
        f"Those two entries are the only custodian of record for this trust; no "
        f"other custodian is entered anywhere in the register."
    )
    para_doc, para_offsets = _document(rng, n_words, [(PARAPHRASE_AT, passage)])
    answer = f"{given} {family}"
    assert answer not in para_doc, "the paraphrase answer must not occur verbatim"

    return {
        "needle": {
            "task_id": "s1-needle",
            "category": "needle",
            "checker": "contains",
            "text": (
                f"Exactly one custody note in this corpus records the archive key "
                f"issued to the {entity}. What is that key? Reply with the key "
                f"itself and nothing else."),
            "answer": key,
            "entity": entity,
            "context_text": needle_doc,
            "needle_text": needle,
            "needle_char_offset": needle_offsets[needle],
        },
        "paraphrase": {
            "task_id": "s1-paraphrase",
            "category": "needle",
            "checker": "contains",
            "text": (
                f"{org[0].upper() + org[1:]} has exactly one custodian of record, but the "
                f"register never prints that custodian's name in full. What is the "
                f"custodian's full name? Reply with the given name followed by the "
                f"family name, and nothing else."),
            "answer": answer,
            "entity": org,
            "context_text": para_doc,
            "needle_text": passage,
            "needle_char_offset": para_offsets[passage],
        },
    }


# --------------------------------------------------------------------------- #
# the control-arm truncation rule
# --------------------------------------------------------------------------- #


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_fixtures(counter: Callable[[str], int], *, seed: int = SEED,
                    n_words: int = N_WORDS, out_dir: Path = TASKS_DIR,
                    min_tokens: int = 64_000) -> dict[str, dict]:
    """Build, VERIFY, and write both fixtures. Refuses rather than degrades.

    Three assertions are the fixture's whole claim to validity, and each is
    checked against the real tokenizer, not against an estimate:
      1. the needle document is >= `min_tokens` leaf tokens (§9 S1);
      2. the needle sits beyond the control cut -- under the real tokenizer
         AND under the offline proxy, since the test cuts with the proxy;
      3. the paraphrase answer occurs nowhere in the paraphrase document.
    """
    built = build(seed=seed, n_words=n_words)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator_sha = _sha256(Path(__file__).read_bytes())
    written: dict[str, dict] = {}

    for name, spec in built.items():
        text = spec["context_text"]
        tokenized_len = counter(text)
        if name == "needle" and tokenized_len < min_tokens:
            raise AssertionError(
                f"needle fixture is {tokenized_len} leaf tokens, below the "
                f"{min_tokens} floor spec §9 S1 states. Raise N_WORDS and "
                f"rebuild; never ship a short needle.")
        context_path = out_dir / f"{name}.context.txt"
        context_path.write_text(text, encoding="utf-8", newline="\n")

        real_cut = control_truncate(text, CONTROL_TOKENS, counter)
        proxy_cut = control_truncate(text, CONTROL_TOKENS)
        for label, cut in (("real", real_cut), ("proxy", proxy_cut)):
            if spec["answer"] in cut:
                raise AssertionError(
                    f"{name}: the answer survives the {label} control cut; the "
                    f"fixture is broken, not the scaffold")
        control_path = out_dir / f"{name}.control.txt"
        control_path.write_text(real_cut, encoding="utf-8", newline="\n")
        control_tokens = counter(real_cut)
        if control_tokens > CONTROL_TOKENS:
            raise AssertionError(
                f"{name}: control document is {control_tokens} tokens, over the "
                f"{CONTROL_TOKENS} budget")

        meta = {
            "task_id": spec["task_id"],
            "text": spec["text"],
            # `context` is the rlm.context spec the episode runner loads;
            # `context_path` is the same file as a plain path, for the fixture
            # tests and for anything that wants to read it without rlm.
            "context": {"path": str(context_path)},
            "context_path": str(context_path),
            "category": spec["category"],
            "checker": spec["checker"],
            "answer": spec["answer"],
            "entity": spec["entity"],
            "seed": seed,
            "n_words": n_words,
            "tokenized_len": tokenized_len,
            "context_chars": len(text),
            "context_sha256": _sha256(text.encode("utf-8")),
            "needle_text": spec["needle_text"],
            "needle_char_offset": spec["needle_char_offset"],
            "needle_fraction_of_chars": round(
                spec["needle_char_offset"] / len(text), 4),
            "control": {
                "rule": "keep the longest whole-word prefix with <= n_tokens "
                        "leaf tokens; drop the rest",
                "n_tokens": CONTROL_TOKENS,
                "path": str(control_path),
                "chars": len(real_cut),
                "tokens": control_tokens,
                "proxy_chars": len(proxy_cut),
                "answer_survives_cut": False,
            },
            "generator_sha256": generator_sha,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (out_dir / f"{name}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n")
        written[name] = meta
    return written


# --------------------------------------------------------------------------- #
# CLI: the real counter is the leaf server's own tokenizer
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the S1 fixtures (spec §9 S1)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--words", type=int, default=N_WORDS)
    parser.add_argument("--leaf-port", type=int, default=8081)
    parser.add_argument("--out", type=Path, default=TASKS_DIR)
    parser.add_argument("--calibrate", action="store_true",
                        help="report leaf tokens for the built documents and exit")
    args = parser.parse_args(argv)

    counter = leaf_counter(args.leaf_port)
    if args.calibrate:
        built = build(seed=args.seed, n_words=args.words)
        for name, spec in built.items():
            text = spec["context_text"]
            print(f"{name}: {args.words} words, {len(text)} chars, "
                  f"{counter(text)} leaf tokens, needle at char "
                  f"{spec['needle_char_offset']}")
        return 0

    written = write_fixtures(counter, seed=args.seed, n_words=args.words,
                              out_dir=args.out)
    for name, meta in written.items():
        print(f"{name}: {meta['tokenized_len']} leaf tokens, "
              f"{meta['context_chars']} chars, needle at char "
              f"{meta['needle_char_offset']} "
              f"({meta['needle_fraction_of_chars']:.0%}); control cut "
              f"{meta['control']['tokens']} tokens / "
              f"{meta['control']['chars']} chars; answer={meta['answer']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
