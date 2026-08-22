"""Does accuracy decay as a slot is asked MANY questions in a row?

`milestones/s2/CARRYOVER.md` measured depth TWO — one priming question, then the measured
one — and found no effect at the shipped 640-token window (40/40 byte-identical
answers). It explicitly left depth open: production asks a window several
questions on that window's slot, and a per-answer residue that is invisible at
question two could still be obvious at question ten.

    deep     all N questions down ONE never-used slot, in order
    shallow  the SAME N questions, each on its OWN never-used slot
             (position-matched control: question i is identical in both arms,
              so any difference at position i is the slot's history)

Degradation shows up as accuracy falling with POSITION in the deep arm while the
shallow arm stays flat. Comparing deep-position-9 against deep-position-0 alone
would confound depth with "later questions happen to be harder".

WHY THIS FILE CARRIES ITS OWN ENTITY POOL. `arch_ladder.bindings()` draws from
an 8-stem pool, and every result in `milestones/s2/results/arch_ladder_*.jsonl` is keyed to
that exact pool via its seed. Ten distinct questions need more stems; editing the
shared pool would silently change what those seeds generate and invalidate the
comparison runs. So this probe generates its own fixtures and leaves that pool
untouched.

Run at production geometry (needs ~90 slots for 8 fixtures at depth 10):
    llama-server ... -c 327680 -np 128 ...
    uv run --python 3.12 --no-project milestones/s2/carryover_depth.py --label prod
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

from arch_ladder import (BASE, PREFIX_PATH, REFUSAL, UUID_RE, apply_template,
                         filler, ntok, post, strip_changelog)

# This probe's OWN pool -- see the module docstring. Never import STEMS here.
POOL = ["Vandreholt", "Cressmarrow", "Ildenbrooke", "Skarnwelter",
        "Pellingfore", "Author­ncliff", "Ravensmuir", "Thornequay",
        "Gilderwatch", "Mossbannon", "Feldencrag", "Wraithmoor"]


def make_fixture(seed: int, target_tokens: int, n_present: int, n_absent: int):
    """Return (chunk, present[(ent,uid)], absent[ent]). Facts sit at the END so
    needle distance is minimal and cannot explain a failure."""
    rng = random.Random(seed)
    pool = rng.sample(POOL, n_present + n_absent)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:n_present]]
    absent = [f"{s} Trust" for s in pool[n_present:]]
    fact = "".join(f"\nRECORD: The custody key for the {e} is {u}.\n"
                   for e, u in present)
    lo, hi, body = 1, target_tokens * 3, ""
    for _ in range(24):
        mid = (lo + hi) // 2
        body = filler(rng, mid)
        got = ntok("DOCUMENT:\n" + body + fact)
        if got < target_tokens:
            lo = mid + 1
        elif got > target_tokens:
            hi = mid - 1
        else:
            break
        if lo > hi:
            break
    return "DOCUMENT:\n" + body + fact, present, absent


def ask(prefix: str, chunk: str, question: str, id_slot: int,
        cache_prompt: bool, temp: float, n_predict: int) -> dict:
    msgs = [{"role": "system", "content": prefix},
            {"role": "user", "content": chunk + "\n\n" + question}]
    rendered = apply_template(msgs)
    r = post("/completion", {"prompt": rendered, "n_predict": n_predict,
                             "cache_prompt": cache_prompt, "temperature": temp,
                             "top_p": 0.9, "seed": 1, "id_slot": id_slot})
    if r.get("id_slot") != id_slot:
        raise SystemExit(f"asked slot {id_slot}, served {r.get('id_slot')} — "
                         f"slot policy not honoured, run measures nothing.")
    raw = r.get("content") or ""
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    return {"answer": raw.strip(),
            "prompt_n": (r.get("timings") or {}).get("prompt_n"),
            "rendered_tokens": ntok(rendered)}


def classify(answer: str, expect: str | None, own: dict[str, str]) -> str:
    a = answer.strip()
    if not a or not re.search(r"[A-Za-z0-9]", a):
        return "MALFORMED"
    found = [m.lower() for m in UUID_RE.findall(a)]
    if expect:
        return "CORRECT" if expect.lower() in a.lower() else (
            "WRONG_KEY" if found else "NO_ANSWER")
    if found:
        return "FALSE_POSITIVE"
    return "REFUSED" if REFUSAL.search(a) else "ANSWERED_ANYWAY"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--fixtures", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=160)
    ap.add_argument("--slot-base", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    n_present = (a.depth + 1) // 2
    n_absent = a.depth // 2
    if n_present + n_absent > len(POOL):
        raise SystemExit(f"depth {a.depth} needs {n_present+n_absent} stems, "
                         f"pool has {len(POOL)}")

    prefix = strip_changelog(PREFIX_PATH.read_text(encoding="utf-8"))
    needed = a.fixtures * (1 + a.depth)          # 1 deep slot + depth shallow
    with urllib.request.urlopen(BASE + "/slots", timeout=30) as r:
        n_slots = len(json.loads(r.read().decode()))
    if a.slot_base + needed > n_slots:
        raise SystemExit(f"need {needed} slots from base {a.slot_base}, server "
                         f"has {n_slots}. Relaunch with -np "
                         f"{a.slot_base + needed} or larger — an out-of-range "
                         f"id_slot is silently reassigned to a used slot.")
    print(f"prefix {ntok(prefix)} tok · depth {a.depth} · {a.fixtures} fixtures "
          f"· {needed} slots of {n_slots} · size {a.size} · temp {a.temp}")

    rows, slot = [], a.slot_base
    for fx in range(a.fixtures):
        chunk, present, absent = make_fixture(9000 + fx, a.size,
                                              n_present, n_absent)
        own = {u.lower(): e for e, u in present}
        # alternate LITERAL / ABSENT so both question kinds appear at early and
        # late positions; a run that put all ABSENTs last would confound
        # position with question kind.
        seq = []
        for i in range(a.depth):
            if i % 2 == 0:
                e, u = present[i // 2]
                seq.append(("LITERAL", f"What is the custody key for the {e}?", u))
            else:
                e = absent[i // 2]
                seq.append(("ABSENT", f"What is the custody key for the {e}?", None))

        deep_slot = slot
        slot += 1
        for pos, (kind, q, expect) in enumerate(seq):
            r = ask(prefix, chunk, q, deep_slot, True, a.temp, a.n_predict)
            rows.append({"label": a.label, "fixture": fx, "arm": "deep",
                         "pos": pos, "kind": kind,
                         "cls": classify(r["answer"], expect, own),
                         "slot": deep_slot, "prompt_n": r["prompt_n"],
                         "rendered_tokens": r["rendered_tokens"],
                         "answer": r["answer"][:160]})
        for pos, (kind, q, expect) in enumerate(seq):
            s = slot
            slot += 1
            r = ask(prefix, chunk, q, s, True, a.temp, a.n_predict)
            rows.append({"label": a.label, "fixture": fx, "arm": "shallow",
                         "pos": pos, "kind": kind,
                         "cls": classify(r["answer"], expect, own),
                         "slot": s, "prompt_n": r["prompt_n"],
                         "rendered_tokens": r["rendered_tokens"],
                         "answer": r["answer"][:160]})
        d = sum(r["cls"] in ("CORRECT", "REFUSED")
                for r in rows if r["fixture"] == fx and r["arm"] == "deep")
        s_ = sum(r["cls"] in ("CORRECT", "REFUSED")
                 for r in rows if r["fixture"] == fx and r["arm"] == "shallow")
        print(f"  fixture {fx}: deep {d}/{a.depth}   shallow {s_}/{a.depth}")

    out = Path(a.out or f"milestones/s2/results/carryover_depth_{a.label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== {a.label} · accuracy by POSITION on the slot "
          f"(size {a.size}, n={a.fixtures} per cell) ===")
    print(f"  {'pos':>3} {'kind':<8} {'deep ok':>9} {'shallow ok':>11}  "
          f"{'deep prompt_n':>14}")
    for pos in range(a.depth):
        d = [r for r in rows if r["pos"] == pos and r["arm"] == "deep"]
        s_ = [r for r in rows if r["pos"] == pos and r["arm"] == "shallow"]
        ok = lambda rs: sum(r["cls"] in ("CORRECT", "REFUSED") for r in rs)
        pn = sorted(r["prompt_n"] or 0 for r in d)
        print(f"  {pos:>3} {d[0]['kind']:<8} {ok(d):>6}/{len(d):<2} "
              f"{ok(s_):>8}/{len(s_):<2}  {pn[len(pn)//2]:>14}")
    for arm in ("deep", "shallow"):
        sel = [r for r in rows if r["arm"] == arm]
        n_ok = sum(r["cls"] in ("CORRECT", "REFUSED") for r in sel)
        print(f"  TOTAL {arm:<8} {n_ok}/{len(sel)}")
    ident = sum(1 for pos in range(a.depth) for fx in range(a.fixtures)
                if next(r for r in rows if r["arm"] == "deep" and r["pos"] == pos
                        and r["fixture"] == fx)["answer"]
                == next(r for r in rows if r["arm"] == "shallow"
                        and r["pos"] == pos and r["fixture"] == fx)["answer"])
    print(f"  byte-identical deep vs shallow: {ident}/{a.depth * a.fixtures}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
