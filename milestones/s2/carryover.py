"""Does the PREVIOUS QUESTION on a slot corrupt the next answer?

Production asks several questions about one window on that window's slot
(§5 slot discipline; `R13-mitigations.md` §4.3 calls same-document reuse legal
and leak-free, and the whole warm re-query economics depend on it). But the
corrected arch ladder found that under one-slot-per-FIXTURE reuse the leaf
answered the ABSENT question with the key it had just emitted for the LITERAL
question on that slot, in 6 of 8 cases. Same-document reuse is leak-free; that
says nothing about whether the previous *question* carries.

THE CONFOUND THIS CONTROLS. The leaf's default misattribution target is often
`present[0]` -- the FIRST binding in the document -- so priming with a question
about the first binding makes an echo indistinguishable from baseline
misattribution. The priming question therefore asks about `present[2]`, the
LAST binding, which the model does not otherwise favour. An answer of uid_C is
then evidence of carry-over; an answer of uid_A is the model's baseline error.

  document plants  A = present[0], B = present[1], C = present[2], D absent
  priming question LITERAL about C      (answer uid_C -- the non-default target)
  measured         ABSENT about D       (correct answer: NONE)
                   LITERAL about B      (correct answer: uid_B)

ARMS, one never-used slot each, prompts byte-identical across arms:
  solo       the measured question alone, nothing before it
  after_cp1  priming question first, cache_prompt TRUE  -- what production does
  after_cp0  priming question first, cache_prompt FALSE -- same slot, no prefix
             reuse, so this separates "KV prefix reuse" from "slot state"

Run at PRODUCTION geometry, because the question is whether production is safe:
    llama-server --host 127.0.0.1 --port 8081 -m <leaf.gguf> \
      -c 327680 -np 128 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 \
      -lm none --no-kv-unified --cont-batching
    uv run --python 3.12 --no-project milestones/s2/carryover.py --label prod
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from arch_ladder import (BASE, PREFIX_PATH, REFUSAL, UUID_RE, apply_template,
                         bindings, build_chunk, ntok, post, strip_changelog)


def ask(prefix: str, chunk: str, question: str, id_slot: int,
        cache_prompt: bool, temp: float, n_predict: int) -> dict:
    """Layout is [system prefix][chunk][question], question LAST -- §4's prefix
    contract, so a re-query on the same slot extends the cached prefix."""
    msgs = [{"role": "system", "content": prefix},
            {"role": "user", "content": chunk + "\n\n" + question}]
    rendered = apply_template(msgs)
    r = post("/completion", {"prompt": rendered, "n_predict": n_predict,
                             "cache_prompt": cache_prompt, "temperature": temp,
                             "top_p": 0.9, "seed": 1, "id_slot": id_slot})
    t = r.get("timings", {})
    raw = r.get("content") or ""
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    if r.get("id_slot") != id_slot:
        raise SystemExit(f"asked for slot {id_slot}, served {r.get('id_slot')} "
                         f"— slot policy not honoured, run measures nothing.")
    return {"answer": raw.strip(), "prompt_n": t.get("prompt_n"),
            "cache_n": r.get("tokens_cached", t.get("cache_n")),
            "rendered_tokens": ntok(rendered)}


def classify(answer: str, expect: str | None, prev_uid: str | None,
             own: dict[str, str], foreign: dict[str, str]) -> tuple[str, str]:
    """ECHO_PREV is tested BEFORE misattribution: the primed uid is also an
    in-chunk uid, so the checks are not mutually exclusive and order decides
    what the number means."""
    a = answer.strip()
    if not a or not re.search(r"[A-Za-z0-9]", a):
        return "MALFORMED", ""
    found = [m.lower() for m in UUID_RE.findall(a)]
    if expect and expect.lower() in a.lower():
        return "CORRECT", ""
    if prev_uid and prev_uid.lower() in found:
        return "ECHO_PREV", "repeated the previous answer on this slot"
    if REFUSAL.search(a) and not found:
        return "REFUSED", ""
    for u in found:
        if u in own:
            return "MISATTRIBUTED", f"own chunk, bound to {own[u]}"
        if u in foreign:
            return "LEAKED", f"from {foreign[u]}"
        return "FABRICATED", "matches nothing planted"
    return "ANSWERED_ANYWAY", "no uuid in answer"


ARMS = ("solo", "after_cp1", "after_cp0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", nargs="*", type=int, default=[640, 1024])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=160)
    ap.add_argument("--slot-base", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    prefix = strip_changelog(PREFIX_PATH.read_text(encoding="utf-8"))
    needed = len(a.sizes) * a.trials * len(ARMS) * 2
    with urllib.request.urlopen(BASE + "/slots", timeout=30) as r:
        n_slots = len(json.loads(r.read().decode()))
    if a.slot_base + needed > n_slots:
        raise SystemExit(f"need {needed} slots from base {a.slot_base}, server "
                         f"has {n_slots}. Relaunch with -np {a.slot_base+needed} "
                         f"or larger — out-of-range id_slot is silently "
                         f"reassigned to a used slot.")
    print(f"prefix {ntok(prefix)} tok · {needed} slots of {n_slots} · "
          f"temp {a.temp} · label {a.label}")

    foreign: dict[str, str] = {}
    for size in a.sizes:
        for trial in range(a.trials):
            for ent, uid in bindings(size, trial)[0]:
                foreign.setdefault(uid.lower(), f"{size}/t{trial} ({ent})")

    rows, slot = [], a.slot_base
    for size in a.sizes:
        for trial in range(a.trials):
            chunk, present, absent_ent = build_chunk(size, trial, size)
            own = {u.lower(): e for e, u in present}
            (ent_a, uid_a), (ent_b, uid_b), (ent_c, uid_c) = present
            # foreign = everything planted anywhere EXCEPT this chunk
            other = {k: v for k, v in foreign.items() if k not in own}

            for mq, question, expect in (
                ("ABSENT", f"What is the custody key for the {absent_ent}?", None),
                ("LITERAL_B", f"What is the custody key for the {ent_b}?", uid_b),
            ):
                for arm in ARMS:
                    s = slot
                    slot += 1
                    prev_uid = None
                    prime = None
                    if arm != "solo":
                        cp = arm == "after_cp1"
                        pr = ask(prefix, chunk,
                                 f"What is the custody key for the {ent_c}?",
                                 s, cp, a.temp, a.n_predict)
                        prev_uid = uid_c
                        prime = {"ok": uid_c.lower() in pr["answer"].lower(),
                                 "cache_n": pr["cache_n"],
                                 "answer": pr["answer"][:80]}
                    r = ask(prefix, chunk, question, s,
                            arm == "after_cp1", a.temp, a.n_predict)
                    cls, detail = classify(r["answer"], expect, prev_uid,
                                           own, other)
                    rows.append({"label": a.label, "size": size, "trial": trial,
                                 "measured": mq, "arm": arm, "cls": cls,
                                 "detail": detail, "slot": s,
                                 "prime": prime, "cache_n": r["cache_n"],
                                 "prompt_n": r["prompt_n"],
                                 "rendered_tokens": r["rendered_tokens"],
                                 "answer": r["answer"][:200]})
                    flag = "  <== ECHO" if cls == "ECHO_PREV" else ""
                    print(f"  {size:>5} t{trial} {mq:<9} {arm:<9} slot{s:<4} "
                          f"-> {cls}{flag}")

    out = Path(a.out or f"milestones/s2/results/carryover_{a.label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== {a.label} · previous-question carry-over ===")
    for mq in ("ABSENT", "LITERAL_B"):
        good = "REFUSED" if mq == "ABSENT" else "CORRECT"
        print(f"\n  measured: {mq}   (wanted: {good})")
        print(f"  {'size':>6} {'arm':<10} {good:>9} {'ECHO_PREV':>10} "
              f"{'MISATTR':>8} {'other':>6}  {'median cache_n':>14}")
        for size in a.sizes:
            for arm in ARMS:
                sel = [r for r in rows if r["size"] == size
                       and r["measured"] == mq and r["arm"] == arm]
                if not sel:
                    continue
                cn = sorted(r["cache_n"] or 0 for r in sel)
                med = cn[len(cn) // 2] if cn else 0
                ok = sum(r["cls"] == good for r in sel)
                echo = sum(r["cls"] == "ECHO_PREV" for r in sel)
                mis = sum(r["cls"] == "MISATTRIBUTED" for r in sel)
                oth = len(sel) - ok - echo - mis
                print(f"  {size:>6} {arm:<10} {ok:>6}/{len(sel):<2} {echo:>10} "
                      f"{mis:>8} {oth:>6}  {med:>14}")
    bad_prime = [r for r in rows if r["prime"] and not r["prime"]["ok"]]
    print(f"\n  priming question answered correctly: "
          f"{len([r for r in rows if r['prime']]) - len(bad_prime)}"
          f"/{len([r for r in rows if r['prime']])}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
