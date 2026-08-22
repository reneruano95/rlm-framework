"""Is the ~1,000-token horizon a property of THIS model, or of the harness?

The leaf (Qwen3.6-35B-A3B) is hybrid: ~25% of layers carry true KV attention,
the rest compress history into a fixed-size, context-INDEPENDENT recurrent state
(measured 62.8 MiB/slot regardless of context). That is the shape that produces
a hard horizon, and §4 measured facts AND instructions failing at one shared
threshold. If the horizon belongs to the architecture, a verified full-attention
model of similar vintage should NOT show the same cliff at the same distances.

Run this against each model in turn (one server at a time, same script):
    uv run --python 3.12 --no-project milestones/s2/arch_ladder.py --label qwen --policy virgin
    uv run --python 3.12 --no-project milestones/s2/arch_ladder.py --label qwen --policy shared

CONFOUND THIS CONTROLS: token counts are model-specific, so each size target is
hit against the SERVER UNDER TEST's own /tokenize. "640 tokens" therefore means
640 of that model's tokens in both arms — which is what a token-distance
hypothesis requires. Char counts are recorded so the difference is auditable.

CONFOUND THE FIRST RUN DID NOT CONTROL, which invalidated it (2026-08-14):
this probe is an ASCENDING ladder 640 -> 1024 -> 2048, and R13-mitigations.md
§4.4 measures document GROWTH down a reused slot as the exact trigger for R13
cross-request leakage ("six windows ascending 159 -> 1,934 tokens: 4/18 leaked").
The first run left slot assignment to the server, so the 2,048 rows were served
by slots that had already held the 1,024 documents — and every "refusal failure"
there was the CORRECT key for the asked entity, lifted from a PRIOR fixture.
That was a leak, not a model property.

Slot policy is therefore explicit, and is itself an arm:
    --policy virgin   one never-reused slot per fixture, both questions on it
                      (same document -> legal warm reuse, measured 0/72 clean)
    --policy shared   every request pinned to one slot (the R13 positive control)
Run both against one server: same weights, same moment, byte-identical prompts,
slot history the only variable. If the 2,048 collapse appears only under
`shared`, it was never about attention architecture.

Server must therefore have enough slots and enough room per slot:
    llama-server --host 127.0.0.1 --port 8081 -m <model.gguf> \
      -c 65536 -np 16 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 \
      -lm none --no-kv-unified --cont-batching
(4,096 tokens/slot x 16 virgin slots; the largest prompt here is ~2,420 + 160.)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8081"
PREFIX_PATH = Path(__file__).resolve().parents[2] / "prompts" / "leaf-prefix.v1.md"

# Entity/UUID pool: unguessable by construction. Same generator for both arms.
STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]


def post(path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ntok(text: str) -> int:
    return len(post("/tokenize", {"content": text})["tokens"])


def strip_changelog(md: str) -> str:
    return re.sub(r"^<!--.*?-->\s*", "", md, flags=re.S)


def filler(rng: random.Random, n_words: int) -> str:
    """Entity-free padding: no ids, no digits, nothing an ABSENT question
    could accidentally match."""
    words = ["ledger", "harbor", "granite", "velvet", "copper", "meadow",
             "lantern", "orchard", "timber", "falcon", "marble", "willow",
             "beacon", "quarry", "ribbon", "saddle", "thistle", "bramble"]
    out = []
    for i in range(n_words):
        out.append(rng.choice(words))
        if i % 14 == 13:
            out.append(".\n")
    return " ".join(out)


def bindings(size: int, trial: int):
    """The entity->uuid bindings for one fixture, drawn before any server call.

    Returns (present, absent_entity, rng) and hands back the LIVE rng so the
    caller can keep drawing filler from the same stream. The offline leak oracle
    calls this too and ignores the rng -- one generator, no duplicated draw
    sequence to drift out of sync.
    """
    rng = random.Random(1000 * size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:3]]
    return present, f"{pool[3]} Trust", rng


def build_chunk(size: int, trial: int, target_tokens: int):
    """Return (chunk, present_bindings, absent_entity).

    The planted facts sit at the END of the chunk so that needle-to-question
    distance is minimal and CANNOT explain a failure -- this isolates the
    instruction-distance axis, which is what differs between the arms.

    THREE bindings, matching the validated fixture design. A single lone entity
    plus an obviously-foreign ABSENT question makes refusal trivial and the
    phenomenon under study does not appear at all -- measured on this box.
    """
    present, absent_ent, rng = bindings(size, trial)
    fact = "".join(f"\nRECORD: The custody key for the {e} is {u}.\n"
                   for e, u in present)

    lo, hi = 1, target_tokens * 3
    body = ""
    for _ in range(24):  # binary search on filler length
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
    return "DOCUMENT:\n" + body + fact, present, absent_ent


_TEMPLATE_KWARGS_OK: bool | None = None


def apply_template(msgs: list[dict]) -> str:
    """Production passes enable_thinking=false (spec §4/D15): with thinking on,
    the model spends its whole budget on a preamble and never emits an answer.
    Not every template accepts the kwarg, so fall back for the control model --
    but RECORD which path was taken, because a silent difference in thinking
    mode between the two model arms would be a confound in itself."""
    global _TEMPLATE_KWARGS_OK
    try:
        p = post("/apply-template", {
            "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        if _TEMPLATE_KWARGS_OK is None:
            _TEMPLATE_KWARGS_OK = True
        return p
    except Exception:
        if _TEMPLATE_KWARGS_OK is None:
            _TEMPLATE_KWARGS_OK = False
        return post("/apply-template", {"messages": msgs})["prompt"]


def ask(prefix: str, chunk: str, question: str, seed: int,
        id_slot: int | None, temp: float) -> dict:
    msgs = [{"role": "system", "content": prefix},
            {"role": "user", "content": chunk + "\n\n" + question}]
    rendered = apply_template(msgs)
    body = {"prompt": rendered, "n_predict": 160, "cache_prompt": False,
            "temperature": temp, "top_p": 0.9, "seed": seed}
    if id_slot is not None:          # policy=auto omits it entirely, so the
        body["id_slot"] = id_slot    # server picks by LRU / LCP similarity
    r = post("/completion", body)
    t = r.get("timings", {})
    raw = (r.get("content") or "")
    if "</think>" in raw:            # keep the tail after the LAST close tag
        raw = raw.rsplit("</think>", 1)[1]
    return {"answer": raw.strip(), "prompt_n": t.get("prompt_n"),
            "id_slot_served": r.get("id_slot"), "rendered_tokens": ntok(rendered)}


REFUSAL = re.compile(
    r"\b(none|not (present|found|stated|mentioned|contain)|no (record|key|mention)"
    r"|does not (contain|mention|state|include|list|specify)|cannot find"
    r"|unable to find|not (in|available)|n/?a)\b", re.I)

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def classify(answer: str, expect_uid: str | None, own_uids: dict[str, str],
             all_uids: dict[str, str]) -> tuple[str, str]:
    """Return (class, detail).

    The distinction that matters: a wrong answer quoting a uuid from THIS chunk
    is the model misattributing (a model behaviour); a wrong answer quoting a
    uuid planted in a DIFFERENT fixture is cross-request leakage (R13, a serving
    defect). The first run conflated the two and read a leak as a model property.
    """
    a = answer.strip()
    if not a or not re.search(r"[A-Za-z0-9]", a):
        return "MALFORMED", ""
    if expect_uid and expect_uid.lower() in a.lower():
        return "CORRECT", ""
    if REFUSAL.search(a):
        return "REFUSED", ""
    for m in UUID_RE.findall(a):
        u = m.lower()
        if u in own_uids:
            return "MISATTRIBUTED", f"own chunk, bound to {own_uids[u]}"
        if u in all_uids:
            return "LEAKED", f"from {all_uids[u]}"
        return "FABRICATED", "uuid matches nothing planted"
    return "ANSWERED_ANYWAY", "no uuid in answer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", nargs="*", type=int, default=[640, 1024, 2048])
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--policy",
                    choices=["isolated", "virgin", "shared", "auto"],
                    default="isolated",
                    help="isolated: ONE QUESTION per never-used slot -- nothing "
                         "whatsoever precedes the request being measured. This "
                         "is the only arm that measures the model rather than "
                         "the serving state, and it is the default for that "
                         "reason. virgin: one never-reused slot per FIXTURE, so "
                         "the second question follows the first on that slot "
                         "(R13-mitigations calls this legal same-document reuse; "
                         "measured here, the model repeats its previous answer). "
                         "shared: everything pinned to one slot. auto: send no "
                         "id_slot at all, so the server picks by LRU / LCP "
                         "similarity -- what the invalidated 2026-08-14 run did.")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="GREEDY BY DEFAULT, deliberately. The first runs used "
                         "0.3, and two server geometries then disagreed 4/4 vs "
                         "1/4 on identical prompts and seeds -- at n=4 sampling "
                         "noise can swamp the effect being measured.")
    ap.add_argument("--slot-base", type=int, default=1)
    ap.add_argument("--shared-slot", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    prefix = strip_changelog(PREFIX_PATH.read_text(encoding="utf-8"))
    print(f"prefix tokens: {ntok(prefix)}   label: {a.label}   policy: {a.policy}")

    # GUARD (added after this bit me): llama-server does NOT reject an id_slot
    # above its slot count -- it silently serves the request on some other slot.
    # An `isolated` run that asked for 25 slots on a 16-slot server therefore
    # got requests 16..24 served by slots 0..8, which had already held the
    # 640-token documents, and leaked. Every never-reuse guarantee in this
    # project rests on id_slot being honoured, so check it up front and again
    # per request rather than trusting it.
    n_requests = len(a.sizes) * a.trials * 2
    needed = {"isolated": n_requests, "virgin": len(a.sizes) * a.trials,
              "shared": 1, "auto": 0}[a.policy]
    if needed:
        with urllib.request.urlopen(BASE + "/slots", timeout=30) as r:
            n_slots = len(json.loads(r.read().decode()))
        if a.slot_base + needed > n_slots:
            raise SystemExit(
                f"policy={a.policy} needs slots {a.slot_base}.."
                f"{a.slot_base + needed - 1} but the server has {n_slots}. "
                f"Relaunch with -np {a.slot_base + needed} or larger. Refusing "
                f"to run: out-of-range slots are silently reassigned to used "
                f"ones, which manufactures exactly the leak this arm controls for.")
        print(f"slot check: need {needed} from base {a.slot_base}, "
              f"server has {n_slots} — ok")

    # Every uuid this run plants anywhere, so a leak can be traced to its donor.
    all_uids: dict[str, str] = {}
    for size in a.sizes:
        for trial in range(a.trials):
            for ent, uid in bindings(size, trial)[0]:
                all_uids.setdefault(uid.lower(), f"{size}/t{trial} ({ent})")

    rows = []
    fixture_i = 0
    for size in a.sizes:
        for trial in range(a.trials):
            chunk, present, absent_ent = build_chunk(size, trial, size)
            own_uids = {u.lower(): e for e, u in present}
            ent, uid = present[0]
            fixture_i += 1
            for qtype, q, expect in (
                ("LITERAL", f"What is the custody key for the {ent}?", uid),
                ("ABSENT", f"What is the custody key for the {absent_ent}?", None),
            ):
                slot = {"isolated": a.slot_base + len(rows),
                        "virgin": a.slot_base + fixture_i - 1,
                        "shared": a.shared_slot,
                        "auto": None}[a.policy]
                r = ask(prefix, chunk, q, seed=trial + 1, id_slot=slot,
                        temp=a.temp)
                if slot is not None and r["id_slot_served"] != slot:
                    raise SystemExit(
                        f"asked for slot {slot}, server served "
                        f"{r['id_slot_served']}. The slot policy is not being "
                        f"honoured, so this run measures nothing.")
                cls, detail = classify(r["answer"], expect, own_uids, all_uids)
                rows.append({"label": a.label, "policy": a.policy, "size": size,
                             "trial": trial, "qtype": qtype, "cls": cls,
                             "detail": detail, "uid": uid, "slot": slot,
                             "slot_served": r["id_slot_served"],
                             "chunk_tokens": ntok(chunk),
                             "rendered_tokens": r["rendered_tokens"],
                             "answer": r["answer"][:200]})
                mark = "  <== LEAK" if cls == "LEAKED" else ""
                shown = "auto" if slot is None else str(slot)
                print(f"  {size:>5} t{trial} slot{shown:<4}"
                      f"(served {r['id_slot_served']}) {qtype:<8} -> "
                      f"{cls}{(' (' + detail + ')') if detail else ''}{mark}")

    out = Path(a.out or f"milestones/s2/results/arch_ladder_{a.label}_{a.policy}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== {a.label} / policy={a.policy} "
          f"(enable_thinking kwarg accepted: {_TEMPLATE_KWARGS_OK}) ===")
    print(f"{'size':>6} {'chunk_tok':>10} {'LITERAL ok':>12} {'ABSENT refused':>15} "
          f"{'LEAKED':>8} {'misattrib':>10}")
    for size in a.sizes:
        lit = [r for r in rows if r["size"] == size and r["qtype"] == "LITERAL"]
        ab = [r for r in rows if r["size"] == size and r["qtype"] == "ABSENT"]
        ct = lit[0]["chunk_tokens"] if lit else 0
        print(f"{size:>6} {ct:>10} "
              f"{sum(r['cls'] == 'CORRECT' for r in lit):>5}/{len(lit):<6} "
              f"{sum(r['cls'] == 'REFUSED' for r in ab):>7}/{len(ab):<7} "
              f"{sum(r['cls'] == 'LEAKED' for r in rows if r['size'] == size):>8} "
              f"{sum(r['cls'] == 'MISATTRIBUTED' for r in rows if r['size'] == size):>10}")
    total_leaks = sum(r["cls"] == "LEAKED" for r in rows)
    print(f"\nTOTAL LEAKED: {total_leaks}/{len(rows)}   wrote {out}")


if __name__ == "__main__":
    main()
