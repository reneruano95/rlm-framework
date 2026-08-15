"""Is the ~1,000-token horizon a property of THIS model, or of the harness?

The leaf (Qwen3.6-35B-A3B) is hybrid: ~25% of layers carry true KV attention,
the rest compress history into a fixed-size, context-INDEPENDENT recurrent state
(measured 62.8 MiB/slot regardless of context). That is the shape that produces
a hard horizon, and §4 measured facts AND instructions failing at one shared
threshold. If the horizon belongs to the architecture, a verified full-attention
model of similar vintage should NOT show the same cliff at the same distances.

Run this against each model in turn (one server at a time, same script):
    uv run --python 3.12 --no-project s2/arch_ladder.py --label qwen-hybrid
    uv run --python 3.12 --no-project s2/arch_ladder.py --label gemma-fullattn

CONFOUND THIS CONTROLS: token counts are model-specific, so each size target is
hit against the SERVER UNDER TEST's own /tokenize. "640 tokens" therefore means
640 of that model's tokens in both arms — which is what a token-distance
hypothesis requires. Char counts are recorded so the difference is auditable.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8081"
PREFIX_PATH = Path(__file__).resolve().parents[1] / "prompts" / "leaf-prefix.v1.md"

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


def build_chunk(rng: random.Random, target_tokens: int) -> tuple[str, str, str, str]:
    """Return (chunk, present_entity, present_uuid).

    The planted fact sits at the END of the chunk so that needle-to-question
    distance is minimal and CANNOT explain a failure -- this isolates the
    instruction-distance axis, which is what differs between the arms.
    """
    # THREE bindings, matching the validated fixture design. A single lone
    # entity plus an obviously-foreign ABSENT question makes refusal trivial
    # and the phenomenon under study does not appear at all -- measured on
    # this box before this change, so the earlier shape was not a valid control.
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:3]]
    absent_ent = f"{pool[3]} Trust"   # same family, plausible, genuinely absent
    ent, uid = present[0]
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
    return "DOCUMENT:\n" + body + fact, ent, uid, absent_ent


def apply_template(msgs: list[dict]) -> str:
    """Production passes enable_thinking=false (spec §4/D15): with thinking on,
    the model spends its whole budget on a preamble and never emits an answer.
    Not every template accepts the kwarg, so fall back for the control model."""
    try:
        return post("/apply-template", {
            "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    except Exception:
        return post("/apply-template", {"messages": msgs})["prompt"]


def ask(prefix: str, chunk: str, question: str, seed: int) -> dict:
    msgs = [{"role": "system", "content": prefix},
            {"role": "user", "content": chunk + "\n\n" + question}]
    rendered = apply_template(msgs)
    r = post("/completion", {"prompt": rendered, "n_predict": 160,
                             "cache_prompt": False, "temperature": 0.3,
                             "top_p": 0.9, "seed": seed})
    t = r.get("timings", {})
    raw = (r.get("content") or "")
    if "</think>" in raw:            # keep the tail after the LAST close tag
        raw = raw.rsplit("</think>", 1)[1]
    return {"answer": raw.strip(),
            "prompt_n": t.get("prompt_n"), "rendered_tokens": ntok(rendered)}


REFUSAL = re.compile(
    r"\b(none|not (present|found|stated|mentioned|contain)|no (record|key|mention)"
    r"|does not (contain|mention|state)|cannot find|unable to find)\b", re.I)


def classify(answer: str, uid: str | None) -> str:
    a = answer.strip()
    if not a or not re.search(r"[A-Za-z0-9]", a):
        return "MALFORMED"
    if uid and uid.lower() in a.lower():
        return "CORRECT"
    if REFUSAL.search(a):
        return "REFUSED"
    return "ANSWERED_ANYWAY"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", nargs="*", type=int,
                    default=[640, 1024, 2048, 4096])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    prefix = strip_changelog(PREFIX_PATH.read_text(encoding="utf-8"))
    print(f"prefix tokens: {ntok(prefix)}   label: {a.label}")
    rows = []
    for size in a.sizes:
        for trial in range(a.trials):
            rng = random.Random(1000 * size + trial)
            chunk, ent, uid, absent_ent = build_chunk(rng, size)
            for qtype, q, expect in (
                ("LITERAL", f"What is the custody key for the {ent}?", uid),
                ("ABSENT", f"What is the custody key for the {absent_ent}?", None),
            ):
                r = ask(prefix, chunk, q, seed=trial + 1)
                cls = classify(r["answer"], expect)
                rows.append({"label": a.label, "size": size, "trial": trial,
                             "qtype": qtype, "cls": cls, "uid": uid,
                             "chunk_tokens": ntok(chunk),
                             "rendered_tokens": r["rendered_tokens"],
                             "answer": r["answer"][:200]})
                print(f"  {size:>5} t{trial} {qtype:<8} -> {cls}")

    out = Path(a.out or f"s2/results/arch_ladder_{a.label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== {a.label} ===")
    print(f"{'size':>6} {'chunk_tok':>10} {'LITERAL correct':>16} {'ABSENT refused':>15}")
    for size in a.sizes:
        lit = [r for r in rows if r["size"] == size and r["qtype"] == "LITERAL"]
        abs_ = [r for r in rows if r["size"] == size and r["qtype"] == "ABSENT"]
        ct = lit[0]["chunk_tokens"] if lit else 0
        print(f"{size:>6} {ct:>10} "
              f"{sum(r['cls'] == 'CORRECT' for r in lit):>7}/{len(lit):<8} "
              f"{sum(r['cls'] == 'REFUSED' for r in abs_):>7}/{len(abs_):<7}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
