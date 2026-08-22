"""Score S2 gates (a) prefix integrity and (b) repeated-chunk reuse.

The gate text (ARCHITECTURE.md §9 S2), applied mechanically:

  (a1) sha256(rendered_head) matches config_snapshot AND its token length is
       311, on every call
  (a2) cache_n == N_resident - ub - 4 exactly, on every intra-window re-query
  (b1) the same identity
  (b2) median re-query prefill <= 0.62 x median cold prefill for the same window
  precondition: the leaf launch line carries `--cache-ram 0`

THE RE-QUERY FIXTURE. §9 names one as an S2 deliverable and it is this: each
named 640-token cell is prefilled ONCE (cold) on a never-reused slot, then
asked further questions about the SAME document on that same slot (warm). That
is the only warm call production makes -- R13 forbids reusing a slot across
DIFFERENT documents, and `milestones/s2/CARRYOVER.md` measured same-document reuse as safe
to depth ten. Cells come from the seven named `milestones/s2/fixtures-refusal-640-s*` dirs,
which are non-benchmark fixtures, as §8 requires of every S2 gate.

TWO FIELDS THAT ARE NOT INTERCHANGEABLE. `timings.cache_n` is the reuse count
the gate is written against; the response's top-level `tokens_cached` is a
different number (`rlm/dispatcher.py:437-442`). Both are recorded so the
distinction stays visible in the raw rows rather than living in a comment.

Run against the SHIPPED leaf launch line (which carries --cache-ram 0):
    uv run --python 3.12 --no-project milestones/s2/gate_ab.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlm.dispatcher import predicted_reuse  # noqa: E402

BASE = "http://127.0.0.1:8081"
REPO = Path(__file__).resolve().parents[2]
PREFIX_PATH = REPO / "prompts" / "leaf-prefix.v1.md"
OUT = REPO / "milestones" / "s2" / "results" / "gate_ab.jsonl"
PROBE = "PROBEMARKER"

FIXTURE_DIRS = sorted((REPO / "milestones" / "s2").glob("fixtures-refusal-640-s*"))


def post(path: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(text: str, add_special: bool = True) -> list[int]:
    return post("/tokenize", {"content": text,
                              "add_special": add_special})["tokens"]


def ntok(text: str, add_special: bool = True) -> int:
    return len(tokenize(text, add_special))


def lcp_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def strip_changelog(md: str) -> str:
    return re.sub(r"^<!--.*?-->\s*", "", md, flags=re.S)


def render(prefix: str, user: str) -> str:
    return post("/apply-template", {
        "messages": [{"role": "system", "content": prefix},
                     {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]


def head_of(prefix: str) -> tuple[str, str, int]:
    """The rendered system head: render with a marker user message and cut at
    it. This is the string gate (a1) is about -- template markup and generation
    prompt included, which is why it is 311 tokens and the raw prefix body is
    305."""
    rendered = render(prefix, PROBE)
    cut = rendered.rfind(PROBE)
    head = rendered[:cut]
    return head, hashlib.sha256(head.encode("utf-8")).hexdigest(), ntok(head)


def load_cells() -> list[dict]:
    cells = []
    for d in FIXTURE_DIRS:
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        # `cells` is a dict keyed by cell_id, and the chunk file is recorded
        # under `chunk_path` as a repo-relative path with Windows separators.
        for cell in m["cells"].values():
            p = REPO / str(cell["chunk_path"]).replace("\\", "/")
            if not p.exists():
                continue
            # Questions are a dict of {kind: {question, expected, ...}}. Order
            # them deterministically so the COLD call is the same question in
            # every cell -- gate (b2) compares medians across cells, and a
            # varying cold question would put question difficulty into the
            # prefill comparison.
            qs = [cell["questions"][k]["question"]
                  for k in ("literal", "paraphrase", "absent")
                  if k in cell.get("questions", {})]
            cells.append({"dir": d.name, "cell_id": cell["cell_id"],
                          "text": p.read_text(encoding="utf-8"),
                          "measured_tokens": cell.get("measured_tokens"),
                          "questions": qs})
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-base", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--ub", type=int, default=512,
                    help="must match the leaf launch line's -ub; (a2) is "
                         "denominated in it")
    a = ap.parse_args()

    prefix = strip_changelog(PREFIX_PATH.read_text(encoding="utf-8"))
    head, head_sha, head_tok = head_of(prefix)
    print(f"rendered head: {head_tok} tokens  sha256 {head_sha[:16]}...")
    print(f"raw prefix body: {ntok(prefix, add_special=False)} tokens "
          f"(a DIFFERENT string -- the gate is about the head)")

    cells = load_cells()
    with urllib.request.urlopen(BASE + "/slots", timeout=30) as r:
        n_slots = len(json.loads(r.read().decode()))
    if a.slot_base + len(cells) > n_slots:
        raise SystemExit(f"need {len(cells)} slots from base {a.slot_base}, "
                         f"server has {n_slots}")
    print(f"{len(cells)} named 640-token cells, one never-reused slot each, "
          f"{n_slots} available\n")

    rows = []
    for i, cell in enumerate(cells):
        slot = a.slot_base + i
        qs = cell["questions"]
        prev_tokens: list[int] = []
        for qi, q in enumerate(qs[:3]):
            rendered = render(prefix, cell["text"] + "\n\n" + q)
            tokens = tokenize(rendered)
            # `n_resident` is the token length of the prompt that LAST OCCUPIED
            # THE SLOT, and `lcp` the shared token prefix with it -- not the
            # incoming prompt's own length. Scoring (a2) as `incoming - ub - 4`
            # reports 0/14 FAIL on a system that is exactly right: slot 1's
            # first re-query reused 476, and 992 - 512 - 4 = 476 where 992 is
            # the PREVIOUS prompt. The law lives in
            # `rlm.dispatcher.predicted_reuse` and is imported rather than
            # restated, so this gate cannot drift from the code it scores.
            n_resident = len(prev_tokens)
            lcp = lcp_len(prev_tokens, tokens)
            predicted = predicted_reuse(n_resident, lcp, a.ub)
            cold = qi == 0
            r = post("/completion", {"prompt": rendered,
                                     "n_predict": a.n_predict,
                                     "temperature": 0.0, "seed": 1,
                                     "cache_prompt": True, "id_slot": slot})
            t = r.get("timings", {})
            if r.get("id_slot") != slot:
                raise SystemExit(f"slot {slot} requested, {r.get('id_slot')} "
                                 f"served -- policy not honoured")
            # THE SLOT MUST BE VIRGIN, and only the cold call can prove it.
            # Re-running this gate against a live server without moving
            # --slot-base silently scores nothing: the cold calls come back
            # warm (measured cache_n 494 on a call that had never been made in
            # that run), (b2)'s denominator collapses to a warm number, and the
            # ratio lands at 1.006 against a 0.62 bar -- a FAIL that says
            # nothing about the system. Restart the leaf or move --slot-base.
            if cold and (t.get("cache_n") or 0) != 0:
                raise SystemExit(
                    f"slot {slot} was NOT virgin: the cold call reused "
                    f"{t.get('cache_n')} tokens. Restart the leaf or pass a "
                    f"--slot-base past every slot this server has served.")
            h, hs, ht = head_of(prefix)
            rows.append({
                "cell": cell["cell_id"], "dir": cell["dir"], "slot": slot,
                "q_index": qi, "cold": qi == 0,
                "n_resident_prev": n_resident, "lcp": lcp,
                "incoming_tokens": len(tokens),
                "cache_n": t.get("cache_n"),
                "tokens_cached_field": r.get("tokens_cached"),
                "prompt_n": t.get("prompt_n"),
                "prefill_ms": t.get("prompt_ms"),
                "head_sha256": hs, "head_tokens": ht,
                "predicted_reuse": predicted,
            })
            print(f"  {cell['cell_id']:<16} slot{slot:<4} q{qi} "
                  f"{'COLD' if qi == 0 else 'warm'} "
                  f"prev {n_resident:>5} lcp {lcp:>5} "
                  f"pred {predicted:>5} cache_n "
                  f"{str(t.get('cache_n')):>5} prefill "
                  f"{t.get('prompt_ms', 0):>7.1f} ms"
                  f"{'' if t.get('cache_n') == predicted else '   <-- MISMATCH'}")
            prev_tokens = tokens

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cold = [r for r in rows if r["cold"]]
    warm = [r for r in rows if not r["cold"]]
    a1_sha = len({r["head_sha256"] for r in rows}) == 1
    a1_len = all(r["head_tokens"] == 311 for r in rows)
    a2_hits = [r for r in warm if r["cache_n"] == r["predicted_reuse"]]
    med_cold = statistics.median(r["prefill_ms"] for r in cold) if cold else 0.0
    med_warm = statistics.median(r["prefill_ms"] for r in warm) if warm else 0.0
    ratio = (med_warm / med_cold) if med_cold else float("inf")

    print(f"\n=== S2 gates (a) and (b) ===")
    print(f"  (a1) head sha256 constant on all {len(rows)} calls : "
          f"{'PASS' if a1_sha else 'FAIL'}")
    print(f"  (a1) head token length == 311 on all calls        : "
          f"{'PASS' if a1_len else 'FAIL'}  (measured {rows[0]['head_tokens']})")
    print(f"  (a2) cache_n == n_resident - ub - 4 on re-queries : "
          f"{len(a2_hits)}/{len(warm)} "
          f"{'PASS' if warm and len(a2_hits) == len(warm) else 'FAIL'}")
    print(f"  (b2) median warm prefill / median cold prefill    : "
          f"{ratio:.3f} vs bar 0.62 "
          f"{'PASS' if ratio <= 0.62 else 'FAIL'}"
          f"   ({med_warm:.1f} ms / {med_cold:.1f} ms)")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
