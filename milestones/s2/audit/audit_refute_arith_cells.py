"""Per-cell breakdown of the leak-* arms with the independent oracle, to test
the r13-truth auditor's like-for-like table:
   default 9/9, --cache-ram 0 9/9, -ctxcp 0 4/9, --cache-ram 0 -ctxcp 0 4/9,
   virgin 0/9  (all at s2-4096-p50 with ONE prior document)
   and 'cell-for-cell identical 0/9,2/9,6/9,7/9' for default vs
   --no-cache-idle-slots, 15/27 vs 15/27 exposed.
Also prints, per file, the slot history so 'one prior document' can be checked.
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import audit_refute_arith_oracle as O   # reuse the independent oracle


def per_cell(name: str):
    p = O.S2 / "results" / name
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    tot = Counter()
    leak = Counter()
    hist = defaultdict(list)
    order = OrderedDict()
    for r in rows:
        cell = r.get("cell_id")
        order.setdefault(cell, None)
        tot[cell] += 1
        s = r.get("slot_id")
        d = r.get("chunk_sha256")
        prior = len(hist[s])
        if d not in hist[s]:
            hist[s].append(d)
        own, how = O.resolve(r)
        ans = r.get("raw_output") or ""
        if own is None or not ans:
            continue
        q = O.tokens_of(r.get("question") or "")
        ot = O.CORPUS_TOKENS[own]
        hits = [t for t in O.tokens_of(ans)
                if t not in ot and t not in q
                and any(t in O.CORPUS_TOKENS[k] for k in O.CORPUS_TOKENS if k != own)]
        if hits:
            leak[cell] += 1
    print(f"{name:34s} total={sum(leak.values())}/{sum(tot.values())}")
    for c in order:
        # prior-document count for the first row of this cell
        print(f"    {str(c):18s} {leak[c]}/{tot[c]}")
    prior_map = {s: len(v) for s, v in hist.items()}
    print(f"    slots -> distinct docs held: {prior_map}")
    ts = [r.get("ts") for r in rows if r.get("ts")]
    print(f"    ts: {ts[0]} .. {ts[-1]}   cold(tokens_cached==0)="
          f"{sum(1 for r in rows if r.get('tokens_cached') == 0)}/{len(rows)}")


if __name__ == "__main__":
    for f in ("leak-erase.jsonl", "leak-cram0.jsonl", "leak-ctxcp0.jsonl",
              "leak-nocram.jsonl", "leak-slotiso.jsonl", "leak-nocacheidle.jsonl",
              "sweep-run1-shared-server.jsonl", "sweep.jsonl"):
        per_cell(f)
        print()
