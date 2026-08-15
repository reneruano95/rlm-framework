"""OFFLINE: every FALSE-POSITIVE of the refusal A/B, decomposed.

For each one:
  * the identifier it handed over, and whether that string is a VERBATIM
    substring of the chunk that was sent (substring, not token equality --
    the model sometimes emits only the tail group of a UUID)
  * whether it is the chunk's own literal/paraphrase key (in-chunk
    misattribution) or something else
  * whether the string occurs in any OTHER fixture cell on disk
  * the server's own cache numbers for that call

Then the same for CONFABULATIONs, and a cold/warm cross-check of the server
log's prompt-eval counts against each record's tokens_in / tokens_cached.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\s2")
sys.path.insert(0, str(S2.parent))
from rlm.leakcheck import identifier_tokens  # noqa: E402

sys.path.insert(0, str(S2 / "audit"))
from audit_refusal_ab import CELLS, load_runs, sent_cell_for  # noqa: E402

LOG = Path(r"D:\PROJECTS\rlm-halo-framework\traces\logs\leaf-server.err.log")
RE_PE = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")


def main():
    recs = load_runs()
    ok = [r for r in recs if r.get("status") == "ok"]

    print("=== every FALSE-POSITIVE, decomposed ===")
    counts = {"in-chunk-verbatim": 0, "in-chunk-fragment": 0,
              "other-fixture": 0, "nowhere": 0, "no-identifier": 0}
    rows = []
    for r in ok:
        if r.get("label") != "FALSE-POSITIVE":
            continue
        cell = sent_cell_for(r)
        chunk = cell["text"]
        low = chunk.lower()
        q = (r.get("question") or "").lower()
        toks = sorted(identifier_tokens(r.get("raw_output") or ""))
        if not toks:
            counts["no-identifier"] += 1
            rows.append((r["_file"], r["_line"], r["arm"], r.get("cell_uid"),
                         "(no identifier)", "no-identifier", ""))
            continue
        for t in toks:
            tl = t.lower()
            if tl in low or tl in q:
                # is it a whole identifier of the chunk, or a fragment of one?
                whole = {x.lower() for x in identifier_tokens(chunk)}
                kind = "in-chunk-verbatim" if tl in whole else "in-chunk-fragment"
                counts[kind] += 1
                # which entity does it belong to?
                owner = "?"
                for qt, spec in cell["questions"].items():
                    exp = (spec.get("expected") or "").lower()
                    if exp and (tl == exp or tl in exp):
                        owner = f"{qt}-key-of[{spec.get('entity')}]"
                rows.append((r["_file"], r["_line"], r["arm"], r.get("cell_uid"),
                             t, kind, owner))
            else:
                others = [(dn, cid) for (dn, cid), c2 in CELLS.items()
                          if tl in c2["text"].lower()]
                kind = "other-fixture" if others else "nowhere"
                counts[kind] += 1
                rows.append((r["_file"], r["_line"], r["arm"], r.get("cell_uid"),
                             t, kind, str(others[:3])))
    for k, v in counts.items():
        print(f"  {k:20s} {v}")
    print()
    for row in rows:
        if row[5] != "in-chunk-verbatim":
            print("  NON-VERBATIM:", row)
    print()
    # a sample of the verbatim ones, to show the misattribution shape
    print("  sample of in-chunk hits (first 8):")
    for row in [r for r in rows if r[5] == "in-chunk-verbatim"][:8]:
        print("   ", row)
    print()

    print("=== CONFABULATIONs (wrong answers to PRESENT facts) ===")
    cc = {"in-chunk": 0, "other-fixture": 0, "nowhere": 0, "no-identifier": 0}
    for r in ok:
        if r.get("label") != "CONFABULATION":
            continue
        cell = sent_cell_for(r)
        low = cell["text"].lower()
        toks = sorted(identifier_tokens(r.get("raw_output") or ""))
        if not toks:
            cc["no-identifier"] += 1
            continue
        for t in toks:
            tl = t.lower()
            if tl in low:
                cc["in-chunk"] += 1
            elif any(tl in c2["text"].lower() for c2 in CELLS.values()):
                cc["other-fixture"] += 1
                print("   OTHER-FIXTURE:", r["_file"], r["_line"], r["arm"],
                      r.get("cell_uid"), t)
            else:
                cc["nowhere"] += 1
                print("   FABRICATED:", r["_file"], r["_line"], r["arm"],
                      r.get("cell_uid"), t, repr((r.get("raw_output") or "")[:80]))
    for k, v in cc.items():
        print(f"  {k:16s} {v}")
    print()

    # CONFABULATION on paraphrase = a NAME. names are not identifier-shaped, so
    # check them against every fixture's text directly.
    print("=== paraphrase CONFABULATIONs: is the wrong NAME in this chunk? ===")
    n_in, n_other, n_no = 0, 0, 0
    for r in ok:
        if r.get("label") != "CONFABULATION" or r.get("question_type") != "paraphrase":
            continue
        cell = sent_cell_for(r)
        ans = " ".join((r.get("raw_output") or "").split())
        words = [w.strip(".,;:'\"()") for w in ans.split()]
        words = [w for w in words if len(w) >= 6 and w[:1].isupper()]
        if not words:
            n_no += 1
            continue
        here = all(w.lower() in cell["text"].lower() for w in words)
        if here:
            n_in += 1
        else:
            miss = [w for w in words if w.lower() not in cell["text"].lower()]
            elsewhere = {w: [(dn, cid) for (dn, cid), c2 in CELLS.items()
                             if w.lower() in c2["text"].lower()][:2] for w in miss}
            if any(elsewhere.values()):
                n_other += 1
                print("   NAME FROM ELSEWHERE:", r["_file"], r["_line"], r["arm"],
                      r.get("cell_uid"), repr(ans[:70]), elsewhere)
            else:
                n_no += 1
                print("   NAME NOWHERE:", r["_file"], r["_line"], r["arm"],
                      r.get("cell_uid"), repr(ans[:70]), miss)
    print(f"  all capitalised words present in own chunk : {n_in}")
    print(f"  some word present only in ANOTHER fixture  : {n_other}")
    print(f"  some word in no fixture at all             : {n_no}")
    print()

    print("=== server log vs record: does any call prefill less than it sent? ===")
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    pe = [(int(m.group(1)), int(m.group(2)), int(m.group(4)))
          for l in lines if (m := RE_PE.search(l))]
    by_slot: dict[int, list[int]] = {}
    for sid, task, ntok in pe:
        by_slot.setdefault(sid, []).append(ntok)
    bad = 0
    checked = 0
    for r in ok:
        sid = r.get("slot_id")
        seq = by_slot.get(sid)
        if not seq:
            continue
        # cold calls only: the first prefill on that slot must equal tokens_in
        if not r.get("cold"):
            continue
        checked += 1
        first = seq[0]
        if abs(first - r.get("tokens_in", 0)) > 2:
            bad += 1
            if bad <= 20:
                print(f"   slot {sid}: first prefill {first} vs record "
                      f"tokens_in {r.get('tokens_in')} cached "
                      f"{r.get('tokens_cached')} ({r['_file']}:{r['_line']})")
    print(f"  cold records checked: {checked}, mismatched first-prefill: {bad}")
    print()

    print("=== tokens_cached distribution by cold/warm ===")
    dist = {}
    for r in ok:
        k = ("cold" if r.get("cold") else "warm",
             "cached=0" if not r.get("tokens_cached") else "cached>0")
        dist[k] = dist.get(k, 0) + 1
    for k in sorted(dist):
        print("  ", k, dist[k])
    # warm calls: is the cached amount consistent with the SAME cell's prefix?
    over = [r for r in ok if not r.get("cold")
            and r.get("tokens_cached", 0) > r.get("tokens_in", 0)]
    print(f"  warm calls where cached > tokens_in: {len(over)}")


if __name__ == "__main__":
    main()
