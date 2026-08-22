"""REFUTE pass 2: re-derive every remaining numeric claim in the report."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                  r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ENT = re.compile(r"\bENT-\d{4,6}\b")
HEX4 = re.compile(r"(?<![0-9a-zA-Z-])[0-9a-f]{4}(?![0-9a-zA-Z-])")


def ids(text, hex4=True):
    out = set(u.lower() for u in UUID.findall(text)) | set(ENT.findall(text))
    if hex4:
        out |= set(h.lower() for h in HEX4.findall(UUID.sub(" ", text)))
    return out


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def index():
    idx = {}
    for f in sorted(ROOT.glob("milestones/s2/fixtures*/**/*.chunk.txt")):
        t = f.read_text(encoding="utf-8")
        idx[hashlib.sha256(t.encode("utf-8")).hexdigest()] = (str(f.relative_to(ROOT)), t)
    return idx


def sec(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    idx = index()

    # ---------------- 1. identifier uniqueness per file ----------------
    sec("1. identifier uniqueness (report: distance 140/140, refusal-1024 50/50, "
        "refusal-640 29/29, sweep 740/743)")
    for fp in ("milestones/s2/results/distance.jsonl", "milestones/s2/results/refusal-ab.jsonl",
               "milestones/s2/results/refusal-ab-640.jsonl", "milestones/s2/results/sweep.jsonl"):
        rows = load(fp)
        used = {r["chunk_sha256"] for r in rows if r.get("chunk_sha256") in idx}
        for hex4 in (True, False):
            m = defaultdict(set)
            for h in used:
                for i in ids(idx[h][1], hex4):
                    m[i].add(h)
            uniq = sum(1 for i, hs in m.items() if len(hs) == 1)
            print(f"  {fp:38s} chunks={len(used):3d} hex4={hex4!s:5s} "
                  f"unique={uniq}/{len(m)}")

    # ---------------- 2. slot discipline ----------------
    sec("2. slot discipline (report: refusal-ab 43 slots, refusal-640 28, "
        "distance 85; zero slots with >1 chunk_sha256; slot_ok/leak_detected present)")
    for fp in ("milestones/s2/results/refusal-ab.jsonl", "milestones/s2/results/refusal-ab-640.jsonl",
               "milestones/s2/results/distance.jsonl", "milestones/s2/results/sweep.jsonl"):
        rows = load(fp)
        slots = defaultdict(set)
        for r in rows:
            if r.get("slot_id") is None or not r.get("chunk_sha256"):
                continue
            slots[r["slot_id"]].add(r["chunk_sha256"])
        multi = {s: len(c) for s, c in slots.items() if len(c) > 1}
        so = Counter(r.get("slot_ok", "MISSING") for r in rows)
        ld = Counter(r.get("leak_detected", "MISSING") for r in rows)
        okmatch = sum(1 for r in rows
                      if r.get("id_slot") is not None
                      and r.get("id_slot") == r.get("slot_id"))
        print(f"  {fp:38s} distinct slot_id={len(slots)} "
              f"slots-with->1-chunk={len(multi)} {multi if multi else ''}")
        print(f"      slot_ok={dict(so)}  leak_detected={dict(ld)}  "
              f"id_slot==slot_id in {okmatch}/{len(rows)}")

    # ---------------- 3. distance headline re-derivation ----------------
    sec("3. distance headline (report: REPLICATION 640->0/21, 1024->30/30 "
        "[25 own, 2 fabricated, 3 no-id, 0 foreign]; A-shipped 640->0/24, 1024->24/24 "
        "[23 own, 1 fabricated])")
    rows = load("milestones/s2/results/distance.jsonl")
    corpus = defaultdict(set)
    for h, (p, t) in idx.items():
        for i in ids(t, False):
            corpus[i].add(h)
    for arm in ("REPLICATION", "A-shipped", "B-repeated", "C-after"):
        for size in (640, 1024, 2048):
            ab = [r for r in rows if r.get("arm") == arm
                  and str(r.get("size_target")) == str(size)
                  and r.get("question_type") == "absent"]
            lit = [r for r in rows if r.get("arm") == arm
                   and str(r.get("size_target")) == str(size)
                   and r.get("question_type") == "literal"]
            if not ab and not lit:
                continue
            fp_ = [r for r in ab if r.get("label") == "FALSE-POSITIVE"]
            c = Counter()
            for r in fp_:
                sha = r.get("chunk_sha256")
                mine = idx.get(sha)
                myids = ids(mine[1], False) if mine else set()
                got = ids(r.get("raw_output") or "", False)
                if not got:
                    c["no-id"] += 1
                elif got & myids:
                    c["own"] += 1
                elif got & set(corpus):
                    c["FOREIGN"] += 1
                else:
                    c["fabricated"] += 1
            print(f"  {arm:12s} {size:>5}  ABSENT n={len(ab):3d} FP={len(fp_):3d} "
                  f"{dict(c)}   | LITERAL n={len(lit):3d} "
                  f"CORRECT={sum(1 for r in lit if r.get('label')=='CORRECT')} "
                  f"MISS={sum(1 for r in lit if r.get('label')=='MISS')}")

    # ---------------- 4. classifier gap ----------------
    sec("4. classifier gap: FALSE-POSITIVE rows whose text is actually a refusal")
    REF = re.compile(r"cannot answer|not provided|does not contain|no mention|"
                     r"not (?:present|found|stated|mentioned)|unable to", re.I)
    for fp in ("milestones/s2/results/refusal-ab.jsonl", "milestones/s2/results/distance.jsonl",
               "milestones/s2/results/sweep.jsonl", "milestones/s2/results/refusal-ab-640.jsonl"):
        rr = load(fp)
        f_ = [r for r in rr if r.get("label") == "FALSE-POSITIVE"]
        pros = [r for r in f_ if REF.search(r.get("raw_output") or "")]
        print(f"  {fp:38s} FP={len(f_):3d} refusal-looking={len(pros)}")
        for r in pros[:6]:
            print(f"      {(r.get('raw_output') or '')[:110]!r}")

    # ---------------- 5. slot-capacity truncation ----------------
    sec("5. slot capacity (report: 2,560 tok/slot; at 2048 the prompt leaves <512 "
        "tokens in 144/144 distance and 27/27 refusal-ab calls; 5 and 3 truncated=true; "
        "0 such at 640/1024)")
    CAP = 2560
    for fp in ("milestones/s2/results/distance.jsonl", "milestones/s2/results/refusal-ab.jsonl",
               "milestones/s2/results/refusal-ab-640.jsonl", "milestones/s2/results/sweep.jsonl"):
        rr = load(fp)
        for size in (640, 1024, 2048, 4096):
            g = [r for r in rr if str(r.get("size_target")) == str(size)
                 and r.get("tokens_in") is not None]
            if not g:
                continue
            tight = [r for r in g if CAP - int(r["tokens_in"]) < 512]
            trunc = [r for r in g if str(r.get("truncated")).lower() == "true"]
            print(f"  {fp:34s} size={size:>5} n={len(g):3d} "
                  f"headroom<512: {len(tight)}/{len(g)}  truncated=true: {len(trunc)}"
                  f"  tokens_in[min,max]=[{min(int(r['tokens_in']) for r in g)},"
                  f"{max(int(r['tokens_in']) for r in g)}]")

    # ---------------- 6. sweep restart evidence ----------------
    sec("6. sweep.jsonl restart evidence (report: all slot_id 0, cell gaps "
        "11,12,16,25,48,72,11,12,13,66,26,52 s)")
    rr = load("milestones/s2/results/sweep.jsonl")
    print("  slot_id values:", dict(Counter(r.get("slot_id") for r in rr)))
    ts = [datetime.fromisoformat(r["ts"]) for r in rr]
    cells = [r["cell_id"] for r in rr]
    gaps = []
    for i in range(1, len(rr)):
        if cells[i] != cells[i - 1]:
            gaps.append(round((ts[i] - ts[i - 1]).total_seconds()))
    print(f"  cell-boundary gaps (s): {gaps}")
    print(f"  n boundaries={len(gaps)}  distinct cells={len(set(cells))}")


if __name__ == "__main__":
    main()
