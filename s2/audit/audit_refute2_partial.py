"""REFUTE pass 2, adversarial arm 2: a leak that arrives as a FRAGMENT.

refusal-ab carries 187 MALFORMED rows; a whole-token detector would miss a
leak that surfaces as a partial identifier. This looks for any >=10-character
substring of a FOREIGN fixture UUID inside any answer, plus the final
foreign-identifier tally with cell_uid used as a fallback chunk resolver.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["s2/results/sweep.jsonl", "s2/results/refusal-ab.jsonl",
         "s2/results/refusal-ab-640.jsonl", "s2/results/distance.jsonl",
         "s2/results/sweep-run1-shared-server.jsonl"]
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                  r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ENT = re.compile(r"\bENT-\d{4,6}\b")


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


idx = {}
for f in sorted(ROOT.glob("s2/fixtures*/**/*.chunk.txt")):
    t = f.read_text(encoding="utf-8")
    idx[hashlib.sha256(t.encode("utf-8")).hexdigest()] = (f.as_posix(), t)

uuid_owner = {}
for h, (p, t) in idx.items():
    for u in set(x.lower() for x in UUID.findall(t)):
        uuid_owner.setdefault(u, set()).add(h)
ent_owner = {}
for h, (p, t) in idx.items():
    for e in set(ENT.findall(t)):
        ent_owner.setdefault(e, set()).add(h)
print(f"corpus: {len(idx)} chunks, {len(uuid_owner)} uuids, {len(ent_owner)} ENT codes")

uid2sha = {}
for fp in FILES:
    for r in load(fp):
        if r.get("cell_uid") and r.get("chunk_sha256"):
            uid2sha[r["cell_uid"]] = r["chunk_sha256"]

K = 10
grand_rows = grand_scanned = 0
for fp in FILES:
    rows = load(fp)
    whole = frag = scanned = unres = noout = 0
    fragdetail = []
    for r in rows:
        out = (r.get("raw_output") or "").lower()
        if not out:
            noout += 1
            continue
        sha = r.get("chunk_sha256") or uid2sha.get(r.get("cell_uid"))
        if sha not in idx:
            unres += 1
            continue
        scanned += 1
        mine_raw = idx[sha][1]
        mine = mine_raw.lower()
        # whole-token foreign (ENT codes are case-sensitive: compare raw)
        got_u = set(x.lower() for x in UUID.findall(out))
        got_e = set(e.upper() for e in ENT.findall((r.get("raw_output") or "").upper()))
        bad = ([g for g in got_u if g not in mine and g in uuid_owner]
               + [g for g in got_e if g not in mine_raw.upper() and g in ent_owner])
        if bad:
            whole += 1
            fragdetail.append((r.get("cell_id"), bad, "WHOLE", out[:70]))
        # fragment foreign: any K-char substring of a foreign uuid
        for u, owners in uuid_owner.items():
            if u in mine:
                continue
            hit = None
            for i in range(0, len(u) - K + 1):
                s = u[i:i + K]
                if "-" in s[:1] or "-" in s[-1:]:
                    continue
                if s in out:
                    hit = s
                    break
            if hit:
                frag += 1
                fragdetail.append((r.get("cell_id"), u, hit, out[:70]))
                break
    print(f"\n{fp}: rows={len(rows)} scanned={scanned} no-output={noout} unresolved={unres}")
    print(f"   whole-token FOREIGN rows: {whole}")
    print(f"   >={K}-char FOREIGN FRAGMENT rows: {frag}")
    for d in fragdetail[:10]:
        print(f"      {d[0]} foreign={d[1]} frag={d[2]!r} out={d[3]!r}")
    if fp != FILES[-1]:
        grand_rows += len(rows)
        grand_scanned += scanned
print(f"\nfour headline probes: {grand_rows} rows, {grand_scanned} with an answer and a resolvable chunk")
