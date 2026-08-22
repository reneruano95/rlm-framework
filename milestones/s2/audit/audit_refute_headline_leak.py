"""INDEPENDENT foreign-identifier scan + slot-discipline check.

Nothing here trusts the records' own `leak_detected` / `slot_ok` fields, nor any
prior audit artefact: the chunk text is re-read from the fixture files and keyed
by a freshly computed sha256.
"""
import hashlib, json, re, collections
from pathlib import Path

ROOT = Path("D:/PROJECTS/rlm-halo-framework/milestones/s2")

PATS = {
    "uuid":  re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
    "ent":   re.compile(r"\bENT-\d{4,6}\b"),
    "hex32": re.compile(r"\b[0-9a-fA-F]{32}\b"),
}

# ---- corpus ---------------------------------------------------------------
chunks = {}                       # sha256 -> text
for p in sorted(ROOT.glob("fixtures*/**/*.chunk.txt")):
    t = p.read_text(encoding="utf-8")
    chunks[hashlib.sha256(t.encode("utf-8")).hexdigest()] = t
ids_of = {h: {k: set(v.lower() for v in pat.findall(t)) for k, pat in PATS.items()}
          for h, t in chunks.items()}
owner = collections.defaultdict(set)
for h, d in ids_of.items():
    for k in ("uuid", "ent"):
        for v in d[k]:
            owner[v].add(h)
print(f"corpus: {len(chunks)} chunk files")
for k in ("uuid", "ent"):
    allv = {v for h in ids_of for v in ids_of[h][k]}
    multi = [v for v in allv if len(owner[v]) > 1]
    print(f"  {k}: {len(allv)} distinct, {len(multi)} appear in >1 chunk"
          + (f"  e.g. {multi[:3]}" if multi else ""))

def scan(path, label):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok" and r.get("raw_output") is not None]
    print(f"\n===== {label}  rows={len(rows)} scored={len(ok)} =====")

    # --- slot discipline, computed from raw fields only
    per_slot = collections.defaultdict(set)
    mism, noreq, missing_sha = 0, 0, 0
    for r in ok:
        sha = r.get("chunk_sha256")
        if sha is None: missing_sha += 1; continue
        per_slot[r.get("slot_id")].add(sha)
        if "requested_slot" in r:
            if r["slot_id"] != r["requested_slot"]: mism += 1
        else:
            noreq += 1
    multi_chunk = {s: len(v) for s, v in per_slot.items() if len(v) > 1}
    print(f"  slots used = {len(per_slot)}   slot_id != requested_slot: {mism}"
          f"   rows with no requested_slot field: {noreq}   rows w/o chunk_sha256: {missing_sha}")
    print(f"  slots that held >1 distinct chunk_sha256: {len(multi_chunk)} {multi_chunk}")

    # --- foreign identifier scan
    tot = collections.Counter()
    foreign_rows, unresolved = [], 0
    for r in ok:
        sha = r.get("chunk_sha256")
        raw = r.get("raw_output") or ""
        q = r.get("question") or ""
        if sha not in ids_of:
            unresolved += 1; continue
        own, sent_text = ids_of[sha], chunks[sha] + "\n\n" + q
        for k, pat in PATS.items():
            for v in {m.lower() for m in pat.findall(raw)}:
                tot[f"{k}:total"] += 1
                if v in own[k] or v in sent_text.lower():
                    tot[f"{k}:in_sent"] += 1
                elif k in ("uuid", "ent") and v in owner and owner[v] - {sha}:
                    tot[f"{k}:FOREIGN"] += 1
                    foreign_rows.append((r, k, v, sorted(owner[v] - {sha})))
                else:
                    tot[f"{k}:fabricated"] += 1
    print(f"  unresolved chunk_sha256 rows skipped: {unresolved}")
    for k in ("uuid", "ent", "hex32"):
        print(f"  {k:<6} emitted={tot[k+':total']:<5} in_sent={tot[k+':in_sent']:<5} "
              f"FOREIGN={tot[k+':FOREIGN']:<4} fabricated={tot[k+':fabricated']}")
    for r, k, v, others in foreign_rows[:15]:
        print(f"    !! FOREIGN {k} {v} in {r.get('cell_uid') or r.get('cell_id')} "
              f"({r.get('arm','-')}/{r.get('question_type')}) owned by {[o[:8] for o in others]}")
    return len(foreign_rows)

for f, lab in [("results/sweep.jsonl", "sweep"),
               ("results/refusal-ab.jsonl", "refusal-ab"),
               ("results/refusal-ab-640.jsonl", "refusal-ab-640"),
               ("results/distance.jsonl", "distance"),
               ("results/sweep-run1-shared-server.jsonl", "CONTROL sweep-run1-shared-server")]:
    scan(ROOT / f, lab)
