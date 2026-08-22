"""INDEPENDENT re-derivation of arch_ladder fixture entities/UUIDs.

Only the pre-server part of build_chunk is replicated (rng.sample + the three
UUID draws + absent_ent). Those all happen BEFORE the first /tokenize call
(as-run milestones/s2/arch_ladder.py:76-86 vs the binary search at :88-101), so no server
is needed and no GPU is touched.

Filler is alphabetic words only (as-run :51-62), so an identifier that is not
one of the three `present` UUIDs cannot occur anywhere in the chunk.
"""
import json, random, sys
from pathlib import Path

STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]

def cell(size, trial):
    rng = random.Random(1000 * size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:3]]
    return present, f"{pool[3]} Trust"

SIZES = [640, 1024, 2048]
TRIALS = [0, 1, 2, 3]

print(f"python {sys.version.split()[0]}")
print("=== replicated cells (serve order = size-major, trial-minor) ===")
order = []
cells = {}
for s in SIZES:
    for t in TRIALS:
        pres, absent = cell(s, t)
        cells[(s, t)] = (pres, absent)
        order.append((s, t))
        print(f"{s:>5} t{t}  present={[ (e.split()[0], u) for e,u in pres ]}")
        print(f"        ABSENT-question entity = {absent}")

# global index: uuid -> list of cells it lives in ; (cell, entity) -> uuid
uuid_owner = {}
for (s, t), (pres, absent) in cells.items():
    for e, u in pres:
        uuid_owner.setdefault(u, []).append((s, t, e))

dups = {u: v for u, v in uuid_owner.items() if len(v) > 1}
print(f"\ncross-cell UUID collisions across all 12 cells: {len(dups)}  {dups}")

# entity reuse across cells (same name, different key?)
byname = {}
for (s, t), (pres, absent) in cells.items():
    for e, u in pres:
        byname.setdefault(e, []).append((s, t, u))
print("\n=== entity name -> (cell, key) ===")
for e, v in sorted(byname.items()):
    keys = {u for _, _, u in v}
    print(f"{e:<24} n_cells={len(v)} distinct_keys={len(keys)}  {v}")

print("\n=== resolve the 5 non-refusal ABSENT answers ===")
rows = []
for f in ["milestones/s2/results/arch_ladder_qwen-hybrid.jsonl",
          "milestones/s2/results/arch_ladder_gemma-fullattn.jsonl"]:
    for line in Path(f).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

serve_idx = {c: i for i, c in enumerate(order)}
for r in rows:
    if r["qtype"] != "ABSENT" or r["cls"] == "REFUSED":
        continue
    s, t = r["size"], r["trial"]
    pres, absent = cells[(s, t)]
    ans = r["answer"].strip()
    own = [u for _, u in pres]
    in_doc = any(u in ans for u in own)
    hits = [(cs, ct, e) for u, v in uuid_owner.items() if u in ans for (cs, ct, e) in v]
    print(f"\n{r['label']} {s} t{t} ABSENT")
    print(f"   asked about : {absent}")
    print(f"   answered    : {ans}")
    print(f"   uuids IN the sent chunk: {own}")
    print(f"   answer is in sent chunk : {in_doc}")
    for (cs, ct, e) in hits:
        d = serve_idx[(s, t)] - serve_idx[(cs, ct)]
        print(f"   -> matches key of '{e}' in cell {cs}/t{ct}  "
              f"(entity match={e == absent}, cells earlier={d})")
    if not hits:
        print("   -> matches NO cell in the 12-cell grid (fabricated / foreign to grid)")
