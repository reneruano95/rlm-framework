"""INDEPENDENT refutation pass on the blast-radius report.

Does NOT trust: the runners' own leak_detected flag, the blast-radius report's
own scripts, or the arch-ladder classifier. Rebuilds every join from raw files.

Tests, in order:
  T1  arch-ladder offline oracle: replicate bindings(size,trial) from the
      COMMITTED arch_ladder.py rng and attribute every emitted uuid.
      Non-leakage mechanism under test: is the emitted uuid ALSO in the record's
      own chunk (misattribution), or does it collide with another fixture by
      accident?
  T2  detector POWER on the load-bearing runs: how many records actually
      resolve their own chunk, and how big is the foreign-donor universe?
      A detector that cannot resolve own-chunk, or has no donors to find,
      reports "clean" for free.
  T3  donor-universe completeness: are all served documents on disk? A leak
      from a document the auditor never loaded is invisible.
  T4  fixture identifier OVERLAP: if fixtures share identifiers (nesting,
      shared corpus), "identifier in own chunk" cannot exclude a leak.
  T5  slot hygiene recomputed from chunk_sha256 (not from `cold`).
  T6  chunk-level reuse: same document on >1 slot, and >1 document per slot.

Offline, stdlib only. No HTTP, no GPU.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\s2")
RES = S2 / "results"

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
ENT_RE = re.compile(r"\bENT-\d{4,6}\b")

STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]


def load(name):
    out = []
    with (RES / name).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- T1
def bindings(size: int, trial: int):
    rng = random.Random(1000 * size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:3]]
    return present, f"{pool[3]} Trust"


def t1_archladder():
    print("=" * 78)
    print("T1  arch-ladder offline oracle (bindings replicated from rng)")
    print("=" * 78)
    sizes, trials = [640, 1024, 2048], range(4)
    fix = {}
    for s in sizes:
        for t in trials:
            fix[(s, t)] = bindings(s, t)
    # global uuid -> owner map; check for accidental collisions first
    owner = collections.defaultdict(list)
    for (s, t), (present, absent) in fix.items():
        for e, u in present:
            owner[u.lower()].append((s, t, e))
    dup = {u: v for u, v in owner.items() if len(v) > 1}
    print(f"  distinct planted uuids: {len(owner)}  "
          f"ACCIDENTAL CROSS-FIXTURE COLLISIONS: {len(dup)}  {dup or ''}")
    for (s, t), (present, absent) in sorted(fix.items()):
        print(f"    {s:>5}/t{t}  absent={absent:<22} present="
              + ", ".join(f"{e.split()[0]}={u[:8]}" for e, u in present))

    for f in ("arch_ladder_qwen-hybrid.jsonl", "arch_ladder_gemma-fullattn.jsonl",
              "arch_ladder_qwen_virgin.jsonl", "arch_ladder_qwen_shared.jsonl"):
        p = RES / f
        if not p.exists():
            continue
        print(f"\n  --- {f}")
        for r in load(f):
            if r["qtype"] != "ABSENT":
                continue
            s, t = r["size"], r["trial"]
            present, absent = fix[(s, t)]
            own = {u.lower(): e for e, u in present}
            ans = r["answer"]
            for m in UUID_RE.findall(ans):
                u = m.lower()
                if u in own:
                    verdict = f"OWN CHUNK (bound to {own[u]})"
                elif u in owner:
                    src = owner[u]
                    # was the asked (absent) entity the one it is bound to?
                    ent_match = any(e == absent for _, _, e in src)
                    verdict = (f"FOREIGN fixture {src} "
                               f"{'ENTITY-CORRECT' if ent_match else 'entity-mismatch'}")
                else:
                    verdict = "matches NOTHING planted (fabricated)"
                print(f"    {s:>5}/t{t} asked={absent:<22} emitted={u[:13]}.. -> {verdict}")


# ---------------------------------------------------------------- chunks
def load_chunks():
    out = {}
    for p in sorted(S2.glob("fixtures*/**/*.chunk.txt")):
        text = p.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out.setdefault(h, []).append((f"{p.parent.name}/{p.name}", text))
    return out


def ids_of(text):
    return set(x.lower() for x in UUID_RE.findall(text)) | \
           set(x.upper() for x in ENT_RE.findall(text))


def t2_t3_t4(chunks):
    print("\n" + "=" * 78)
    print("T2/T3  detector POWER: own-chunk resolution + donor universe")
    print("=" * 78)
    print(f"  fixture .chunk.txt files on disk: "
          f"{sum(len(v) for v in chunks.values())} "
          f"({len(chunks)} distinct sha256)")

    disk_ids = {}
    for h, lst in chunks.items():
        for lbl, txt in lst:
            disk_ids[lbl] = ids_of(txt)
    all_disk_ids = set().union(*disk_ids.values()) if disk_ids else set()
    print(f"  distinct identifiers across all fixture files on disk: {len(all_disk_ids)}")

    for name in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
                 "sweep.jsonl", "sweep-run1-shared-server.jsonl"):
        recs = [r for r in load(name) if r.get("status") in (None, "ok")]
        shas = collections.Counter(r.get("chunk_sha256") for r in recs)
        resolved = sum(c for s, c in shas.items() if s in chunks)
        distinct = set(shas) - {None}
        distinct_res = {s for s in distinct if s in chunks}
        print(f"\n  {name}: {len(recs)} ok records")
        print(f"     distinct chunk_sha256 served      : {len(distinct)}")
        print(f"     ...that exist on disk as .chunk.txt: {len(distinct_res)}"
              f"   ({100*len(distinct_res)/max(1,len(distinct)):.1f}%)")
        print(f"     records whose OWN chunk resolves   : {resolved}/{len(recs)}"
              f"   ({100*resolved/max(1,len(recs)):.1f}%)")
        # how many DOCUMENTS actually served vs how many donors auditor can see
        print(f"     >>> donor universe the auditor searched = {len(chunks)} files; "
              f"documents actually served in this run = {len(distinct)}")
        if len(distinct_res) < len(distinct):
            miss = sorted(distinct - distinct_res)[:3]
            print(f"     >>> UNRESOLVED example sha: {miss}")


def t4_overlap(chunks):
    print("\n" + "=" * 78)
    print("T4  fixture identifier OVERLAP (can 'in own chunk' exclude a leak?)")
    print("=" * 78)
    files = []
    for h, lst in chunks.items():
        for lbl, txt in lst:
            files.append((lbl, ids_of(txt), txt))
    print(f"  {len(files)} fixture files")
    shared_pairs = 0
    examples = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            inter = files[i][1] & files[j][1]
            if inter:
                shared_pairs += 1
                if len(examples) < 8:
                    examples.append((files[i][0], files[j][0], len(inter),
                                     sorted(inter)[0]))
    print(f"  file pairs sharing >=1 identifier: {shared_pairs} "
          f"of {len(files)*(len(files)-1)//2}")
    for e in examples:
        print(f"     {e[0]}  &  {e[1]}  share {e[2]}  e.g. {e[3][:13]}..")
    # substring nesting: is one chunk contained in another?
    nest = 0
    for i in range(len(files)):
        for j in range(len(files)):
            if i == j:
                continue
            a, b = files[i][2], files[j][2]
            if len(a) < len(b) and a[:400] and a[:400] in b:
                nest += 1
                if nest <= 6:
                    print(f"     NESTED PREFIX: {files[i][0]} head found inside {files[j][0]}")
    print(f"  nested-prefix relations: {nest}")


def t5_t6_slots():
    print("\n" + "=" * 78)
    print("T5/T6  slot hygiene recomputed from chunk_sha256 (ignore `cold`)")
    print("=" * 78)
    for name in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
                 "sweep.jsonl", "sweep-run1-shared-server.jsonl"):
        recs = [r for r in load(name) if r.get("status") in (None, "ok")]
        # slot -> ordered list of chunk shas
        per_slot = collections.defaultdict(list)
        for r in recs:
            s = r.get("slot_id")
            if s is None:
                continue
            per_slot[s].append(r.get("chunk_sha256"))
        docs_per_slot = collections.Counter(len(set(v) - {None}) for v in per_slot.values())
        # a call is "second-sight" if its slot already served a DIFFERENT doc
        second = 0
        seen = collections.defaultdict(set)
        for r in recs:
            s, c = r.get("slot_id"), r.get("chunk_sha256")
            if s is None:
                continue
            if seen[s] and c not in seen[s]:
                second += 1
            seen[s].add(c)
        # doc served on >1 slot
        doc_slots = collections.defaultdict(set)
        for r in recs:
            if r.get("chunk_sha256") and r.get("slot_id") is not None:
                doc_slots[r["chunk_sha256"]].add(r["slot_id"])
        multi = sum(1 for v in doc_slots.values() if len(v) > 1)
        mism = sum(1 for r in recs if r.get("requested_slot") is not None
                   and r.get("slot_id") is not None
                   and r["requested_slot"] != r["slot_id"])
        cached_gt0 = [r for r in recs if (r.get("tokens_cached") or 0) > 0]
        print(f"\n  {name}: {len(recs)} ok, slots={len(per_slot)}")
        print(f"     docs-per-slot histogram      : {dict(sorted(docs_per_slot.items()))}")
        print(f"     calls landing on a slot that already held ANOTHER doc: {second}")
        print(f"     documents served on >1 slot  : {multi}/{len(doc_slots)}")
        print(f"     requested_slot != slot_id    : {mism}")
        print(f"     tokens_cached>0 calls        : {len(cached_gt0)}/{len(recs)}")
        if cached_gt0:
            v = sorted(r["tokens_cached"] for r in cached_gt0)
            print(f"        min/med/max = {v[0]}/{v[len(v)//2]}/{v[-1]}")


def main():
    t1_archladder()
    chunks = load_chunks()
    t2_t3_t4(chunks)
    t4_overlap(chunks)
    t5_t6_slots()


if __name__ == "__main__":
    main()
