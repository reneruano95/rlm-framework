"""Independent re-derivation of every count in the blast-radius report.

For each load-bearing result file:
  - row counts, ok counts
  - slot hygiene: distinct slots, distinct chunk_sha256 per slot, requested vs
    returned slot mismatch, tokens_cached/cache_n on the first call seen on
    each slot
  - identifier origin: every UUID-shaped token emitted in the model output,
    classified as OWN-CHUNK / FOREIGN-FIXTURE / FABRICATED

stdlib only. No network. No GPU.
"""
import collections
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
S2 = ROOT / "s2"
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def load_corpus():
    by_sha = {}
    texts = {}
    for p in sorted(S2.glob("fixtures*/**/*.chunk.txt")):
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        rel = str(p.relative_to(S2)).replace("\\", "/")
        by_sha[sha] = rel
        texts[rel] = raw.decode("utf-8", errors="replace")
    uuid_of = {}          # rel -> set(uuid)
    all_uuid = {}         # uuid -> set(rel)
    for rel, t in texts.items():
        s = {u.lower() for u in UUID_RE.findall(t)}
        uuid_of[rel] = s
        for u in s:
            all_uuid.setdefault(u, set()).add(rel)
    return by_sha, uuid_of, all_uuid


BY_SHA, UUID_OF, ALL_UUID = load_corpus()


def rows(name):
    p = S2 / "results" / name
    out = []
    for i, line in enumerate(p.open(encoding="utf-8"), 1):
        line = line.strip()
        if line:
            r = json.loads(line)
            r["_ln"] = i
            out.append(r)
    return out


def out_text(r):
    for k in ("raw_output", "answer", "reduced_text"):
        if r.get(k):
            return r[k]
    return ""


def analyse(name, slot_req_key, slot_ret_key, cache_key, ok_pred=None):
    rs = rows(name)
    ok = [r for r in rs if (ok_pred(r) if ok_pred else r.get("status") == "ok")]
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    print(f"rows={len(rs)}  ok={len(ok)}  not-ok={len(rs)-len(ok)}")

    # --- slot hygiene -------------------------------------------------
    per_slot_docs = collections.defaultdict(set)
    mismatch = 0
    first_seen = {}
    for r in ok:
        sid = r.get(slot_ret_key)
        req = r.get(slot_req_key)
        if req is not None and sid is not None and req != sid:
            mismatch += 1
        sha = r.get("chunk_sha256") or r.get("doc")
        if sid is not None:
            per_slot_docs[sid].add(sha)
            if sid not in first_seen:
                first_seen[sid] = r
    hist = collections.Counter(len(v) for v in per_slot_docs.values())
    print(f"distinct slots with ok calls: {len(per_slot_docs)}")
    print(f"docs-per-slot histogram (ok calls only): {dict(sorted(hist.items()))}")
    print(f"requested != returned slot: {mismatch}")
    nz = [(s, r.get(cache_key)) for s, r in first_seen.items() if r.get(cache_key)]
    print(f"first call on a slot with {cache_key} > 0: {len(nz)}  {nz[:8]}")
    # slots that appear anywhere in the file (incl. non-ok requests)
    allslots = {r.get(slot_req_key) for r in rs if r.get(slot_req_key) is not None}
    print(f"distinct requested slots over ALL rows (incl. errors): {len(allslots)}")

    # --- identifier origin --------------------------------------------
    own = foreign = fab = 0
    foreign_detail = []
    fab_detail = []
    calls_with_id = 0
    for r in ok:
        ids = {u.lower() for u in UUID_RE.findall(out_text(r))}
        if ids:
            calls_with_id += 1
        sha = r.get("chunk_sha256")
        rel = BY_SHA.get(sha)
        ownset = UUID_OF.get(rel, set()) if rel else None
        for u in ids:
            if ownset is not None and u in ownset:
                own += 1
            elif u in ALL_UUID:
                foreign += 1
                foreign_detail.append(
                    (r["_ln"], r.get("cell_id") or r.get("cell_uid"), u,
                     sorted(ALL_UUID[u]))
                )
            else:
                fab += 1
                fab_detail.append((r["_ln"], r.get("cell_id") or r.get("cell_uid"), u))
    print(f"identifiers emitted: own={own} foreign={foreign} fabricated={fab} "
          f"(total {own+foreign+fab}; in {calls_with_id} calls)")
    unmapped = {r.get("chunk_sha256") for r in ok if r.get("chunk_sha256") not in BY_SHA}
    print(f"chunk_sha256 values NOT found among on-disk fixtures: {len(unmapped)}")
    for f in foreign_detail[:20]:
        print(f"   FOREIGN ln{f[0]} cell={f[1]} {f[2]} -> {f[3]}")
    for f in fab_detail[:20]:
        print(f"   FABRICATED ln{f[0]} cell={f[1]} {f[2]}")
    return rs, ok


if __name__ == "__main__":
    analyse("distance.jsonl", "requested_slot", "slot_id", "tokens_cached")
    analyse("refusal-ab.jsonl", "requested_slot", "slot_id", "tokens_cached")
    analyse("refusal-ab-640.jsonl", "requested_slot", "slot_id", "tokens_cached")
    analyse("sweep.jsonl", "requested_slot", "slot_id", "tokens_cached")
    analyse("sweep-run1-shared-server.jsonl", "requested_slot", "slot_id",
            "tokens_cached")
