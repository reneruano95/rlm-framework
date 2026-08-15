"""Re-derive the auditor's identifier-origin counts under THEIR definition
(identifier = UUID or ENT-\\d{4,6}), and then measure how much discriminating
power that detector actually has.

Power question: an identifier only reads as a LEAK if it occurs in some other
fixture and NOT in the record's own chunk. If ENT ids collide across fixtures,
a leaked ENT id is silently reclassified as "own chunk". Quantify that.
"""
import collections
import hashlib
import json
import pathlib
import re

S2 = pathlib.Path(__file__).resolve().parents[1]
RES = S2 / "results"
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
ENT_RE = re.compile(r"\bENT-\d{4,6}\b")


def load_chunks():
    out = {}
    for p in sorted(S2.glob("fixtures*/**/*.chunk.txt")):
        text = p.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out[h] = (f"{p.parent.name}/{p.name}", text)
    return out


def load(name):
    return [json.loads(l) for l in (RES / name).open(encoding="utf-8") if l.strip()]


def audit(name, chunks):
    recs = [r for r in load(name) if r.get("status") in (None, "ok")]
    all_texts = [(lbl, txt) for lbl, txt in chunks.values()]
    c = collections.Counter()
    for r in recs:
        raw = r.get("raw_output") or ""
        ids = set(UUID_RE.findall(raw)) | set(ENT_RE.findall(raw))
        if not ids:
            c["no-identifier"] += 1
            continue
        own = chunks.get(r.get("chunk_sha256") or "")
        own_txt = own[1] if own else None
        for ident in ids:
            if own_txt is None:
                c["UNRESOLVED"] += 1
            elif ident.lower() in own_txt.lower():
                c["own"] += 1
                if UUID_RE.fullmatch(ident):
                    c["own-uuid"] += 1
                else:
                    c["own-ENT"] += 1
            elif any(ident.lower() in t.lower() and lbl != own[0]
                     for lbl, t in all_texts):
                c["foreign"] += 1
            else:
                c["fabricated"] += 1
    print(f"{name:34s} recs={len(recs):4d} own={c['own']:4d} "
          f"(uuid {c['own-uuid']}, ENT {c['own-ENT']})  foreign={c['foreign']} "
          f"fabricated={c['fabricated']} unresolved={c['UNRESOLVED']}")
    return c


def power(chunks):
    """How unique are the identifier tokens across the fixture corpus?"""
    ent_files = collections.defaultdict(set)
    uuid_files = collections.defaultdict(set)
    for lbl, txt in chunks.values():
        for e in set(ENT_RE.findall(txt)):
            ent_files[e].add(lbl)
        for u in set(UUID_RE.findall(txt)):
            uuid_files[u.lower()].add(lbl)
    ent_multi = {e: f for e, f in ent_files.items() if len(f) > 1}
    uuid_multi = {u: f for u, f in uuid_files.items() if len(f) > 1}
    print(f"\ndistinct ENT ids in corpus : {len(ent_files)}; "
          f"appearing in >1 fixture: {len(ent_multi)} "
          f"({100*len(ent_multi)/max(1,len(ent_files)):.1f}%)")
    print(f"distinct UUIDs in corpus   : {len(uuid_files)}; "
          f"appearing in >1 fixture: {len(uuid_multi)}")
    hist = collections.Counter(len(f) for f in ent_files.values())
    print(f"ENT id -> #fixtures histogram: {dict(sorted(hist.items()))}")
    # per-chunk: what fraction of an average chunk's ENT ids are shared?
    frac = []
    for lbl, txt in chunks.values():
        e = set(ENT_RE.findall(txt))
        if e:
            frac.append(sum(1 for x in e if len(ent_files[x]) > 1) / len(e))
    if frac:
        print(f"mean fraction of a chunk's ENT ids that also occur in another "
              f"fixture: {sum(frac)/len(frac):.3f} (n={len(frac)} chunks)")


def main():
    chunks = load_chunks()
    print(f"fixture chunks on disk: {len(chunks)}\n")
    for f in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
              "sweep.jsonl", "sweep-run1-shared-server.jsonl"):
        audit(f, chunks)
    power(chunks)


if __name__ == "__main__":
    main()
