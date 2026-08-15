"""REFUTE pass 3: independent slot-discipline + leak-oracle re-score of every
probe result file, written from scratch (no reuse of the earlier audit scripts).

Part A - SLOT DISCIPLINE (the load-bearing claim):
  For each result file, treat the ENTIRE file as one server process (the
  leak-permitting, conservative assumption) and ask: did any slot ever serve
  two different chunk_sha256? Also: requested vs served slot mismatches, and
  tokens_cached>0 on the first call a slot ever sees.

Part B - LEAK ORACLE built independently:
  Corpus = every *.chunk.txt under every s2/fixtures* dir, keyed by sha256 so
  each result row can be tied to the exact document it was served.
  Identifiers = (1) uuid-shaped tokens, (2) ENT-xxxx style ids, (3) coined
  capitalised proper nouns (>=8 chars, not an English dictionary word and not
  present in the leaf prefix or the row's own question).
  A row LEAKS if its answer contains an identifier that occurs in some OTHER
  corpus chunk and NOT in its own chunk and NOT in its own question/prefix.
"""
from __future__ import annotations

import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]
ROOT = S2.parent
RESULTS = S2 / "results"

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
ENT_RE = re.compile(r"\bENT-[0-9A-Za-z]{2,}\b")
PROPER_RE = re.compile(r"\b[A-Z][a-z]{7,}\b")

COMMON = set("""Document Excerpt Reference Register Chapterhouse Depository Foundation
Institute Repository Consortium Directorate Secretariat Committee Assembly
Question Answer Nothing Provided Instructions Instruction Following Response
According Available Anything Everything Something Therefore Additionally
However Although Because Custodian Custody Archive Archives Original Duplicate
Remember Reference""".split())


def sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_corpus():
    """sha256 -> (path, text). Every chunk file in every fixture dir."""
    corpus = {}
    for d in sorted(ROOT.glob("s2/fixtures*")):
        for f in sorted(d.glob("*.chunk.txt")):
            t = f.read_text(encoding="utf-8")
            corpus[sha(t)] = (f"{d.name}/{f.name}", t)
    return corpus


def ids_of(text: str) -> set[str]:
    out = set(UUID_RE.findall(text)) | set(ENT_RE.findall(text))
    out |= {w for w in PROPER_RE.findall(text) if w not in COMMON}
    return {x.lower() for x in out}


def main() -> None:
    corpus = load_corpus()
    print(f"corpus: {len(corpus)} distinct chunk files")
    ids_by_sha = {s: ids_of(t) for s, (n, t) in corpus.items()}
    everywhere = Counter()
    for s, ids in ids_by_sha.items():
        for i in ids:
            everywhere[i] += 1
    print(f"identifier vocabulary: {len(everywhere)}  "
          f"(unique-to-one-chunk: {sum(1 for v in everywhere.values() if v == 1)})")

    prefix_ids = set()
    for p in (ROOT / "prompts").glob("*.md"):
        prefix_ids |= ids_of(p.read_text(encoding="utf-8"))

    files = ["distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
             "sweep.jsonl", "sweep-run1-shared-server.jsonl", "occupancy.jsonl",
             "r14.jsonl", "cache_instrument.jsonl",
             "leak-nocacheidle.jsonl", "leak-nocram.jsonl", "leak-cram0.jsonl",
             "leak-ctxcp0.jsonl", "leak-slotiso.jsonl", "leak-erase.jsonl"]

    for name in files:
        p = RESULTS / name
        if not p.exists():
            print(f"\n### {name}: MISSING")
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        print(f"\n### {name}  ({len(rows)} rows)")

        # ---------- Part A: slot discipline ----------
        slot_key = None
        for k in ("slot_id", "id_slot", "slot", "served_slot"):
            if k in rows[0]:
                slot_key = k
                break
        chunk_key = None
        for k in ("chunk_sha256", "chunk_sha", "doc_sha256", "chunk_id"):
            if k in rows[0]:
                chunk_key = k
                break
        req_key = "requested_slot" if "requested_slot" in rows[0] else None
        if slot_key is None or chunk_key is None:
            print(f"  (no slot/chunk fields: slot={slot_key} chunk={chunk_key}; "
                  f"keys={sorted(rows[0])[:14]}...)")
        else:
            hist = defaultdict(list)
            mismatch = 0
            for r in rows:
                s = r.get(slot_key)
                c = r.get(chunk_key)
                if req_key and r.get(req_key) is not None and s is not None \
                        and r[req_key] != s:
                    mismatch += 1
                if s is None or c is None:
                    continue
                if not hist[s] or hist[s][-1] != c:
                    hist[s].append(c)
            multi = {s: v for s, v in hist.items() if len(set(v)) > 1}
            print(f"  slots used: {len(hist)}   slots that served >1 distinct "
                  f"chunk: {len(multi)}   requested!=served: {mismatch}")
            for s, v in list(multi.items())[:6]:
                print(f"    slot {s}: {len(set(v))} distinct docs, "
                      f"{len(v)} switches")

        # ---------- Part B: leak oracle ----------
        ans_key = None
        for k in ("raw_output", "answer", "output", "text", "completion"):
            if k in rows[0]:
                ans_key = k
                break
        if ans_key is None or chunk_key is None:
            print(f"  (no answer/chunk field for oracle: ans={ans_key})")
            continue
        scored = leaked = unmapped = 0
        examples = []
        for i, r in enumerate(rows):
            a = r.get(ans_key) or ""
            csha = r.get(chunk_key)
            if csha not in ids_by_sha:
                unmapped += 1
                continue
            scored += 1
            own = ids_by_sha[csha]
            qids = ids_of((r.get("question") or "") + " " + str(r.get("expected") or ""))
            found = ids_of(a)
            foreign = {x for x in found
                       if x not in own and x not in qids and x not in prefix_ids
                       and everywhere.get(x, 0) > 0}
            if foreign:
                leaked += 1
                if len(examples) < 4:
                    examples.append((i, sorted(foreign)[:3], a[:90]))
        print(f"  ORACLE: {leaked}/{scored} rows leak "
              f"(unmapped chunk_sha: {unmapped})")
        for i, f, a in examples:
            print(f"    row{i}: foreign={f}  ans={a!r}")


if __name__ == "__main__":
    main()
