"""Independent cross-fixture attribution test for the LOAD-BEARING s2 runs.

The arch-ladder probe (v0.3.3) answered ABSENT questions with UUIDs that are the
TRUE keys of the same entity in a DIFFERENT fixture -- i.e. cross-request
retrieval. This script asks whether the SAME signature exists in the runs the
spec's headline claims actually rest on (distance / refusal-ab / sweep), without
trusting those runs' own `leak_detected` flag.

Method, per record:
  * pull every identifier-shaped token out of `raw_output`
  * classify it: present in the record's OWN chunk / present in some OTHER
    fixture chunk on disk / present nowhere
Own-chunk membership is resolved by hashing every fixture .chunk.txt on disk and
matching `chunk_sha256`; where that fails the fallback is cell_id -> filename.

Offline, stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import re
import collections
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\s2")
RES = S2 / "results"

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
ENT_RE = re.compile(r"\bENT-\d{4,6}\b")
KEYLIKE_RE = re.compile(r"\b[A-Z]{2,6}-[0-9A-Za-z]{4,}\b")


def load_chunks() -> dict[str, tuple[str, str]]:
    """sha256 -> (label, text) for every fixture chunk on disk."""
    out: dict[str, tuple[str, str]] = {}
    for p in sorted(S2.glob("fixtures*/**/*.chunk.txt")):
        text = p.read_text(encoding="utf-8", errors="replace")
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out[h] = (f"{p.parent.name}/{p.name}", text)
    return out


def load(name):
    recs = []
    with (RES / name).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def audit(name: str, chunks: dict[str, tuple[str, str]]):
    recs = [r for r in load(name) if r.get("status") in (None, "ok")]
    matched = sum(1 for r in recs if r.get("chunk_sha256") in chunks)
    print(f"\n=== {name}: {len(recs)} ok records; own-chunk resolved by sha256 "
          f"for {matched}")

    counts = collections.Counter()
    foreign_examples = []
    all_texts = [(lbl, txt) for lbl, txt in chunks.values()]

    for r in recs:
        raw = r.get("raw_output") or ""
        ids = set(UUID_RE.findall(raw)) | set(ENT_RE.findall(raw))
        if not ids:
            counts["no-identifier-emitted"] += 1
            continue
        own = chunks.get(r.get("chunk_sha256") or "")
        own_txt = own[1] if own else None
        for ident in ids:
            in_own = (own_txt is not None) and (ident.lower() in own_txt.lower())
            elsewhere = [lbl for lbl, txt in all_texts
                         if ident.lower() in txt.lower()
                         and (own is None or lbl != own[0])]
            if own_txt is None:
                counts["own-chunk-UNRESOLVED"] += 1
            elif in_own:
                counts["identifier in OWN chunk"] += 1
            elif elsewhere:
                counts["identifier ONLY in ANOTHER fixture (LEAK SIGNATURE)"] += 1
                if len(foreign_examples) < 12:
                    foreign_examples.append({
                        "cell": r.get("cell_uid") or r.get("cell_id"),
                        "arm": r.get("arm"), "qtype": r.get("question_type"),
                        "label": r.get("label"), "emitted": ident,
                        "found_in": elsewhere[:3],
                        "own": own[0], "slot": r.get("slot_id"),
                        "tokens_cached": r.get("tokens_cached")})
            else:
                counts["identifier in NO fixture on disk (fabricated)"] += 1

    for k, v in counts.most_common():
        print(f"   {k:52s} {v}")
    for ex in foreign_examples:
        print("   LEAK-SHAPED:", ex)


def main():
    chunks = load_chunks()
    print(f"fixture chunks on disk: {len(chunks)}")
    for lbl, _ in sorted(chunks.values()):
        print("   ", lbl)
    for f in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
              "sweep.jsonl", "sweep-run1-shared-server.jsonl"):
        audit(f, chunks)


if __name__ == "__main__":
    main()
