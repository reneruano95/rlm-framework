"""Independent re-derivation of the blast-radius auditor's numbers.

Step 1: build a sha256 index of every fixture chunk file on disk, so that a
record's chunk_sha256 can be mapped back to a file, and so that "does this
identifier occur in some OTHER fixture" can be answered exhaustively.

stdlib only. No network. No GPU.
"""
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
S2 = ROOT / "milestones" / "s2"

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def build_chunk_index():
    """sha256 -> [paths]; path -> text."""
    by_sha = {}
    texts = {}
    for p in sorted(S2.glob("fixtures*/**/*.chunk.txt")):
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        rel = str(p.relative_to(S2)).replace("\\", "/")
        by_sha.setdefault(sha, []).append(rel)
        texts[rel] = raw.decode("utf-8", errors="replace")
    return by_sha, texts


def main():
    by_sha, texts = build_chunk_index()
    print(f"fixture chunk files on disk: {len(texts)}")
    print(f"distinct sha256:             {len(by_sha)}")
    dupes = {s: v for s, v in by_sha.items() if len(v) > 1}
    print(f"sha collisions (same bytes in >1 file): {len(dupes)}")
    for s, v in list(dupes.items())[:10]:
        print(f"   {s[:12]} -> {v}")

    # how many distinct UUIDs live in the corpus, and in how many files each
    uuid_files = {}
    for rel, t in texts.items():
        for u in set(UUID_RE.findall(t)):
            uuid_files.setdefault(u.lower(), set()).add(rel)
    print(f"distinct UUIDs across all fixture chunks: {len(uuid_files)}")
    shared = {u: f for u, f in uuid_files.items() if len(f) > 1}
    print(f"UUIDs appearing in >1 chunk FILE: {len(shared)}")
    # collapse by sha (same bytes copied to 2 dirs is not a real cross-fixture)
    rel_to_sha = {}
    for s, v in by_sha.items():
        for rel in v:
            rel_to_sha[rel] = s
    shared_sha = {u: {rel_to_sha[r] for r in f} for u, f in uuid_files.items()}
    shared_sha = {u: s for u, s in shared_sha.items() if len(s) > 1}
    print(f"UUIDs appearing in >1 distinct chunk SHA: {len(shared_sha)}")
    for u, s in list(shared_sha.items())[:10]:
        print(f"   {u} in {len(s)} shas: {sorted(by_sha[x][0] for x in s)}")

    out = S2 / "audit" / "_chunk_index.json"
    out.write_text(
        json.dumps(
            {
                "by_sha": by_sha,
                "uuid_files": {u: sorted(f) for u, f in uuid_files.items()},
            },
            indent=0,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
