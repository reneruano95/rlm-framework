"""OFFLINE audit of the S2 probe code's own defects. No GPU, no HTTP.

Three questions, all answerable from files that already exist:

  1. SLOT DISCIPLINE per probe. Which probes pinned a never-reused slot per
     document and which shared one slot across documents (R13's reproducing
     condition)?
  2. LEAK, re-checked against the WHOLE fixture corpus rather than the subset
     each run happened to load. `rlm.leakcheck.ChunkIndex` only knows the
     chunks handed to it, so an identifier from a fixture directory the run
     did not load reads as CLEAN.
  3. CLASSIFIER. How many FALSE-POSITIVE labels are prose refusals the pinned
     `is_refusal` phrase list misses?
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]
REPO = S2.parent
sys.path.insert(0, str(REPO))

from rlm.leakcheck import identifier_tokens  # noqa: E402
from s2.run_sweep import classify, is_refusal, normalize  # noqa: E402

# --------------------------------------------------------------------------- #
# the WHOLE fixture corpus
# --------------------------------------------------------------------------- #

CHUNK_BY_SHA: dict[str, tuple[str, str]] = {}      # sha256 -> (dir, cell_id)
CHUNK_TEXT_BY_SHA: dict[str, str] = {}
TOKEN_OWNERS: dict[str, set[str]] = defaultdict(set)  # ident -> {dir/cell}
EXPECTED_OWNERS: dict[str, set[str]] = defaultdict(set)  # expected val -> cells


def load_corpus() -> None:
    for d in sorted(S2.glob("fixtures*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        for cell in m["cells"].values():
            p = Path(cell["chunk_path"])
            if not p.exists():
                p = d / Path(cell["chunk_path"]).name
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            key = f"{d.name}/{cell['cell_id']}"
            CHUNK_BY_SHA[sha] = (d.name, cell["cell_id"])
            CHUNK_TEXT_BY_SHA[sha] = text
            for t in identifier_tokens(text):
                TOKEN_OWNERS[t.lower()].add(key)
            for q in cell.get("questions", {}).values():
                if q.get("expected"):
                    EXPECTED_OWNERS[str(q["expected"]).lower()].add(key)


# --------------------------------------------------------------------------- #
# a BROADENED refusal detector, used only to measure the pinned one's gap
# --------------------------------------------------------------------------- #

BROAD_REFUSAL = re.compile(
    r"(does not (contain|mention|state|include|provide|list|specify|record|appear)"
    r"|do not (contain|mention|state|include|provide|list|specify)"
    r"|is not (present|stated|found|given|mentioned|specified|listed|recorded|"
    r"provided|included|in the (excerpt|text|document|passage))"
    r"|not (present|stated|found|given|mentioned|specified|listed|recorded|provided)"
    r"|no (record|key|mention|entry|custody|archive key|such)"
    r"|there is no|cannot be determined|could not find|couldn'?t find"
    r"|unable to (find|locate|determine)|i (can|could)\s*n[o']?t find"
    r"|^none\b)", re.I)


def summarize(path: Path, label: str) -> None:
    if not path.exists():
        print(f"\n### {label}: MISSING {path}")
        return
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    ok = [r for r in recs if r.get("status") == "ok" and r.get("question_type")]
    print(f"\n### {label}  ({path.name}: {len(recs)} rows, {len(ok)} scored)")

    # ---- slot discipline ---- #
    by_slot: dict[object, set[str]] = defaultdict(set)
    for r in ok:
        s = r.get("requested_slot", r.get("id_slot"))
        by_slot[s].add(r.get("chunk_sha256", "?"))
    multi = {s: len(c) for s, c in by_slot.items() if len(c) > 1}
    print(f"  slots used: {len(by_slot)};  slots that held >1 distinct document: "
          f"{len(multi)}"
          + (f"  -> {dict(list(multi.items())[:6])}" if multi else ""))

    # ---- labels + leak recheck ---- #
    lab = Counter()
    fp_ident = fp_foreign = fp_invented = fp_noident = 0
    fp_prose_refusal = 0
    foreign_examples: list[str] = []
    leak_field = Counter()
    for r in ok:
        raw = r.get("raw_output", "")
        try:
            c = classify(raw, question_type=r["question_type"],
                         expected=r.get("expected"),
                         expected_kind=r.get("expected_kind"))
        except Exception:
            continue
        lab[c["label"]] += 1
        leak_field[r.get("leak_detected")] += 1
        if c["label"] != "FALSE-POSITIVE":
            continue
        sha = r.get("chunk_sha256")
        sent = (CHUNK_TEXT_BY_SHA.get(sha, "") + "\n\n" + str(r.get("question", "")))
        own = {t.lower() for t in identifier_tokens(sent)}
        ans_tokens = identifier_tokens(raw)
        if not ans_tokens:
            fp_noident += 1
        else:
            fp_ident += 1
        foreign = [t for t in ans_tokens
                   if t.lower() not in own and t.lower() in TOKEN_OWNERS]
        if foreign:
            fp_foreign += 1
            if len(foreign_examples) < 8:
                foreign_examples.append(
                    f"{r.get('cell_uid', r.get('cell_id'))} q={r['question_type']} "
                    f"-> {foreign[0]} @ {sorted(TOKEN_OWNERS[foreign[0].lower()])[:2]}"
                    + ("  [is a TRUE expected answer elsewhere]"
                       if foreign[0].lower() in EXPECTED_OWNERS else ""))
        elif ans_tokens - {t for t in ans_tokens if t.lower() in own}:
            fp_invented += 1
        if BROAD_REFUSAL.search(normalize(raw)):
            fp_prose_refusal += 1
    print("  labels:", dict(lab))
    print(f"  leak_detected field as recorded: {dict(leak_field)}")
    n_fp = lab['FALSE-POSITIVE']
    if n_fp:
        print(f"  FALSE-POSITIVE = {n_fp}")
        print(f"    ...carrying any identifier-shaped token : {fp_ident}")
        print(f"    ...carrying NO identifier at all        : {fp_noident}")
        print(f"    ...carrying a token FOREIGN to its own doc but present "
              f"elsewhere in the WHOLE fixture corpus: {fp_foreign}")
        print(f"    ...carrying an identifier present in NO fixture (invented): "
              f"{fp_invented}")
        print(f"    ...whose text is a PROSE REFUSAL the pinned classifier "
              f"missed: {fp_prose_refusal}")
        for e in foreign_examples:
            print(f"      leak example: {e}")

    # ---- MISS / recall sanity ---- #
    miss = lab["MISS"]
    if miss:
        print(f"  MISS (refused a present fact) = {miss}")


def main() -> None:
    load_corpus()
    print(f"corpus: {len(CHUNK_BY_SHA)} chunk files, "
          f"{len(TOKEN_OWNERS)} distinct identifier tokens, "
          f"{len(EXPECTED_OWNERS)} distinct expected answers")
    R = S2 / "results"
    summarize(R / "sweep.jsonl", "run_sweep (RESULTS.md: the 95% FP + size cliff)")
    summarize(R / "refusal-ab.jsonl", "run_refusal_ab 1024 (REFUSAL-AB.md)")
    summarize(R / "refusal-ab-640.jsonl", "run_refusal_ab 640 (REFUSAL-AB-640.md)")
    summarize(R / "distance.jsonl", "run_distance (DISTANCE.md: instruction decay)")


if __name__ == "__main__":
    main()
