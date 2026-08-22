"""Definitive offline re-score of every R13 artifact on disk.

Two oracles are applied to every recorded answer:

  LOOSE  = milestones/s2/r13_repro.py's `foreign_strings` verbatim: an id-shaped or
           CamelCase token that is absent from the chunk sent and present in
           some OTHER fixture chunk.
  STRICT = LOOSE, plus (a) the token must also be absent from the QUESTION
           (R13.md §2 claims this exclusion; the shipped code does not do it),
           and (b) the token must occur in exactly ONE other chunk.

Offline only. Reads files already on disk; starts nothing.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:/PROJECTS/rlm-halo-framework")
FIX = ROOT / "milestones" / "s2" / "fixtures"
RES = ROOT / "milestones" / "s2" / "results"

_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|ENT-\d{4,6}", re.I)
_PROPER = re.compile(r"\b[A-Z][a-z]{6,}\b")

TEXTS = {
    p.name.replace(".chunk.txt", ""): p.read_text(encoding="utf-8")
    for p in sorted(FIX.glob("*.chunk.txt"))
}
MANIFEST = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))


def hits(answer: str, own_cell: str, question: str = "", strict: bool = False):
    own = TEXTS[own_cell]
    out = []
    for tok in set(_ID.findall(answer)) | set(_PROPER.findall(answer)):
        if tok.lower() in own.lower():
            continue
        if strict and question and tok.lower() in question.lower():
            continue
        where = [c for c, t in TEXTS.items() if c != own_cell and tok.lower() in t.lower()]
        if not where:
            continue
        if strict and len(where) != 1:
            continue
        out.append((tok, where[0] if len(where) == 1 else f"{len(where)} cells"))
    return out


def id_only(hs):
    return [h for h in hs if _ID.fullmatch(h[0])]


# ---------------------------------------------------------------- leak-* arms
ARM_OF_FILE = {
    # file -> (server stderr log, launch flags inferred from that log's filename)
    "leak-nocacheidle.jsonl": ("traces/logs/leaf-server-ub512nocacheidleslots.log.err",
                               "--no-cache-idle-slots"),
    "leak-nocram.jsonl": ("traces/logs/leaf-server-ub512cacheram0ctxcp0.log.err",
                          "--cache-ram 0 -ctxcp 0"),
    "leak-cram0.jsonl": ("traces/logs/leaf-server-ub512cacheram0.log.err",
                         "--cache-ram 0"),
    "leak-ctxcp0.jsonl": ("traces/logs/leaf-server-ub512ctxcp0.log.err",
                          "-ctxcp 0"),
    "leak-slotiso.jsonl": ("traces/logs/leaf-server-ub512.log.err",
                           "S0 defaults; client used slot 1 for the 2nd cell"),
    "leak-erase.jsonl": ("traces/logs/leaf-server-ub512.log.err",
                         "S0 defaults; SAME PROCESS as leak-slotiso"),
}
# the two files that shared one server process, in wire order
PROCESS_GROUPS = [
    ["leak-nocacheidle.jsonl"],
    ["leak-nocram.jsonl"],
    ["leak-cram0.jsonl"],
    ["leak-ctxcp0.jsonl"],
    ["leak-slotiso.jsonl", "leak-erase.jsonl"],
]


def leak_arms() -> None:
    print("#" * 96)
    print("# PART 1 -- the leak-*.jsonl server-flag arms (2026-08-13 19:33-19:41 UTC)")
    print("#" * 96)
    for group in PROCESS_GROUPS:
        rows = []
        for fn in group:
            for l in (RES / fn).read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    r["_file"] = fn
                    rows.append(r)
        rows.sort(key=lambda r: r["ts"])
        log, flags = ARM_OF_FILE[group[0]]
        print("=" * 96)
        print(f"process: {log}")
        print(f"  flags (from log filename): {flags}")
        print(f"  files: {group}   calls: {len(rows)}")
        seen: dict[int, list[str]] = defaultdict(list)
        tally = defaultdict(lambda: [0, 0, 0])  # cell -> [loose, strict, n]
        exposed = [0, 0, 0]  # loose, strict, n  over calls whose slot held an earlier cell
        for r in rows:
            cell, slot = r["cell_id"], r["id_slot"]
            prior = list(seen[slot])
            lo = hits(r["raw_output"] or "", cell)
            st = hits(r["raw_output"] or "", cell, r["question"], strict=True)
            st_prior = [h for h in st if h[1] in prior]
            tally[cell][2] += 1
            if lo:
                tally[cell][0] += 1
            if st:
                tally[cell][1] += 1
            if prior:
                exposed[2] += 1
                if lo:
                    exposed[0] += 1
                if st_prior:
                    exposed[1] += 1
            if cell not in seen[slot]:
                seen[slot].append(cell)
        print("  per cell  cell=loose/strict/n : " + ", ".join(
            f"{c}={v[0]}/{v[1]}/{v[2]}" for c, v in tally.items()))
        print(f"  EXPOSED CALLS (slot already held a different cell): n={exposed[2]}  "
              f"loose={exposed[0]}  STRICT={exposed[1]}  "
              f"strict rate={exposed[1]/exposed[2]:.1%}" if exposed[2] else "  no exposed calls")
        # what the answers leaked
        c = Counter()
        for r in rows:
            for tok, src in hits(r["raw_output"] or "", r["cell_id"], r["question"], strict=True):
                c[(tok, src)] += 1
        for (tok, src), n in c.most_common(8):
            print(f"      x{n:<3d} {tok:<40s} lives only in {src}")


# ------------------------------------------------------- r13_repro replay runs
def replay_runs() -> None:
    print()
    print("#" * 96)
    print("# PART 2 -- milestones/s2/r13_repro.py replay runs (the R13.md headline table)")
    print("#" * 96)
    files = ["r13_replay_hybrid.jsonl", "r13_replay_erase.jsonl",
             "r13_replay_gemma_fullattn.jsonl", "r13_replay_paired.jsonl",
             "r13_twoprompt_matrix.jsonl", "r13_twoprompt_sizesweep.jsonl"]
    for fn in files:
        p = RES / fn
        if not p.exists():
            print(f"MISSING {fn}")
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        print("=" * 96)
        conds = Counter(r["condition"] for r in rows)
        print(f"{fn}: n={len(rows)} conditions={dict(conds)}")
        by_arm = defaultdict(lambda: [0, 0, 0, 0])  # loose, strict, n, cold_leaks
        cachen0 = defaultdict(lambda: [0, 0])
        for r in rows:
            lab = r.get("label", "")
            if ":" not in lab:
                arm = r.get("label_tag") or r["condition"]
                by_arm[arm][2] += 1
                continue
            cell = lab.split(":")[0]
            if cell not in TEXTS:
                continue
            tag = r.get("label_tag") or ""
            arm = f"{r['condition']}/{tag or 'slot' + str(r['slot_id'])}"
            # the replay records carry the asked-about entity, not the full
            # question; excluding it removes question-echo false positives
            # (e.g. "Guildhall" from "the Prylirmeholm Guildhall").
            asked = r.get("asked_about") or ""
            lo = hits(r["raw_output"] or "", cell)
            st = hits(r["raw_output"] or "", cell, asked, strict=True)
            by_arm[arm][2] += 1
            if lo:
                by_arm[arm][0] += 1
            if st:
                by_arm[arm][1] += 1
            if r.get("cache_n") == 0:
                cachen0[arm][1] += 1
                if st:
                    cachen0[arm][0] += 1
                    by_arm[arm][3] += 1
        for arm in sorted(by_arm):
            lo, st, n, cold = by_arm[arm]
            c0 = cachen0[arm]
            print(f"   {arm:<34s} n={n:<4d} loose={lo:<3d} STRICT={st:<3d} "
                  f"({st/n:.0%} of n)   of the cache_n==0 calls: {c0[0]}/{c0[1]} leaked")


# ------------------------------------------------------------- the real sweeps
def sweeps() -> None:
    print()
    print("#" * 96)
    print("# PART 3 -- the production sweeps: is sweep.jsonl really clean?")
    print("#" * 96)
    for fn in ("sweep.jsonl", "sweep-run1-shared-server.jsonl", "diag.jsonl"):
        p = RES / fn
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r.get("cell_id") in TEXTS and r.get("raw_output")]
        lo = st = ido = 0
        detail = []
        for r in rows:
            h = hits(r["raw_output"], r["cell_id"])
            s = hits(r["raw_output"], r["cell_id"], r["question"], strict=True)
            if h:
                lo += 1
            if s:
                st += 1
            if id_only(s):
                ido += 1
                detail.append((r["cell_id"], r["question_type"], r.get("trial"),
                               id_only(s), r["raw_output"][:90].replace("\n", " ")))
        print("=" * 96)
        print(f"{fn}: n={len(rows)}  loose={lo}  strict={st}  strict-and-IDENTIFIER-shaped={ido}")
        for d in detail[:14]:
            print(f"    {d[0]:14s} {d[1]:10s} t{d[2]} {d[3]} :: {d[4]}")


if __name__ == "__main__":
    leak_arms()
    replay_runs()
    sweeps()
