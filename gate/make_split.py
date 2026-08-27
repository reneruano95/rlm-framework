"""Freeze the S6-lite v0 train/held-out split (spec §6, plan step 2.1).

WHY THIS IS A SCRIPT AND NOT A HAND-WRITTEN FILE. The split's whole claim is that
each side is a sample of the SAME question shape. That property is checkable --
`question_sha256` in `bench/manifest.json` is the shape -- so it is checked here
and the check fails the build rather than living in prose.

THE TRAP IT AVOIDS (spec §6, measured from the manifest). `regex_solvable` is
perfectly confounded with the question text: agg-01/02/03 share one
`question_sha256`, agg-04..07 share another. Splitting aggregation the natural way
would train on one question shape and evaluate on a different one -- a different
distribution, not a held-out sample. So v0 splits WITHIN a shape and records what
it excluded, and why, in the artifact itself.

Run: python gate/make_split.py [--out bench/splits/s6lite-v0.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Fixed here, before any run. Spec §6.
TRAIN = ["codeqa-01", "codeqa-02", "codeqa-03", "codeqa-06",
         "needle-01", "needle-02", "needle-03", "needle-04",
         "agg-04", "agg-05"]
HELD_OUT = ["codeqa-04", "codeqa-05", "codeqa-07",
            "needle-05", "needle-06", "needle-07", "needle-08",
            "agg-06", "agg-07"]
EXCLUDED = {
    "agg-01": "different question shape from agg-04..07 (regex_solvable group); also 1 of only 2 adversarial tasks",
    "agg-02": "different question shape from agg-04..07 (regex_solvable group)",
    "agg-03": "different question shape from agg-04..07 (regex_solvable group)",
    **{f"synth-0{i}": "synthesis held for a later round so v0's held-out is not the whole benchmark"
       for i in range(1, 9)},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "bench" / "splits" / "s6lite-v0.json"))
    args = ap.parse_args()

    manifest_path = REPO / "bench" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    missing = [t for t in TRAIN + HELD_OUT + list(EXCLUDED) if t not in tasks]
    if missing:
        print(f"ERROR: not in manifest: {missing}", file=sys.stderr)
        return 2

    covered = set(TRAIN) | set(HELD_OUT) | set(EXCLUDED)
    if covered != set(tasks):
        print(f"ERROR: split does not account for every v1 task; missing {sorted(set(tasks) - covered)}",
              file=sys.stderr)
        return 2
    if set(TRAIN) & set(HELD_OUT):
        print(f"ERROR: task on both sides: {sorted(set(TRAIN) & set(HELD_OUT))}", file=sys.stderr)
        return 2

    def row(tid: str) -> dict:
        t = tasks[tid]
        return {"task_id": tid, "category": t["category"], "checker": t["checker"],
                "question_sha256": t["question_sha256"], "corpus_sha256": t["corpus_sha256"],
                "corpus_tokens": t["corpus_tokens"], "adversarial": t.get("adversarial", False)}

    train = [row(t) for t in TRAIN]
    held = [row(t) for t in HELD_OUT]

    # THE CHECK THIS SCRIPT EXISTS FOR, stated precisely.
    #
    # `question_sha256` hashes the whole question text, so it separates task SHAPES
    # only where several tasks share one. Measured from the manifest 2026-08-27:
    #
    #   aggregation  2 hashes, groups of 4 and 3   -> the hash IS the shape
    #   synthesis    3 hashes, groups of 3, 3, 2   -> the hash IS the shape
    #   code QA      7 hashes, all singletons      -> the hash is the target symbol
    #   needle       8 hashes, all singletons      -> the hash is the target org
    #
    # So a CLUSTER (>= 2 tasks on one hash) is evidence of a distinct shape, and a
    # split that straddles clusters within a category trains on one shape and
    # evaluates on another -- the v1 trap. Singletons carry no shape information and
    # this check must not pretend otherwise: for those categories the same-shape
    # claim rests on the shared template (and, for code QA, on a literally identical
    # corpus -- all 7 share one corpus_sha256), which is asserted separately below.
    cats: dict[str, dict[str, list[str]]] = {}
    for tid, t in tasks.items():
        cats.setdefault(t["category"], {}).setdefault(t["question_sha256"], []).append(tid)

    straddled = []
    for cat, groups in cats.items():
        clusters = {sha: ids for sha, ids in groups.items() if len(ids) > 1}
        if len(clusters) < 2:
            continue
        tr_c = {sha for sha, ids in clusters.items() if set(ids) & set(TRAIN)}
        ho_c = {sha for sha, ids in clusters.items() if set(ids) & set(HELD_OUT)}
        if tr_c and ho_c and tr_c != ho_c:
            straddled.append((cat, sorted(tr_c), sorted(ho_c)))
    if straddled:
        print("ERROR: split straddles question-shape clusters (this is the v1 trap):", file=sys.stderr)
        for cat, tr_c, ho_c in straddled:
            print(f"  {cat}: train shapes {[s[:12] for s in tr_c]} vs held-out {[s[:12] for s in ho_c]}",
                  file=sys.stderr)
        return 2

    # ANSWER DISJOINTNESS. Found the hard way 2026-08-27, on the screens' first run:
    # the original draw put codeqa-06 (`rlm/dispatcher.py`) and codeqa-07
    # (`rlm/budget.py`) on the held-out side while codeqa-01/03 and codeqa-04 -- which
    # have the SAME answers -- sat in train. A model that memorised a train answer
    # would then score the held-out task without reading anything, which is the exact
    # contamination the gate exists to prevent. Answers must partition: every task
    # sharing an answer belongs to one side.
    by_answer: dict[str, set[str]] = {}
    for tid in TRAIN + HELD_OUT:
        ans = json.loads((REPO / "bench" / "tasks" / f"{tid}.json").read_text(encoding="utf-8"))["answer"]
        by_answer.setdefault(ans, set()).add(tid)
    crossing = {a: sorted(ids) for a, ids in by_answer.items()
                if (ids & set(TRAIN)) and (ids & set(HELD_OUT))}
    if crossing:
        print("ERROR: an answer appears on both sides of the split:", file=sys.stderr)
        for a, ids in sorted(crossing.items()):
            print(f"  {a!r}: {ids}", file=sys.stderr)
        return 2

    # For a singleton category the same-shape claim is the shared corpus where there
    # is one. Assert it rather than assume it.
    for cat in ("code_qa",):
        shas = {tasks[t]["corpus_sha256"] for t in TRAIN + HELD_OUT if tasks[t]["category"] == cat}
        if len(shas) > 1:
            print(f"ERROR: {cat} tasks no longer share one corpus ({len(shas)} distinct) — "
                  "the same-shape claim for this category rested on that.", file=sys.stderr)
            return 2

    train_shapes = {r["question_sha256"] for r in train}

    split = {
        "split_id": "s6lite-v0",
        "created": "2026-08-27",
        "spec": "docs/superpowers/specs/2026-08-27-s6-lite-v0-artifact-gate.md",
        "benchmark_version": manifest.get("benchmark_version", "v1"),
        "manifest_sha256": manifest_sha,
        "seeds": [1, 2, 3],
        "rule": ("Drawn WITHIN question shapes. `question_sha256` separates shapes only where "
                 "several tasks share one: aggregation clusters 4+3 and synthesis 3+3+2, so for "
                 "those the hash IS the shape; code QA (7) and needle (8) are all singletons "
                 "because the hash carries the target symbol or organisation, not the shape. "
                 "gate/make_split.py refuses to write a split that straddles clusters within a "
                 "category, and separately asserts that all code-QA tasks still share one "
                 "corpus_sha256 -- which is what the same-shape claim rests on there."),
        "shape_evidence": {
            "aggregation": "question_sha256 cluster 67c4d0898a (agg-04..07); the 4+3 split of v1 makes "
                           "the hash load-bearing, and agg-01/02/03 are excluded for that reason",
            "code_qa": "one identical corpus (code-bundle.txt) across all 7, one template, differing "
                       "only in the target symbol name -- the strongest same-distribution evidence in v1",
            "needle": "one template across all 8, differing only in the target organisation; distinct "
                      "corpora drawn the same way (~63.5K tokens each). A template claim, not a hash claim",
        },
        "decision_budget": {"max_decisions": 5, "reason": "spec D-S6 / C-3 -- repeated evaluation of "
                            "successive artifact versions against one held-out set is multiple comparisons"},
        "known_limitations": [
            "No adversarial task is on the held-out side: needle-01 (adversarial) is in train and "
            "agg-01 (the only other) is excluded. Recorded, not an oversight.",
            "Aggregation contributes 2 train / 2 held-out, the smallest per-category support here.",
            "codeqa-04 and codeqa-07 share the answer `rlm/budget.py` and are BOTH held-out. "
            "That is safe for contamination -- neither side leaks to the other -- but the two "
            "tasks are not fully independent, so held-out code QA has 3 tasks and 2 distinct "
            "answers. Forced by the answer-disjointness rule on a 7-task category.",
        ],
        "train": train,
        "held_out": held,
        "excluded": [{"task_id": t, "reason": r} for t, r in sorted(EXCLUDED.items())],
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(split, indent=2) + "\n"
    out.write_text(body, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (out.parent / (out.stem + ".sha256")).write_text(sha + "\n", encoding="utf-8", newline="\n")

    print(f"wrote {out}")
    print(f"  sha256      {sha}")
    print(f"  manifest    {manifest_sha[:16]}…")
    print(f"  train       {len(train)}: {', '.join(TRAIN)}")
    print(f"  held-out    {len(held)}: {', '.join(HELD_OUT)}")
    print(f"  excluded    {len(EXCLUDED)}")
    print(f"  shapes      train {len(train_shapes)}, held-out {len({r['question_sha256'] for r in held})}, orphan 0")
    by_shape: dict[str, list[str]] = {}
    for r in train + held:
        by_shape.setdefault(r["question_sha256"][:12], []).append(r["task_id"])
    for sha12, ids in sorted(by_shape.items()):
        t = [i for i in ids if i in TRAIN]
        h = [i for i in ids if i in HELD_OUT]
        print(f"    {sha12}…  train={t}  held_out={h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
