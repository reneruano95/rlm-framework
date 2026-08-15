"""VERIFICATION-ONLY audit (read-only, zero GPU).

Re-derives the numbers a survey plan asserts about the leaf-envelope A/B from
`s2/results/refusal-ab*.jsonl` + fixture manifests, independently of
`audit_envelope_entity_binding.py`, and checks the claims that script does NOT
make: the abstain-exempt form of the entity rule, the empty-evidence shape of
the correct abstentions, the abstain-with-answer contract violations, sampling
and slot hygiene, and the label vocabulary.

Stdlib only. Reads; writes nothing but stdout.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return _WS.sub(" ", t or "")


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def fixtures() -> dict[tuple[str, str], dict]:
    idx: dict[tuple[str, str], dict] = {}
    for d in sorted(ROOT.joinpath("s2").glob("fixtures-refusal*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        seed = str(m.get("seed"))
        for cell_id, cell in m.get("cells", {}).items():
            chunk = d / f"{cell_id}.chunk.txt"
            c = dict(cell)
            c["_chunk"] = chunk.read_text(encoding="utf-8") if chunk.exists() else ""
            c["_dir"] = d.name
            idx[(cell_id, seed)] = c
    return idx


def ekey(e: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", norm(e).strip(), flags=re.IGNORECASE)


def main() -> int:
    fx = fixtures()
    for name in ("refusal-ab-640.jsonl", "refusal-ab.jsonl"):
        recs = load(ROOT / "s2" / "results" / name)
        print(f"\n{'='*78}\n{name}: {len(recs)} records\n{'='*78}")

        # --- hygiene ---------------------------------------------------------
        print("temperature values:", Counter(r.get("temperature") for r in recs))
        print("top_p values      :", Counter(r.get("top_p") for r in recs))
        print("seed values       :", Counter(r.get("seed") for r in recs))
        print("status values     :", Counter(r.get("status") for r in recs))
        print("label vocabulary  :", Counter(r.get("label") for r in recs))
        print("leak_detected     :", Counter(r.get("leak_detected") for r in recs))
        print("slot_ok           :", Counter(r.get("slot_ok") for r in recs))
        mism = [r for r in recs if r.get("id_slot") != r.get("requested_slot")]
        print("id_slot != requested_slot:", len(mism))
        slots_by_key: dict[tuple, set] = defaultdict(set)
        for r in recs:
            slots_by_key[(r["arm"], r["cell_uid"])].add(r["requested_slot"])
        allslots = [s for v in slots_by_key.values() for s in v]
        print(f"distinct requested slots: {len(set(allslots))} "
              f"over {len(slots_by_key)} (arm,cell) groups; "
              f"slots used by >1 group: "
              f"{sum(1 for s, c in Counter(allslots).items() if c > 1)}")
        ts = sorted(r["ts"] for r in recs if r.get("ts"))
        print("ts range:", ts[0], "->", ts[-1])

        # --- detectors -------------------------------------------------------
        rows = []
        for r in recs:
            if str(r.get("envelope")) != "True" or str(r.get("envelope_ok")) != "True":
                continue
            cell = fx.get((r["cell_id"], str(r["fixture_seed"])))
            if cell is None:
                continue
            q = cell["questions"].get(r["question_type"])
            if not q:
                continue
            ent = ekey(q.get("entity", ""))
            spans = r.get("evidence") or []
            hay = norm(cell["_chunk"])
            span_ok = bool(spans) and all(
                norm(s).strip() and norm(s).strip() in hay for s in spans)
            ent_in = any(ent and ent in norm(s) for s in spans)
            rows.append({
                "arm": r["arm"], "cell": r["cell_uid"], "q": r["question_type"],
                "label": r["label"], "abstain": r.get("abstain"),
                "answer_empty": not (r.get("reduced_text") or "").strip(),
                "n_spans": len(spans), "span_ok": span_ok, "ent_in": ent_in,
                "ent": ent, "raw": r.get("raw_output") or "",
                "entity_in_chunk": bool(ent) and ent in hay,
                "awa": r.get("abstain_with_answer"),
            })
        if not rows:
            print("(no parsed envelope replies)")
            continue

        wrong = [x for x in rows if x["label"] in ("FALSE-POSITIVE", "CONFABULATION")]
        right = [x for x in rows if x["label"] == "CORRECT"]
        print(f"\nparsed={len(rows)} wrong={len(wrong)} right={len(right)} "
              f"other={len(rows)-len(wrong)-len(right)}")

        def exempt_reject(x: dict) -> bool:
            """PROPOSED RULE AS THE PLAN WORDS IT: only applies when evidence is
            non-empty."""
            return x["n_spans"] > 0 and not x["ent_in"]

        print("\n-- entity rule, ABSTAIN-EXEMPT form (plan's stated wording) --")
        print(f"  rejects wrong  : {sum(1 for x in wrong if exempt_reject(x))}/{len(wrong)}")
        print(f"  rejects correct: {sum(1 for x in right if exempt_reject(x))}/{len(right)}")
        print("-- entity rule, UNCONDITIONAL form (what the audit script computes) --")
        print(f"  rejects wrong  : {sum(1 for x in wrong if not x['ent_in'])}/{len(wrong)}")
        print(f"  rejects correct: {sum(1 for x in right if not x['ent_in'])}/{len(right)}")

        print("\n-- correct replies, by question type: spans / entity / rejections --")
        agg: dict[str, Counter] = defaultdict(Counter)
        for x in right:
            agg[x["q"]]["n"] += 1
            agg[x["q"]]["empty_spans"] += (x["n_spans"] == 0)
            agg[x["q"]]["ent_missing"] += (not x["ent_in"])
            agg[x["q"]]["exempt_rejects"] += exempt_reject(x)
        for k in sorted(agg):
            print(f"  {k:<12}{dict(agg[k])}")

        print("\n-- wrong replies with abstain True --")
        for x in wrong:
            if x["abstain"] is True:
                print(f"  {x['arm']:<15}{x['cell']:<18}n_spans={x['n_spans']} "
                      f"answer_empty={x['answer_empty']} awa={x['awa']} "
                      f"ent_in={x['ent_in']}")
                print(f"     raw={x['raw'][:220]!r}")

        print("\n-- abstain_with_answer over ALL parsed envelope replies --")
        print("  ", Counter((x["label"], x["awa"]) for x in rows))

        print("\n-- entity-in-chunk satisfiability (parsed replies, unique cell x qtype) --")
        sat: dict[str, Counter] = defaultdict(Counter)
        seen = set()
        for x in rows:
            k = (x["cell"], x["q"])
            if k in seen:
                continue
            seen.add(k)
            sat[x["q"]]["cells"] += 1
            sat[x["q"]]["entity_in_chunk"] += x["entity_in_chunk"]
        for k in sorted(sat):
            print(f"  {k:<12}{dict(sat[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
