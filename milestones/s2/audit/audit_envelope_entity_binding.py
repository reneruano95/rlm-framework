"""Offline, zero-GPU audit of the leaf-envelope A/B data already on disk.

Answers three questions from `milestones/s2/results/refusal-ab*.jsonl` plus the fixture
manifests, with NO model calls:

  1. FORMAT COST -- prefix tokens, tokens_in, tokens_out, wall_s, by arm.
  2. THE SPEC'D SPAN CHECK -- of the envelope replies that are WRONG, how many
     does `normalized span in normalized chunk` reject? (Expected: ~none; the
     failures are misattribution, and a misattributed answer quotes a real
     in-chunk span.)
  3. THE PROPOSED ENTITY-BINDING CHECK -- does any evidence span contain the
     QUESTION's entity (from the fixture manifest's `entity` field)? Reported
     as catch rate on wrong answers AND false-rejection rate on right ones,
     because a detector is only as good as its cost on the correct arm.

Stdlib only. Reads; writes nothing but stdout.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """rlm.envelope.normalize_ws, re-implemented so this audit never depends on
    the module it is auditing."""
    return _WS.sub(" ", text or "")


def load(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def fixture_index() -> dict[tuple[str, str], dict]:
    """(cell_id, fixture_seed) -> cell manifest entry, with its chunk text."""
    idx: dict[tuple[str, str], dict] = {}
    for d in sorted(ROOT.joinpath("s2").glob("fixtures-refusal*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        seed = str(m.get("seed"))
        for cell_id, cell in m.get("cells", {}).items():
            chunk = d / f"{cell_id}.chunk.txt"
            cell = dict(cell)
            cell["_chunk"] = chunk.read_text(encoding="utf-8") if chunk.exists() else ""
            cell["_dir"] = d.name
            idx[(cell_id, seed)] = cell
    return idx


def entity_key(entity: str) -> str:
    """The question's entity with a leading article dropped: the manifest writes
    'the Ravvmestthorpe Trust', a leaf writes 'Ravvmestthorpe Trust'."""
    e = norm(entity).strip()
    return re.sub(r"^(the|a|an)\s+", "", e, flags=re.IGNORECASE)


def head_token(entity: str) -> str:
    """The coined proper noun alone ('Ravvmestthorpe'), the strictest single
    token that identifies the entity. Reported beside the full-phrase rule so
    the choice of strictness is visible rather than assumed."""
    k = entity_key(entity)
    return k.split()[0] if k else ""


def main() -> int:
    fixtures = fixture_index()
    for name in ("refusal-ab-640.jsonl", "refusal-ab.jsonl"):
        path = ROOT / "milestones" / "s2" / "results" / name
        if not path.exists():
            print(f"MISSING {path}")
            continue
        recs = load(path)
        print(f"\n{'=' * 78}\n{name}: {len(recs)} records\n{'=' * 78}")

        # ---- 1. format cost ------------------------------------------------
        by_arm: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_arm[r["arm"]].append(r)
        print("\n-- format cost (per arm; median over that arm's records) --")
        print(f"{'arm':<18}{'n':>5}{'prefix_tok':>12}{'tok_in':>9}"
              f"{'tok_out':>9}{'wall_s':>9}{'parse_ok':>10}")
        for arm in sorted(by_arm):
            rs = by_arm[arm]

            def med(key: str, rs: list[dict] = rs) -> float:
                vals = sorted(float(r[key]) for r in rs
                              if r.get(key) not in (None, "None"))
                return vals[len(vals) // 2] if vals else float("nan")

            pref = {r.get("prefix_tokens") for r in rs}
            envelope_recs = [r for r in rs if str(r.get("envelope")) == "True"]
            ok = sum(1 for r in envelope_recs if str(r.get("envelope_ok")) == "True")
            parse = f"{ok}/{len(envelope_recs)}" if envelope_recs else "-"
            print(f"{arm:<18}{len(rs):>5}{str(sorted(pref)):>12}"
                  f"{med('tokens_in'):>9.0f}{med('tokens_out'):>9.0f}"
                  f"{med('wall_s'):>9.2f}{parse:>10}")

        # ---- 2 + 3. detectors on envelope replies --------------------------
        rows = []
        for r in recs:
            if str(r.get("envelope")) != "True":
                continue
            if str(r.get("envelope_ok")) != "True":
                continue
            cell = fixtures.get((r["cell_id"], str(r["fixture_seed"])))
            if cell is None:
                continue
            q = cell["questions"].get(r["question_type"])
            if not q:
                continue
            ent = entity_key(q.get("entity", ""))
            head = head_token(q.get("entity", ""))
            spans = r.get("evidence") or []
            if isinstance(spans, str):
                spans = json.loads(spans.replace("'", '"')) if spans != "None" else []
            hay = norm(cell["_chunk"])
            span_ok = bool(spans) and all(
                norm(s).strip() and norm(s).strip() in hay for s in spans)
            ent_in_span = any(ent and ent in norm(s) for s in spans)
            head_in_span = any(head and head in norm(s) for s in spans)
            rows.append({
                "arm": r["arm"], "cell": r["cell_uid"], "qtype": r["question_type"],
                "label": r["label"], "abstain": r.get("abstain"),
                "span_ok": span_ok, "ent_in_span": ent_in_span,
                "head_in_span": head_in_span, "n_spans": len(spans),
                "entity": ent,
            })
        if not rows:
            print("\n(no parsed envelope replies joined to a fixture)")
            continue

        wrong = [x for x in rows if x["label"] in ("FALSE-POSITIVE", "CONFABULATION")]
        right = [x for x in rows if x["label"] == "CORRECT"]
        print(f"\n-- detectors over {len(rows)} parsed envelope replies "
              f"({len(wrong)} wrong, {len(right)} correct) --")

        def rate(n: int, d: int) -> str:
            return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "-"

        print("WRONG answers (FALSE-POSITIVE + CONFABULATION):")
        print(f"  spec'd span check REJECTS      : "
              f"{rate(sum(1 for x in wrong if not x['span_ok']), len(wrong))}")
        print(f"  proposed entity-in-span REJECTS: "
              f"{rate(sum(1 for x in wrong if not x['ent_in_span']), len(wrong))}")
        print(f"  proposed head-token   REJECTS  : "
              f"{rate(sum(1 for x in wrong if not x['head_in_span']), len(wrong))}")
        print("CORRECT answers (a rejection here is a FALSE REJECTION):")
        print(f"  spec'd span check REJECTS      : "
              f"{rate(sum(1 for x in right if not x['span_ok']), len(right))}")
        print(f"  proposed entity-in-span REJECTS: "
              f"{rate(sum(1 for x in right if not x['ent_in_span']), len(right))}")
        print(f"  proposed head-token   REJECTS  : "
              f"{rate(sum(1 for x in right if not x['head_in_span']), len(right))}")

        print("\n-- by arm x question_type --")
        agg: dict[tuple, dict] = defaultdict(
            lambda: {"n": 0, "span_bad": 0, "ent_bad": 0, "labels": defaultdict(int)})
        for x in rows:
            a = agg[(x["arm"], x["qtype"])]
            a["n"] += 1
            a["span_bad"] += (not x["span_ok"])
            a["ent_bad"] += (not x["ent_in_span"])
            a["labels"][x["label"]] += 1
        for k in sorted(agg):
            a = agg[k]
            print(f"  {k[0]:<16}{k[1]:<12} n={a['n']:<4} "
                  f"span-rejects={a['span_bad']:<4} entity-rejects={a['ent_bad']:<4} "
                  f"{dict(a['labels'])}")

        print("\n-- every WRONG parsed envelope: what the span actually said --")
        for x in wrong[:40]:
            print(f"  {x['arm']:<16}{x['cell']:<18}{x['qtype']:<11}"
                  f"abstain={str(x['abstain']):<6}span_ok={str(x['span_ok']):<6}"
                  f"entity_in_span={str(x['ent_in_span']):<6}q-entity={x['entity']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
