"""S2 -- R14: scoring the concurrency ladder, and telling DEGENERATE from wrong.

WHY THIS FILE EXISTS SEPARATELY FROM THE RUNNER. `milestones/s2/run_occupancy.py` records
one quality field, `answer_correct` (does the reply contain this document's own
identifier?). That was enough to SEE R14 -- the correct count falls off a cliff
the moment a second call is in flight -- and it is not enough to CHARACTERISE
it, because it cannot distinguish the three ways a reply can fail to carry the
identifier:

  * the leaf answered fluently and got it wrong    -> CONFABULATION
  * the leaf answered fluently and offered nothing -> MISS
  * the decode broke                               -> DEGENERATE

Only the third is R14's signature. R13's foreign-identifier detector is blind
to all three (degenerate text carries no identifier to be foreign), which is
exactly why a contamination-clean audit is not an answer-quality audit.

THE DETECTOR IS PRE-REGISTERED AND MECHANICAL, and its specificity is measured
rather than asserted: `--validate` scores the serial arms, where the model is
known to be working, and reports how often each rule fires there. A rule that
fires on a healthy serial reply is a false positive and is reported as one.

Eight rules, any one of which marks a reply DEGENERATE:

  D1 STUB      -- fewer than 12 non-space characters, or `predicted_n <= 2`.
  D2 ECHO      -- a >= 32-character verbatim window of one of the eight known
                  filler sentences appears in the reply. The filler is the
                  document's own body text; an answer that quotes it back is
                  regurgitating prompt, not answering.
  D3 QECHO     -- the question is repeated verbatim in the reply.
  D4 NONLATIN  -- a character outside Latin-1 and Latin Extended-B. The corpus
                  and the prompt are pure ASCII, so a Cyrillic or CJK codepoint
                  cannot be a legitimate continuation of either.
  D5 LOOP      -- some WORD repeats >= 4 times in a row.
  D6 FRAGMENT  -- an unclosed bold span (`**` an odd number of times), which is
                  how a truncated `**<identifier>**` answer terminates.
  D7 CHARLOOP  -- some CHARACTER repeats >= 6 times in a row. This is the
                  single commonest signature at concurrency 8 (`//////...`) and
                  D5 cannot see it, because a punctuation run contains no words.
  D9 THINK     -- a `<think>` control block. `leaf.enable_thinking` is false
                  and the prefix is rendered with thinking off, so the model
                  emitting the control token at all is a broken decode.
  D10 EOSCUT   -- the reply STOPPED ON `eos` in mid-phrase: it ends neither in
                  terminal punctuation nor in a complete UUID. This is the
                  quietest signature and the one that most changes the counts.
                  `'The record identifier stated in thecik'` and
                  `'Based on the text, additionalandrope'` are whole replies,
                  not excerpts -- the model emitted end-of-sequence eight
                  tokens into a sentence. Scored naively they look like a
                  fluent MISS; they are a decode that died.
                  RESTRICTED TO `stop_type == "eos"` ON PURPOSE: a reply cut by
                  the `n_predict` budget is also unterminated, and that is a
                  budget cut rather than a defect. Without the restriction this
                  rule fired on 1 of the 866 healthy replies (a `limit` stop);
                  with it, on 0.

MEASURED SPECIFICITY (this is the reason to trust the column). Run against the
866 CORRECT replies in the seven SERIAL conditions of `milestones/s2/results/occupancy.jsonl`
-- `baseline`, `cram0`, `nocacheidle`, `sps0`, `shuffle`, `w640`, `cram0-w640`,
where the model is known to be working -- every rule above fires **0 times**.
`--validate` re-runs that check on whatever file it is pointed at.

ONE RULE WAS TRIED AND REJECTED, recorded so it is not re-proposed: "the reply
starts mid-sentence (leading lowercase)" fired on 227 of those 866 healthy
replies, because the leaf's shortest correct answer is a bare UUID and a UUID
often starts with a lowercase hex digit. It is not in the set.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2.run_occupancy import FILLER, QUESTION, chunk_identifier  # noqa: E402

S2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = S2_DIR / "results"

CORRECT = "CORRECT"
MISS = "MISS"
CONFABULATION = "CONFABULATION"
FALSE_POSITIVE = "FALSE-POSITIVE"
MALFORMED = "MALFORMED"
LABELS = (CORRECT, MISS, CONFABULATION, FALSE_POSITIVE, MALFORMED)

#: An identifier-shaped token: the corpus's own shape is a 8-4-4-4-12 hex UUID,
#: but a confabulating leaf invents things like `RECORD-2023-04-15-001`, so the
#: test is "hyphenated alphanumeric run" rather than "valid UUID".
IDENT_RE = re.compile(r"\b[A-Za-z0-9]{2,}(?:-[A-Za-z0-9]{2,}){2,}\b")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
WORD_RE = re.compile(r"[A-Za-z0-9]+")
CHARLOOP_RE = re.compile(r"(.)\1{5,}", re.S)
UUID_TAIL_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
TERMINAL_CHARS = set(".!?`*\")]:’”")
ECHO_WINDOW = 32


def echo_windows() -> list[str]:
    """Every 48-character window of every filler sentence, once."""
    out: set[str] = set()
    for s in FILLER:
        for i in range(0, max(len(s) - ECHO_WINDOW + 1, 0)):
            out.add(s[i:i + ECHO_WINDOW])
    return sorted(out)


_ECHO = echo_windows()


def degenerate_rules(answer: str, predicted_n: int,
                     stop_type: str = "") -> list[str]:
    """Which rules fire on this reply. Empty list = not degenerate."""
    fired: list[str] = []
    a = answer or ""
    stripped = a.strip()

    if len(stripped) < 12 or predicted_n <= 2:
        fired.append("D1_STUB")
    if any(w in a for w in _ECHO):
        fired.append("D2_ECHO")
    if QUESTION in a:
        fired.append("D3_QECHO")
    if any(ord(c) > 0x24F for c in a):
        fired.append("D4_NONLATIN")

    words = [w.lower() for w in WORD_RE.findall(a)]
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run >= 4:
            fired.append("D5_LOOP")
            break

    if a.count("**") % 2 == 1:
        fired.append("D6_FRAGMENT")
    if CHARLOOP_RE.search(a):
        fired.append("D7_CHARLOOP")
    if "<think>" in a or "</think>" in a:
        fired.append("D9_THINK")

    if stop_type == "eos" and stripped:
        tail = re.split(r"\s", stripped)[-1].strip(".,;:")
        if stripped[-1] not in TERMINAL_CHARS and not UUID_TAIL_RE.search(tail):
            fired.append("D10_EOSCUT")
    return fired


def offered_identifier(answer: str, own: str) -> str | None:
    """The identifier-shaped string this reply hands over, if any. Bold spans
    first (the leaf's habit is `**<id>**`), then any hyphenated run."""
    for m in BOLD_RE.finditer(answer):
        cand = m.group(1).strip().strip(".")
        if IDENT_RE.fullmatch(cand) or cand == own:
            return cand
    m = IDENT_RE.search(answer)
    return m.group(0) if m else None


def classify(rec: dict[str, Any]) -> dict[str, Any]:
    """One record -> taxonomy label + the degeneracy verdict alongside it."""
    if rec.get("status") != "ok":
        return {"label": MALFORMED, "degenerate": True,
                "rules": ["D0_ERROR"], "offered": None}

    answer = rec.get("answer") or ""
    own = chunk_identifier(int(rec["ordinal"]))
    rules = degenerate_rules(answer, int(rec.get("predicted_n") or 0),
                             str(rec.get("stop_type") or ""))
    correct = own in answer

    # A reply that states the right identifier is CORRECT even if a rule fires
    # on the prose around it -- the leaf did the job. This ordering is
    # deliberate and it biases AGAINST the finding.
    if correct:
        return {"label": CORRECT, "degenerate": False,
                "rules_on_correct": rules, "rules": [], "offered": own}
    if rules:
        return {"label": MALFORMED, "degenerate": True, "rules": rules,
                "offered": offered_identifier(answer, own)}

    offered = offered_identifier(answer, own)
    if offered is not None:
        return {"label": CONFABULATION, "degenerate": False, "rules": [],
                "offered": offered}
    return {"label": MISS, "degenerate": False, "rules": [], "offered": None}


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]. Implemented here because
    the project has no scipy and the whole question -- 'is the fixed arm FIXED
    or merely LESS BAD?' -- is a hypothesis test on two small 2x2 tables. Sums
    the probability of every table at least as extreme as the observed one."""
    from math import comb

    n = a + b + c + d
    row1, col1 = a + b, a + c
    total = comb(n, col1)
    p_obs = comb(row1, a) * comb(n - row1, col1 - a) / total
    p = 0.0
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    for k in range(lo, hi + 1):
        pk = comb(row1, k) * comb(n - row1, col1 - k) / total
        if pk <= p_obs * (1 + 1e-9):
            p += pk
    return min(p, 1.0)


def load(path: Path, conditions: set[str] | None = None) -> list[dict[str, Any]]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if conditions is None or r.get("condition") in conditions:
            recs.append(r)
    return recs


def score_condition(recs: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    rules = Counter()
    for r in recs:
        v = classify(r)
        r["_v"] = v
        labels[v["label"]] += 1
        for rule in v["rules"]:
            rules[rule] += 1
    ok = [r for r in recs if r.get("status") == "ok"]
    walls = [r["wall_ms"] / 1000.0 for r in ok if "wall_ms" in r]
    meta = recs[0]
    return {
        "condition": meta["condition"],
        "extra": meta.get("extra", ""),
        "cont_batching": meta.get("cont_batching", True),
        "temperature": meta.get("temperature"),
        "drain_stream": meta.get("drain_stream", False),
        "ub": meta.get("ub", 512),
        "concurrency": meta.get("concurrency", 1),
        "n": len(recs),
        "labels": {k: labels.get(k, 0) for k in LABELS},
        "degenerate": sum(1 for r in recs if r["_v"]["degenerate"]),
        "rules": dict(rules),
        "leaks": sum(1 for r in recs if r.get("leak_detected")),
        "slot_mismatches": sum(1 for r in recs if r.get("slot_mismatch")),
        "errors": sum(1 for r in recs if r.get("status") != "ok"),
        "median_wall_s": round(statistics.median(walls), 2) if walls else None,
        "total_wall_s": round(sum(walls), 1) if walls else None,
    }


def treatment(row: dict[str, Any]) -> str:
    """Everything this condition moved off the arm-B baseline, read back out of
    the RECORDED metadata rather than out of the condition's name. The first
    attempt at the hypothesis battery ran every treatment with default flags
    (`-Args` collides with PowerShell's automatic `$Args`), so a name is not
    evidence that a flag took effect -- this column is."""
    bits = [f"`{row['extra']}`" if row["extra"] else "`(host cache default)`"]
    if not row.get("cont_batching", True):
        bits.append("`--no-cont-batching`")
    if row.get("ub", 512) != 512:
        bits.append(f"`-ub {row['ub']}`")
    if row.get("temperature") is not None and abs(row["temperature"] - 0.3) > 1e-9:
        bits.append(f"`temp {row['temperature']}`")
    if row.get("drain_stream"):
        bits.append("`drain-stream`")
    return " + ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=str(RESULTS_DIR / "r14.jsonl"))
    ap.add_argument("--conditions", default="",
                    help="comma-separated; default every condition in the file")
    ap.add_argument("--validate", action="store_true",
                    help="report per-rule firing on CORRECT replies "
                         "(the detector's false-positive rate)")
    ap.add_argument("--samples", type=int, default=0,
                    help="print N verbatim degenerate replies per condition")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--compare", default="",
                    help="semicolon-separated CONDA,CONDB pairs; Fisher exact "
                         "on CORRECT counts")
    args = ap.parse_args()

    conds = set(c for c in args.conditions.split(",") if c) or None
    recs = load(Path(args.path), conds)
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        by_cond.setdefault(r["condition"], []).append(r)

    rows = [score_condition(rs) for rs in by_cond.values()]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        hdr = ("| condition | conc | treatment | n | CORRECT | MISS | CONFAB | "
               "MALFORMED | of which DEGENERATE | R13 hits | median wall s |")
        print(hdr)
        print("|" + "---|" * 11)
        for row in sorted(rows, key=lambda r: (r["condition"])):
            lab = row["labels"]
            print(f"| `{row['condition']}` | {row['concurrency']} | "
                  f"{treatment(row)} | {row['n']} | "
                  f"{lab[CORRECT]} | {lab[MISS]} | {lab[CONFABULATION]} | "
                  f"{lab[MALFORMED]} | {row['degenerate']} | {row['leaks']} | "
                  f"{row['median_wall_s']} |")
        print()
        for row in sorted(rows, key=lambda r: r["condition"]):
            print(f"{row['condition']}: rules={row['rules']} "
                  f"errors={row['errors']} mismatches={row['slot_mismatches']}")

    if args.validate:
        print("\n### detector specificity: rules firing on CORRECT replies")
        fp = Counter()
        n_correct = 0
        for r in recs:
            v = r.get("_v") or classify(r)
            if v["label"] == CORRECT:
                n_correct += 1
                for rule in v.get("rules_on_correct", []):
                    fp[rule] += 1
        print(f"CORRECT replies: {n_correct}; rules that would have fired on "
              f"them: {dict(fp) or 'none'}")

    if args.compare:
        print("\n### Fisher exact on CORRECT counts")
        index = {r["condition"]: r for r in rows}
        for pair in args.compare.split(";"):
            x, y = pair.split(",")
            rx, ry = index[x], index[y]
            ax, nx = rx["labels"][CORRECT], rx["n"]
            ay, ny = ry["labels"][CORRECT], ry["n"]
            p = fisher_exact(ax, nx - ax, ay, ny - ay)
            print(f"{x} {ax}/{nx} ({ax/nx:.1%}) vs {y} {ay}/{ny} "
                  f"({ay/ny:.1%}): p = {p:.3g}")

    if args.samples:
        print("\n### verbatim degenerate samples")
        for cond, rs in sorted(by_cond.items()):
            shown = 0
            for r in rs:
                v = r.get("_v") or classify(r)
                if not v["degenerate"]:
                    continue
                print(f"\n[{cond}] ordinal {r['ordinal']} slot {r.get('id_slot')} "
                      f"stop={r.get('stop_type')} predicted_n={r.get('predicted_n')} "
                      f"rules={v['rules']} R13={r.get('leak_detected')}")
                print("    " + repr(r.get("answer", ""))[:500])
                shown += 1
                if shown >= args.samples:
                    break


if __name__ == "__main__":
    main()
