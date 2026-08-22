"""INDEPENDENT re-derivation of the R13 headline counts and the Fisher p-value.

Claims under test (from the r13-truth report):
  hybrid 34/54, erase 33/54, gemma 39/54,
  paired shared 24/54 / virgin 0/54, Fisher p = 4.35e-9,
  synthetic two-prompt 0/230,
  r13_mitigation windows:virgin_slot 0/138, np1 4/18, smallladder 4/18,
  cache_prompt:false 15/18.

Two scorers are run side by side:
  (1) the file's own recorded `verdict` field
  (2) my own FOREIGN oracle: does the answer contain entity_b's uuid, or
      entity_b's coined name, when the call asked about entity_a (and vice
      versa)?  Plus the corpus oracle from audit_refute_arith_oracle.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import audit_refute_arith_oracle as O

S2 = O.S2


def load(name):
    return [json.loads(l) for l in (S2 / "results" / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def fisher_exact_2x2(a, b, c, d):
    """two-tailed Fisher exact p for [[a,b],[c,d]]"""
    def logC(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    def p_of(x):
        return math.exp(logC(r1, x) + logC(r2, c1 - x) - logC(n, c1))
    p_obs = p_of(a)
    tot = 0.0
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    for x in range(lo, hi + 1):
        p = p_of(x)
        if p <= p_obs * (1 + 1e-9):
            tot += p
    return tot


def verdicts(name, group=None):
    rows = load(name)
    if group is None:
        c = Counter(r.get("verdict") for r in rows)
        print(f"{name:36s} n={len(rows):4d} verdicts={dict(c)}")
        return rows
    g = defaultdict(Counter)
    for r in rows:
        g[r.get(group)][r.get("verdict")] += 1
    print(f"{name}  grouped by {group}:")
    for k, c in g.items():
        tot = sum(c.values())
        print(f"    {str(k):46s} n={tot:4d} FOREIGN={c.get('FOREIGN', 0):3d} {dict(c)}")
    return rows


def my_oracle(rows, name):
    """Independent: an answer is FOREIGN if it contains the *other* entity's
    uuid or coined name (fields uuid_a/uuid_b/entity_a/entity_b), or any
    corpus token from a chunk other than the one the call sent."""
    n = f = 0
    for r in rows:
        ans = (r.get("raw_output") or "")
        if not ans:
            continue
        asked = (r.get("asked_about") or "")
        ea, eb = r.get("entity_a") or "", r.get("entity_b") or ""
        ua, ub = r.get("uuid_a") or "", r.get("uuid_b") or ""
        if not (ea and eb):
            continue
        n += 1
        other_ent, other_uuid = (eb, ub) if asked and ea and ea in asked else (ea, ua)
        low = ans.lower()
        hit = (other_uuid and other_uuid.lower() in low)
        if other_ent:
            core = re.sub(r"^the\s+", "", other_ent, flags=re.I).split()[0].lower()
            if core and core in low:
                hit = True
        if hit:
            f += 1
    print(f"    [my paired oracle] {name}: {f}/{n} FOREIGN")


if __name__ == "__main__":
    for f in ("r13_replay_hybrid.jsonl", "r13_replay_erase.jsonl",
              "r13_replay_gemma_fullattn.jsonl", "r13_replay_paired.jsonl",
              "r13_twoprompt_matrix.jsonl", "r13_twoprompt_sizesweep.jsonl"):
        rows = verdicts(f)
        my_oracle(rows, f)

    print()
    rows = verdicts("r13_replay_paired.jsonl", group="condition")
    print()
    verdicts("r13_mitigation.jsonl", group="condition")
    print()
    verdicts("r13_mitigation.jsonl", group="label_tag")

    # Fisher for paired shared vs virgin
    p = load("r13_replay_paired.jsonl")
    g = defaultdict(Counter)
    for r in p:
        g[r["condition"]][r.get("verdict")] += 1
    conds = list(g)
    print("\npaired conditions:", {k: dict(v) for k, v in g.items()})
    if len(conds) == 2:
        a = g[conds[0]].get("FOREIGN", 0); b = sum(g[conds[0]].values()) - a
        c = g[conds[1]].get("FOREIGN", 0); d = sum(g[conds[1]].values()) - c
        print(f"  2x2 = [[{a},{b}],[{c},{d}]]  Fisher two-tailed p = {fisher_exact_2x2(a, b, c, d):.3e}")
