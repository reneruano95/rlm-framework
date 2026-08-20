# The delegation arm: pricing `llm_query` on the frozen v1 benchmark

**Status:** planned, not implemented · **Date:** 2026-08-20
**Answers:** `s4/RESULTS.md` §"The one thing to do next", option 2
**Blocks:** the DIRECTION.md §4a decision (is the appliance one model or two?)

## 1. The question, stated so it can be answered wrong

After S1, S2 and S4 the RLM arm has made **zero** `llm_query` calls in a scored
episode. What passed S4 is a root LLM with a sandboxed REPL and a deterministic
scaffold. Root-as-orchestrator-of-leaves is unmeasured, so "recursive language
models beat these baselines" has no evidence behind it.

This arm does not ask whether delegation *works* — it does; B2 exercised it
15,888 times in the same run. It asks whether delegation is **worth anything**
the root cannot already get from code.

## 2. Operationalization, and why this one

§8's option 2 is *"an RLM-variant with the REPL's scan powers restricted,
forcing delegation on the existing v1 tasks"*. Three readings were considered:

| | restriction | verdict |
|---|---|---|
| (a) | `chunks` become opaque handles: passable to `llm_query`, unreadable in Python | **chosen** |
| (b) | keep chunk text, drop `context`, cap REPL output | too weak — the root re-derives scanning from `chunks` |
| (c) | ban regex/`in` but allow slicing | arbitrary, and trivially circumvented by a loop |

(a) is the only one that makes delegation the *only* path to content, which is
what "forcing delegation" means. It is deliberately a **handicap**: the arm is
strictly weaker than `rlm`, and a loss against the baselines is an expected,
publishable outcome. The comparison that carries meaning is
**restricted-RLM vs `rlm`** (what the leaf adds or subtracts) with the
baselines as the fixed backdrop they already are.

## 3. Mechanism

The corpus crosses into the sandbox at exactly one place — `rlm/episode.py`,
the `# I2:` block:

```python
await session.setvar("context", context_text)
await session.setvar("chunks", chunks)
```

Restricted mode changes what those two names are bound to:

- `chunks` → a list of `ChunkRef` objects. `__repr__` gives `<chunk 3 of 424>`;
  no `__str__` returning text, no `__len__`, `__contains__`, `__getitem__`,
  `__iter__`, or comparison. Touching one as text raises a `TypeError` whose
  message names the intended path (`await llm_query(q, chunk=chunks[i])`).
- `context` → **withheld**. Leaving the full corpus readable would make the
  handles decorative.

`ChunkRef` is not JSON-serializable, so the bridge changes too. Today
`rlm/sandbox/child.py::_llm_query_template` sends
`{"question", "chunk", "role"}`. It will send `{"chunk_ref": i}` when the
argument is a `ChunkRef`, and `episode.py::_on_llm_query` resolves the index
against the episode's own chunk list — **scaffold-side, per I1**. The sandbox
never holds the text and cannot forge a ref it was not given (the index is
range-checked against the episode's chunk count; out of range is a
`DispatchError`, not a clamp).

`ChunkRef` must survive the reserved-name rule: `_SETTABLE_RESERVED` in
`child.py` already contains `{"context", "chunks"}`, so no new plumbing there.

## 4. Bench integration

- `rlm/bench.py`: `ARM_ORDER` gains `"rlm-restricted"`; `ARM_PROFILE` maps it to
  `RESIDENT_PROFILE` (same servers as `rlm`, so it adds **no relaunches** —
  it slots next to `rlm` in the per-block order and the two-relaunch bound
  from §8 is unchanged).
- `_bench_extra` stamps `arm = "rlm-restricted"` so the trace and the ledger
  distinguish it.
- `rlm/episode.py` takes a `restrict_chunks: bool` (default `False`). Default
  false is load-bearing: `run_episode` is the S1/S3 path too, and this must be
  inert unless the bench asks for it.
- `config_snapshot` records the flag, per R11 — an arm that is invisible to the
  snapshot is an arm §8 cannot score.

## 5. What must be true before it runs

1. **Zero diff to the `rlm` arm.** The restricted arm is additive. If any
   `rlm` episode changes behaviour, the S4 re-validation is invalid and the
   two questions can no longer share one bench pass.
2. **The handicap must bite.** A pre-flight fixture asserts a restricted
   episode makes `>= 1` `llm_query` call and that `repl_exec` cells cannot read
   chunk text. An arm that quietly still scans is measuring nothing.
3. **Suite green**, and new tests for: `ChunkRef` refuses text operations; the
   bridge round-trips a ref; an out-of-range ref is refused; `restrict_chunks`
   defaults off; the shipped `ARM_ORDER` contains both arms.

## 6. Cost and what it buys

Five arms instead of four over 30 tasks × 3 seeds = **450 cells**, up from 360.
The added arm is `rlm`-shaped, and S4 measured `rlm` as the *cheapest* arm
(0.78×/0.10×/0.43× the baselines' wall clock), so the projection rises from the
measured 39.6 h by roughly the `rlm` arm's own share rather than by a quarter.
The DFlash2 root is 1.46× faster on decode than the MTP root S4 actually ran,
which pulls the other way.

The pass therefore settles **both** open items in one grid:
- the S4 re-validation owed by the DFlash2 swap (`s2/DFLASH2.md` §7), and
- the delegation price, which is the project's oldest unanswered question.

## 7. Deliberately not in this plan

- **Benchmark v2** (§8 option 1, "author tasks code cannot solve"). That is a
  separate, larger piece of work and the v1 freeze stands.
- Any change to checkers, corpora, or `benchmark.manifest_sha256`. The frozen
  v1 tasks are the reference precisely because their answers are known.
