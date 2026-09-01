# rlm-halo

A local **Recursive Language Model** runtime for one AMD Strix Halo box: a big root
model writes Python in a sandboxed REPL where the long context is a *variable*, not
a message, and delegates semantic sub-queries to a cheap leaf model. Two llama.cpp
servers, 128 GB of unified memory, no network.

The governing documents, in order of authority:

| | |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The spec — invariants, gates, and how claims are judged. Binding. Start at its Contents. |
| [`DIRECTION.md`](DIRECTION.md) | Product direction (decided 2026-08-16). Where this is going, and why. |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history, extracted from `ARCHITECTURE.md` §14 on 2026-08-22. |
| [`config.yaml`](config.yaml) | **The only source of truth for configuration.** Not a sketch — the real file. |

**Prime directive:** the LLM proposes; the scaffold disposes.

---

## What each directory is

Every top-level directory belongs to exactly one of five kinds. The kind is what
tells you whether you may touch it.

| directory | kind | notes |
|---|---|---|
| `src/rlm/` | **live code** | The runtime, and the only thing in the wheel. Moved here from `rlm/` on 2026-08-22. |
| `bench/` | **live code + frozen artifact** | The benchmark builder *and* frozen v1 (`manifest.json`, `tasks/`, `corpora/`). The frozen half is pinned by `benchmark.manifest_sha256`. |
| `tests/` | **live code** | Repo-level tests: the benchmark, the gate, citations, the dependency lint. The package carries its own contract suite at `src/rlm/_tests/`, which ships inside the copy unit. 869 tests across the two. `test_import_rules.py` enforces the C1–C6 dependency rule as a checked invariant, and resolves the package by `find_spec` so it cannot pass vacuously on a copy. |
| `src/rlm/_data/` | **frozen artifact, inside the package** | The 18 sha-pinned prompts and `config.default.yaml`. Moved here from a root-level `prompts/` on 2026-09-01 so a copied `rlm/` needs no repo. See "before you move anything". |
| `upstream/` | **active work, not evidence** | llama.cpp defect reports (R13 slot leak, R14 continuous batching). Both still open upstream. It belongs to no gate and `ARCHITECTURE.md` cites it zero times — it was briefly filed under `milestones/` on 2026-08-22 and moved back out the same day, because "a bug report about someone else's project" is not a milestone. |
| `docs/` | **live documents** | Research, specs, plans. Conventions in [`docs/README.md`](docs/README.md). |
| `traces/` | **gitignored, and irreplaceable** | The trace store: DuckDB + 67,136 blobs. Under I4 this is the *sole* episode truth. Not in git. Not reconstructible. Kept at the top level deliberately — it is the project's most valuable artifact, not a build output. |
| `tools/` | **gitignored, machine-bound** | Three live llama.cpp builds, ~1.3 GB. |

Anything not listed above is temporary and is deleted on sight:
`sandbox_bootstrap/`, `**/__pycache__/`, `.pytest_cache/`, `.hypothesis/`.
`sandbox_bootstrap/` looks load-bearing because
the directory carries a one-time `icacls /grant *S-1-15-2-1` ACE — it is not.
`rlm validate` recreates the directory, its four staged files **and** the ACE.
Verified on 2026-08-22 by deleting it: confinement re-tested as denied.

## Three things that cost the most to rediscover

1. **The gate evidence is in git, not on disk.** `milestones/` was deleted from
   the working tree on 2026-08-22. Read any of it with
   `git show 4e75b53:milestones/s2/R13.md`. The ~176 citations to it across
   `ARCHITECTURE.md`, `CHANGELOG.md`, `config.yaml` and `src/rlm` docstrings
   were deliberately kept — a spec that stops saying where its numbers came
   from is worse than one whose citations need a `git show` — and
   `tests/test_citations.py` checks every one is still retrievable.
2. **`traces/` is gitignored *and* is the sole episode truth.** Both halves matter:
   git will not save you, and nothing else holds the record.
3. **`tools/llamacpp-vulkan-dflash2/` has no zip and cannot be re-downloaded.** It
   is the only build that loads the DFlash2 drafter, and its source is an unmerged
   llama.cpp PR at commit `5ecbe1a` that can be force-pushed. Back it up before you
   touch `tools/`.

## Commands

```bash
uv run pytest -q                              # 869 tests, ~10 min (repo + the package's own)
uv run pytest --pyargs rlm -q                 # just the package's contract suite, ~7 s
uv run python tests/verify_distribution.py    # copy it out, build a wheel, prove both run
uv run rlm validate --no-server-probe         # config + prompt pins + sandbox confinement
uv run rlm replay <episode-id>                # re-derive an episode from the store alone (I4)
uv run rlm bench --smoke                      # the only path that launches the root server
python -c "from rlm.config import load_config; load_config()"   # ./config.yaml, else the shipped one
```

Always run from the repository root: nothing resolves paths against the config
file's own directory.

## Before you move anything

This repo pins file *contents* by sha256 into every episode's `config_snapshot`,
and `rlm replay` is the gate that proves episode state is re-derivable from the
trace store alone. Several things are therefore frozen, and two of them fail
**silently**.

- **`src/rlm/_data/prompts/**` — all 18 files, including versions that look
  superseded.** 614 episode snapshots store the path as it was DECLARED
  (`prompts/root.v3.md`), and the registry is rebuilt from the *snapshot's* path.
  Hashes cover content only, so a byte-identical file in a new place used to break
  replay. **This is why the 2026-09-01 move was survivable and a plain `mv` would
  not have been:** `rlm.config.resolve_prompt_path` tries the recorded path first and
  the package copy second, and it runs inside `PromptRegistry._load_one`, the single
  point where a prompt is opened. An edit — even to a changelog comment the loader
  strips — still trips `PromptDrift`, and that is the safety net for the fallback.
- **`bench/manifest.json` with `bench/tasks/` and `bench/corpora/`.** The
  repo-relative path strings live *inside* the JSON whose sha256 is pinned. There
  is no move that is both correct and pin-preserving.
- ~~**`src/rlm/` at exactly two levels below the root.**~~ **No longer true, and it
  was a real defect.** Both files computed `REPO_ROOT` by counting parents, which in
  an INSTALLED package resolves to the venv's `Lib` directory — and `LEDGER_PATH` and
  `DEFAULT_REPORT_PATH` are built from it, so `rlm bench` would have written into a
  consumer's site-packages and reported success. Both now call
  `rlm.config.find_repo_root()`, which walks up for `bench/` + `pyproject.toml` and
  returns a `NO_REPO` sentinel that names the reason in its own path. Depth is free.
- **`src/rlm/trace/schema.sql` beside `src/rlm/trace/store.py`** — a `with_name()`
  sibling lookup. Still not *declared* package data, and verified 2026-09-01 that it
  does not need to be: hatchling includes it because it sits inside the package, and
  a built wheel carries all 22 non-`.py` files (this, `config.default.yaml`, the 18
  prompts, two fixtures). `tests/verify_distribution.py` checks that on demand.
- **`.gitattributes`** — `*.md text eol=lf` and `bench/corpora/** -text` are what
  make every content hash reproducible across clones. Changing it breaks all 13
  prompt pins and 24 corpus hashes on the next fresh clone, invisibly on the
  machine where the change was made.
- **`traces/rlm.duckdb` together with `traces/blobs/`** — 67,136 refs are stored
  relative to `blob_root`. Separating them breaks all of them at once.

Two silent ones, stated plainly because they produce no error:

- **Renaming `src/rlm/serve/dispatcher.py`, `budget.py`, `serverproc.py` or `trace.py`**
  is safe *today* only because `bench/corpora/code-bundle.txt` is frozen text and
  the seven codeqa answers are frozen strings. Never rebuild the v1 corpus — see
  the comment in `bench/build.py`.
- **`sandbox_bootstrap/`'s ACL.** The directory holds a one-time
  `icacls /grant *S-1-15-2-1` grant and its absolute path is in `config.yaml`.
  Relocating it makes the AppContainer child unable to read its own script, with no
  obvious cause.

## This checkout cannot be relocated

The absolute path `D:\PROJECTS\rlm-halo-framework` is baked into both config
files and all 614 episode snapshots. Moving the checkout produces silent
`FileNotFoundError`s with no fallback. This is a known, accepted constraint,
not an oversight — the snapshots are a record of runs that already happened,
and rewriting them would be rewriting the record.

## Two renames on 2026-08-22, and what was swept

`rlm/` → `src/rlm/`, and `s0/ s1/ s2/ s3/ s4/` → `milestones/` — which was then
deleted from the working tree entirely on 2026-08-22, along with the external
`D:/ARCHIVE`. `upstream/` went to `milestones/` too and came back out before the
deletion: it is a third-party bug report, not a gate.

**Milestone paths were swept everywhere** — 936 repo-relative references across
220 files, 70 absolute paths, 41 depth anchors (`parents[N]` → `parents[N+1]`,
applied only where the anchor resolved to the repo root; the 13 that resolve
inside `s2/` were left alone because they survive the move), and 32 path-component
constructions (`REPO_ROOT / "s4"`). Every resulting `milestones/…` path was then
checked to exist on disk.

**`rlm/` paths in the LIVE docs were swept** when `src/rlm/` was grouped into
`context/ serve/ trace/ measure/` — this file, `ARCHITECTURE.md`, `config.yaml`
and `docs/README.md`. The map was built from what is actually on disk, so a
citation only moves if its target does, and every remaining one was checked to
resolve.

Three classes were excluded, one of them a functional break:

- the sandbox **destination** strings in `src/rlm/sandbox/manager.py:101-103`.
  They describe the layout *inside* `sandbox_bootstrap/`, where the AppContainer
  child imports `rlm.bridge` — they must stay `rlm/`, and that is why
  `bridge.py` did not move with the rest.
- citations to the upstream `alexzhang13/rlm` harness, a different project.
- `rlm/benchserve.py`, a module that was considered and never built.

**Historical documents were left alone**: `CHANGELOG.md`, `docs/research/` and
`docs/superpowers/`. They describe where a file was when the entry was written,
and rewriting a dated record to match today's tree falsifies it.
