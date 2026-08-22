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
| `tests/` | **live code** | 958 tests. `test_import_rules.py` enforces the C1–C6 dependency rule as a checked invariant. |
| `prompts/` | **frozen artifact** | 18 sha-pinned prompt files. See "before you move anything". |
| `s0/ s1/ s2/ s3/ s4/` | **evidence record** *(and `s1`/`s2` are also live packages — see below)* | One directory per milestone gate. Each `RESULTS.md` is the evidence behind a published verdict. |
| `upstream/` | **evidence record** | llama.cpp defect reproducers and issue drafts (R13 slot leak, R14 continuous batching). Ahead of the `s2/` docs it derives from. |
| `docs/` | **live documents** | Research, specs, plans. Conventions in [`docs/README.md`](docs/README.md). |
| `traces/` | **gitignored, and irreplaceable** | The trace store: DuckDB + 67,136 blobs. Under I4 this is the *sole* episode truth. Not in git. Not reconstructible. |
| `tools/` | **gitignored, machine-bound** | Three live llama.cpp builds. Not in git, ~2.3 GB. |
| `sandbox_bootstrap/` | **gitignored, generated, load-bearing** | Regenerated from `src/rlm/` every run — but the *directory* carries a one-time ACL grant. Delete it freely; never relocate it. |

## Three things that cost the most to rediscover

1. **`s1/` and `s2/` are importable Python packages the build and the test suite
   depend on.** `bench/build.py` and `bench/vocab.py` import them at module import
   time, and seven test modules import them directly — so moving either makes
   `pytest` fail at *collection*. They look like finished milestones. They are not
   only that.
2. **`traces/` is gitignored *and* is the sole episode truth.** Both halves matter:
   git will not save you, and nothing else holds the record.
3. **`tools/llamacpp-vulkan-dflash2/` has no zip and cannot be re-downloaded.** It
   is the only build that loads the DFlash2 drafter, and its source is an unmerged
   llama.cpp PR at commit `5ecbe1a` that can be force-pushed. Back it up before you
   touch `tools/`.

## Commands

```bash
uv run pytest -q                              # 958 tests, ~16 min
uv run rlm validate --no-server-probe         # config + prompt pins + sandbox confinement
uv run rlm replay <episode-id>                # re-derive an episode from the store alone (I4)
uv run rlm bench --smoke                      # the only path that launches the root server
python -c "from pathlib import Path; from rlm.config import load_config; load_config(Path('config.yaml'))"
```

Always run from the repository root: nothing resolves paths against the config
file's own directory.

## Before you move anything

This repo pins file *contents* by sha256 into every episode's `config_snapshot`,
and `rlm replay` is the gate that proves episode state is re-derivable from the
trace store alone. Several things are therefore frozen, and two of them fail
**silently**.

- **`prompts/**` — all 18 files, including versions that look superseded.** 614
  episode snapshots store the path, and the registry is rebuilt from the
  *snapshot's* path, not the live config. Hashes cover content only, so moving a
  byte-identical file still breaks it. `cp` is safe; `mv` is not. An edit — even to
  a changelog comment the loader strips — trips `PromptDrift`.
- **`bench/manifest.json` with `bench/tasks/` and `bench/corpora/`.** The
  repo-relative path strings live *inside* the JSON whose sha256 is pinned. There
  is no move that is both correct and pin-preserving.
- **`src/rlm/` at exactly two levels below the root.** `src/rlm/bench.py` and
  `src/rlm/cli.py` define `REPO_ROOT = parents[2]`.
- **`src/rlm/schema.sql` beside `src/rlm/trace.py`** — a `with_name()` sibling
  lookup, not declared package data.
- **`s4/RESULTS.md` and `s4/results/ledger.jsonl`** — hardcoded paths with two
  deliberate tripwire tests. Move them and `rlm bench` silently starts a *fresh*
  ledger instead of resuming.
- **`.gitattributes`** — `*.md text eol=lf` and `bench/corpora/** -text` are what
  make every content hash reproducible across clones. Changing it breaks all 13
  prompt pins and 24 corpus hashes on the next fresh clone, invisibly on the
  machine where the change was made.
- **`traces/rlm.duckdb` together with `traces/blobs/`** — 67,136 refs are stored
  relative to `blob_root`. Separating them breaks all of them at once.

Two silent ones, stated plainly because they produce no error:

- **Renaming `src/rlm/dispatcher.py`, `budget.py`, `serverproc.py` or `trace.py`**
  is safe *today* only because `bench/corpora/code-bundle.txt` is frozen text and
  the seven codeqa answers are frozen strings. Never rebuild the v1 corpus — see
  the comment in `bench/build.py`.
- **`sandbox_bootstrap/`'s ACL.** The directory holds a one-time
  `icacls /grant *S-1-15-2-1` grant and its absolute path is in `config.yaml`.
  Relocating it makes the AppContainer child unable to read its own script, with no
  obvious cause.

## This checkout cannot be relocated

The absolute path `D:\PROJECTS\rlm-halo-framework` is baked into 63+ scripts under
`s1/` and `s2/`, both config files, and all 614 episode snapshots. Moving the
checkout produces silent `FileNotFoundError`s with no fallback. This is a known,
accepted constraint, not an oversight: those scripts have already produced their
committed output, and making them portable is a project with its own testing
burden and no current payoff.

## Note on paths in prose

The package moved from `rlm/` to `src/rlm/` on 2026-08-22. Citations written
before that date — in `ARCHITECTURE.md`, in `s0/`–`s4/`, in `upstream/`, and in
docstrings — spell the old path. They were not swept: the milestone documents are
frozen measurement records that correctly describe where a file was when it was
measured, and a mechanical rewrite would also have corrupted the sandbox
*destination* strings in `src/rlm/sandbox/manager.py` and the citations to the
upstream `alexzhang13/rlm` harness, which are a different project. There is
exactly one `episode.py`; read `rlm/x.py` in prose as `src/rlm/x.py`.
