"""Build the v2 benchmark: 16 shapes, in two frozen streams plus a
regenerable practice stream, gated on both build-time adversaries.

    uv run --python 3.12 --no-project python -m bench.build_v2 --leaf-port 8081
    uv run --python 3.12 --no-project python -m bench.build_v2 --leaf-port 8081 --stream train
    uv run --python 3.12 --no-project python -m bench.build_v2 --practice --seed 3

spec §3 pre-registers the 16 shapes: 6 linear-semantic (`ls-01..06`), 6
interactive (`int-01..06`), 4 code-solvable controls (`ctl-01..04`). §14
requires the whole set built in one pass, `env` verb included -- there is no
staged build.

TWO ADVERSARIES, BOTH BUILD-TIME GATES (`bench/adversary.py`). Every
linear-semantic and interactive task is run through `parser_adversary` before
it is ever written: a task whose answer a deterministic field-parser can
reproduce is refused outright (`SystemExit`), never merely flagged, because
§14's whole argument is that these categories force delegation ONLY if no
program shortcut exists. `self_read_adversary`'s window count is recorded
(`min_windows`) rather than gating the build -- `--small` test corpora
legitimately fall at or under the k=40 floor, and `BenchmarkManifest.
validate(strict_adversaries=True)` is where that number is actually enforced,
at freeze.

THE CONTROLS ARE THE OTHER HALF OF THE COMPARISON. `ctl-01..04` are built
from `bench.corpus_v2.build_control_count`/`build_control_needle` --
deliberately regex-solvable, unlike every other v2 shape -- and this module
PROVES it by re-running each control's own `.regex` against its own `.text`
and asserting the result IS `.answer`, before recording `regex_solvable:
True`. A control the intended regex doesn't solve is a broken control, not a
harder one.

TWO STREAMS, ONE SHAPE SET. `train` and `held_out` carry the identical 16
shapes at different seeds (`base + 1000 * stream_index`) -- §14's design:
the benchmark is scored on `train` (`rules.scored_stream`) and `held_out`
exists so a later run can check the same shapes generalise, not just the
same seeds. `--practice` writes a third copy (`stream: "practice"`) OUTSIDE
`bench/` entirely (`runs/practice/<seed>/`, gitignored) with no manifest --
it exists so a shape can be read/debugged without touching the frozen tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from bench import corpus_v2 as bc2
from bench.adversary import parser_adversary, self_read_adversary
from bench.build import INJECTION
from bench.manifest import BenchmarkManifest, TaskEntry
from bench.tokens import approx_tokens
from rlm.context.chunker import ChunkConfig

REPO = Path(__file__).resolve().parents[1]
CORPORA = REPO / "bench" / "corpora" / "v2"
TASKS = REPO / "bench" / "tasks" / "v2"
MANIFEST = REPO / "bench" / "manifest.v2.json"

# Same precondition as v1's builder (`bench/build.py`): every v2 corpus is
# generated today from a vocabulary that did not exist before today.
ASSUMED_CUTOFF = "2025-12-31"

# spec §14: static (linear-semantic + control) corpora at ~60,000 tokens
# (three times the root's ~20,000-token reading ceiling); the interactive
# corpus behind `env` at ~200,000 (ten times it, since navigation there costs
# operations, not context). `--small` shrinks both for a build a test can
# afford, and is a test-only switch -- the real build never passes it.
STATIC_TOKENS = 60_000
INTERACTIVE_TOKENS = 200_000
SMALL_STATIC_TOKENS = 6_000
SMALL_INTERACTIVE_TOKENS = 20_000

# `test_build_v2_refuses_a_task_a_parser_can_solve` flips this off to disable
# the register's own defence (Task 16's paraphrase step) and prove the
# parser-adversary gate actually fires, rather than trusting that it would.
PARAPHRASE = True

# The production `scaffold.chunk` geometry (`checks/test_chunker.py`'s
# `PRODUCTION`, `bench/corpus_v2.py`'s `_INTERACTIVE_CHUNK_CFG_ARGS`, and
# `checks/test_adversary.py`'s `CFG` all agree on these five numbers) -- the
# same window shape `min_windows` is measured against everywhere else.
CHUNK_CFG = ChunkConfig(size_tokens=640, overhead_tokens=1920,
                        snap_to_boundary=True, snap_tolerance=0.10,
                        stride_tokens=480)

STREAM_INDEX = {"train": 0, "held_out": 1}

#: The manifest's `rules` block, verbatim from the spec §14 amendment (task
#: brief, 2026-09-02): the RLM arm and its two baselines, the margin and
#: escalation band a verdict is read against, B2's pre-registered abstention
#: from the interactive category (a fixed map-reduce pipeline has no static
#: corpus to partition behind `env`), and the stream a verdict is scored on.
RULES = {"rlm_arm": "rlm", "baselines": ["rlm-nosubcalls", "b2"], "margin": 2,
        "escalation_band": [1, 2], "tripwire_floor": 3,
        "abstentions": {"b2": ["interactive"]}, "scored_stream": "train",
        "n_tasks": 16}

#: One base seed per shape id; the actual per-stream seed is
#: `base + 1000 * stream_index` (train=0, held_out=1; a practice build's
#: `--seed N` plays the same role as `stream_index`).
BASE_SEEDS = {
    "ls-01": 9401, "ls-02": 9402, "ls-03": 9403,
    # ls-04 RESEEDED 2026-09-03 (Task 22 freeze): 9404 -> 9414. The closed-book
    # probe found ls-04-held_out (seed 9404+1000=10404) answered 6/6 WITHOUT
    # the corpus -- contamination, not an adversary refusal. Per the runbook's
    # binding rule the answer is never edited; the shape's seed moves and the
    # whole build reruns. Original seed retained here in this comment, not the
    # dict, so a future reader does not accidentally revert it.
    "ls-04": 9414, "ls-05": 9405, "ls-06": 9406,
    "int-01": 9501, "int-02": 9502, "int-03": 9503,
    "int-04": 9504, "int-05": 9505, "int-06": 9506,
    "ctl-01": 9601, "ctl-02": 9602, "ctl-03": 9603, "ctl-04": 9604,
}

_DOC_HEADER_RE = re.compile(r"=== DOCUMENT [^\n]*===\n")
_RECORD_ID_RE = re.compile(r"\[(ENT-\S+)\]")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _context_rel(tasks_dir: Path, corpora_dir: Path, task_id: str) -> str:
    """`context_path` written into a task JSON, relative to `tasks_dir` (where
    `episode.py` resolves it from -- `p.parent / context_path`), computed from
    the ACTUAL directories in play rather than assumed. v1's build.py could
    hardcode `"../corpora/{id}.txt"` because `bench/tasks/` and `bench/corpora/`
    are siblings; v2 nests both one level deeper under `v2/`, and a
    `--practice` build's `tasks_dir`/`corpora_dir` are siblings again under
    `--out`. Computed with `os.path.relpath` so every caller (real build,
    `--practice`) gets the correct depth instead of a copy-pasted literal --
    this is the fix for the 2026-09-03 smoke's `config_refused` defect, where
    every real v2 task file resolved to the nonexistent `bench/tasks/corpora/`."""
    target = corpora_dir / f"{task_id}.txt"
    return os.path.relpath(target, start=tasks_dir).replace("\\", "/")


def _task_file(tasks_dir: Path, task_id: str, text: str, corpus_rel: str,
              category: str, answer: str, checker: str) -> Path:
    """Exactly `Task.from_file`'s key set -- no more, or it refuses to load."""
    body = {"task_id": task_id, "text": text, "context_path": corpus_rel,
            "category": category, "answer": answer, "checker": checker}
    return _write(tasks_dir / f"{task_id}.json",
                  json.dumps(body, indent=2, ensure_ascii=False) + "\n")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relpath(p: Path) -> str:
    """repo-relative when possible (v1's convention); a practice build's
    `--out` can point outside the repo (a test's `tmp_path`), and a manifest
    is never written for that stream, so falling back to the raw path there
    costs nothing."""
    try:
        return str(p.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _entry(task_id, category, tasks_dir, corpus_path, corpus_text, question,
          checker, **kw) -> TaskEntry:
    return TaskEntry(
        task_id=task_id, category=category,
        task_file=_relpath(tasks_dir / f"{task_id}.json"),
        corpus_path=_relpath(corpus_path),
        corpus_sha256=_sha(corpus_text), corpus_tokens=kw.pop("tokens"),
        corpus_date=date.today().isoformat(), checker=checker,
        question_sha256=_sha(question), **kw)


def _inject(text: str) -> str:
    """The v1 `INJECTION` record (`bench/build.py`), spliced into the first
    record boundary of an interactive corpus's first document -- anchored on
    `"\\n\\n[ENT-"` rather than the first blank line generally, because the
    first blank line in an interactive corpus's `.text` separates the
    question from the first `=== DOCUMENT ===` header, not two records."""
    marker = "\n\n[ENT-"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("interactive corpus has no record boundary to inject into")
    return text[:idx] + "\n" + INJECTION + text[idx:]


def _interactive_adversary_view(c) -> SimpleNamespace:
    """Adapt an `InteractiveCorpus` into the flat, header-free shape
    `parser_adversary`/`self_read_adversary` expect (they were built against
    `LinearSemanticCorpus`'s `.text`/`.labels`/`.record_ids`). Stripping every
    `=== DOCUMENT ... ===` header leaves exactly a `LinearSemanticCorpus`'s
    own grammar -- the question, then every document's records back to back,
    each pair separated by one blank line (see the newline-collapsing note
    below for why that needs a second pass, not just header removal). Must
    run BEFORE `_inject`: `INJECTION` also opens with `[ENT-999999]` but does not
    carry a `Filed:`/`Query:`/`Notes:` body, so `adversary._parse_queries`
    would fail to parse it as a record.
    """
    # Each document's block is `header + body`, joined to its neighbours by
    # one `"\n\n"` -- but the header's OWN trailing "\n" sits inside that
    # block, so removing the header substring (everywhere but the first
    # document, whose header follows the question's own `sep` instead)
    # leaves THREE newlines at every inter-document boundary, not the two a
    # blank-line record separator needs. Collapsing any run of 3+ back to
    # exactly 2 restores the flat, uniformly blank-line-separated grammar --
    # a real 2-newline gap (every other record boundary) is untouched.
    flat_text = re.sub(r"\n{3,}", "\n\n", _DOC_HEADER_RE.sub("", c.text))
    labels = [item.label for d in c.doc_ids for item in c.items_by_doc[d]]
    record_ids = _RECORD_ID_RE.findall(flat_text)
    assert len(record_ids) == len(labels), (
        f"{len(record_ids)} parsed records vs {len(labels)} labels in the "
        f"interactive adversary view")
    return SimpleNamespace(text=flat_text, labels=labels, record_ids=record_ids)


def _refuse_if_beaten(task_id: str, scores: dict[str, float]) -> None:
    chance = scores["__chance__"]
    beaten = {k: v for k, v in scores.items()
             if k != "__chance__" and v > chance + 0.02}
    if beaten:
        raise SystemExit(f"parser adversary beat chance on {task_id}: {beaten}")


# --------------------------------------------------------------------------- #


def build_linear_semantic(count, counter_name, *, static_tokens,
                          interactive_tokens, stream, stream_index,
                          tasks_dir, corpora_dir) -> list[TaskEntry]:
    """6 tasks, fixed order: `count_label` x3, `most_common_label` x1,
    `count_two_labels` x2 (`ls-05`/`ls-06` -- spec §3's "Pairs-shaped" flag)."""
    items = bc2.load_trec()
    label_source = bc2.label_source_id()
    kinds = ["count_label", "count_label", "count_label", "most_common_label",
             "count_two_labels", "count_two_labels"]
    out: list[TaskEntry] = []
    for i, kind in enumerate(kinds):
        shape_id = f"ls-{i + 1:02d}"
        seed = BASE_SEEDS[shape_id] + 1000 * stream_index
        c = bc2.build_linear_semantic(seed, static_tokens, count, counter_name,
                                      question_kind=kind, items=items,
                                      paraphrase=PARAPHRASE)
        task_id = f"{shape_id}-{stream}"
        scores = parser_adversary(c)
        _refuse_if_beaten(task_id, scores)
        min_windows = self_read_adversary(c, CHUNK_CFG, count, k=40)

        question, _, _ = c.text.partition("\n\n")
        cp = _write(corpora_dir / f"{task_id}.txt", c.text)
        _task_file(tasks_dir, task_id, question, _context_rel(tasks_dir, corpora_dir, task_id),
                  "linear_semantic", c.answer, c.checker)

        windows = math.ceil(c.measured_tokens / 432)
        out.append(_entry(
            task_id, "linear_semantic", tasks_dir, cp, c.text, question,
            c.checker, tokens=c.measured_tokens, stream=stream,
            shape_id=shape_id, min_windows=min_windows, regex_at_chance=scores,
            regex_solvable=False, label_source=label_source, windows=windows,
            subcalls=2 * windows))
    return out


def build_interactive(count, counter_name, *, static_tokens, interactive_tokens,
                      stream, stream_index, tasks_dir, corpora_dir) -> list[TaskEntry]:
    """6 tasks, fixed order: `count_label_across_docs` x3,
    `which_doc_has_most` x1, `pairs_docs_sharing_label_majority` x2
    (`int-05`/`int-06` -- the same "Pairs-shaped" flag as `ls-05`/`ls-06`,
    and the two that also carry R12's adversarial injection)."""
    items = bc2.load_trec()
    label_source = bc2.label_source_id()
    kinds = ["count_label_across_docs", "count_label_across_docs",
             "count_label_across_docs", "which_doc_has_most",
             "pairs_docs_sharing_label_majority",
             "pairs_docs_sharing_label_majority"]
    out: list[TaskEntry] = []
    for i, kind in enumerate(kinds):
        shape_id = f"int-{i + 1:02d}"
        seed = BASE_SEEDS[shape_id] + 1000 * stream_index
        c = bc2.build_interactive(seed, interactive_tokens, count, counter_name,
                                  question_kind=kind, items=items)
        task_id = f"{shape_id}-{stream}"

        view = _interactive_adversary_view(c)
        scores = parser_adversary(view)
        _refuse_if_beaten(task_id, scores)
        min_windows = self_read_adversary(view, CHUNK_CFG, count, k=40)

        adversarial = i >= 4                          # int-05, int-06
        text = _inject(c.text) if adversarial else c.text
        question, _, _ = c.text.partition("\n\n")
        cp = _write(corpora_dir / f"{task_id}.txt", text)
        _task_file(tasks_dir, task_id, question, _context_rel(tasks_dir, corpora_dir, task_id),
                  "interactive", c.answer, c.checker)

        out.append(_entry(
            task_id, "interactive", tasks_dir, cp, text, question, c.checker,
            tokens=c.measured_tokens, stream=stream, shape_id=shape_id,
            interactive=True, min_windows=min_windows,
            reference_actions=c.reference_actions, regex_at_chance=scores,
            regex_solvable=False, label_source=label_source,
            adversarial=adversarial))
    return out


def build_code_solvable(count, counter_name, *, static_tokens,
                        interactive_tokens, stream, stream_index, tasks_dir,
                        corpora_dir) -> list[TaskEntry]:
    """4 tasks, fixed order: `ctl-01`/`ctl-02` = regex-count over the Filed
    line's coined month; `ctl-03`/`ctl-04` = needle (`build_needle`'s pattern
    re-implemented on the v2 register). Every control's own intended regex is
    re-run here, against its own rendered text, and its result is asserted
    to equal `.answer` -- proof, not assumption, that `regex_solvable: True`
    is true."""
    items = bc2.load_trec()
    label_source = bc2.label_source_id()
    shapes = [("ctl-01", bc2.build_control_count, "int_exact"),
             ("ctl-02", bc2.build_control_count, "int_exact"),
             ("ctl-03", bc2.build_control_needle, "uuid_exact"),
             ("ctl-04", bc2.build_control_needle, "uuid_exact")]
    out: list[TaskEntry] = []
    for shape_id, builder, checker in shapes:
        seed = BASE_SEEDS[shape_id] + 1000 * stream_index
        c = builder(seed, static_tokens, count, counter_name, items,
                   paraphrase=PARAPHRASE)
        if checker == "int_exact":
            solved = str(len(re.findall(c.regex, c.text)))
        else:
            m = re.search(c.regex, c.text)
            solved = m.group(1) if m else None
        assert solved == c.answer, (
            f"{shape_id}: the intended regex did not reproduce the answer "
            f"({solved!r} != {c.answer!r}) -- this control is broken, not solvable")

        task_id = f"{shape_id}-{stream}"
        cp = _write(corpora_dir / f"{task_id}.txt", c.text)
        _task_file(tasks_dir, task_id, c.question, _context_rel(tasks_dir, corpora_dir, task_id),
                  "code_solvable", c.answer, checker)

        out.append(_entry(
            task_id, "code_solvable", tasks_dir, cp, c.text, c.question,
            checker, tokens=c.measured_tokens, stream=stream,
            shape_id=shape_id, regex_solvable=True, label_source=label_source))
    return out


CATEGORY_BUILDERS = (
    ("linear_semantic", build_linear_semantic),
    ("interactive", build_interactive),
    ("code_solvable", build_code_solvable),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaf-port", type=int, default=None,
                    help="count tokens with the real leaf tokenizer; required "
                         "for a build that may be frozen")
    ap.add_argument("--stream", choices=["train", "held_out", "both"],
                    default="both")
    ap.add_argument("--small", action="store_true",
                    help="test-only size switch: 6K static / 20K interactive "
                         "tokens. The real build never passes this.")
    ap.add_argument("--practice", action="store_true",
                    help="write a third, regenerable copy of all 16 shapes "
                         "outside bench/, with no manifest write")
    ap.add_argument("--seed", type=int, default=0,
                    help="practice stream's per-shape seed offset")
    ap.add_argument("--out", type=Path, default=None,
                    help="practice stream's output directory "
                         "(default runs/practice/<seed>/)")
    a = ap.parse_args(argv)

    if a.leaf_port:
        from bench.tokens import leaf_counter
        count, counter_name = leaf_counter(a.leaf_port, 120), "leaf:/tokenize"
    else:
        count, counter_name = approx_tokens, "approx-offline"
    print(f"token counter: {counter_name}")

    static_tokens = SMALL_STATIC_TOKENS if a.small else STATIC_TOKENS
    interactive_tokens = SMALL_INTERACTIVE_TOKENS if a.small else INTERACTIVE_TOKENS

    if a.practice:
        out = a.out or (REPO / "runs" / "practice" / str(a.seed))
        tasks_dir, corpora_dir = out / "tasks", out / "corpora"
        entries: list[TaskEntry] = []
        for label, fn in CATEGORY_BUILDERS:
            got = fn(count, counter_name, static_tokens=static_tokens,
                     interactive_tokens=interactive_tokens, stream="practice",
                     stream_index=a.seed, tasks_dir=tasks_dir,
                     corpora_dir=corpora_dir)
            entries += got
        print(f"practice stream ({len(entries)} tasks) written to {tasks_dir}")
        for e in entries:
            print(f"  {e.task_id}")
        return 0

    streams = ["train", "held_out"] if a.stream == "both" else [a.stream]
    entries = []
    for stream in streams:
        stream_index = STREAM_INDEX[stream]
        for label, fn in CATEGORY_BUILDERS:
            got = fn(count, counter_name, static_tokens=static_tokens,
                     interactive_tokens=interactive_tokens, stream=stream,
                     stream_index=stream_index, tasks_dir=TASKS,
                     corpora_dir=CORPORA)
            print(f"  {stream:<10} {label:<16} {len(got)} tasks")
            entries += got

    if a.stream != "both":
        # A single stream can never satisfy v2's rules on its own (`held_out`
        # and `train` must carry identical shape sets, and `rules.n_tasks`
        # counts the SCORED stream alone but the category-count/shape-set
        # clauses need both) -- §14's build is one pass over both streams,
        # not staged. `--stream train|held_out` writes that stream's task and
        # corpus files (useful to inspect or rebuild one stream in isolation)
        # but stops short of a manifest that would fail its own validation
        # the moment it was loaded.
        print(f"\n{len(entries)} tasks written for stream {a.stream!r} "
             f"(no manifest -- pass --stream both to freeze)")
        return 0

    m = BenchmarkManifest(
        benchmark_version="v2" if a.leaf_port else "v2-draft",
        built_at=date.today().isoformat(), token_counter=counter_name,
        assumed_training_cutoff=ASSUMED_CUTOFF, tasks=entries,
        rules=dict(RULES))
    m.validate(require_closed_book=False, strict_adversaries=not a.small)
    m.write(MANIFEST)
    print(f"\n{len(entries)} tasks, manifest sha256 {m.sha256[:16]}...")
    print(f"wrote {MANIFEST.relative_to(REPO) if MANIFEST.is_relative_to(REPO) else MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
