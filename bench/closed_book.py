"""§8 precondition 1: the closed-book probe.

    "**Closed-book probe:** every benchmark question runs context-free against
     both the root and leaf models, 3 seeds each; any task answered correctly in
     >=1/3 seeds without the corpus is rewritten or replaced (memorized answers
     differentially inflate the single-shot arms -- B1/B3 lean hardest on
     parametric knowledge)."

The asymmetry in that parenthesis is the reason it is a precondition and not a
nicety. A memorised answer does not inflate every arm equally: B1 and B3 answer
in ONE call from parametric knowledge, while RLM and B2 spend hundreds of leaf
calls reading a corpus they did not need. A benchmark carrying such a task
would report the scaffold losing on cost while tying on quality, and the
write-up would read as a clean negative result.

Both models, three seeds, no corpus. Graded with the task's OWN checker, so a
question is only "answered" here in the same sense S4 would score it.

    uv run --python 3.12 --no-project python -m bench.closed_book
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from bench.manifest import BenchmarkManifest
from rlm.measure.checkers import check
from rlm.episode import Task

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "bench" / "manifest.json"
SEEDS = (1, 2, 3)

# Deliberately generous framing: the probe should give the model its BEST shot
# at answering from memory, because a false negative here ships a contaminated
# task into the frozen benchmark.
SYSTEM = ("You are answering from memory. You have no document. If you know the "
          "answer, give it exactly and nothing else. If you do not know, reply "
          "NONE.")


def post(base: str, path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ask(base: str, question: str, seed: int, n_predict: int) -> str:
    rendered = post(base, "/apply-template", {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": question}],
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    r = post(base, "/completion", {"prompt": rendered, "n_predict": n_predict,
                                   "temperature": 0.0, "top_p": 0.8,
                                   "seed": seed, "cache_prompt": False})
    raw = r.get("content") or ""
    return raw.rsplit("</think>", 1)[-1].strip()


def main(args: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST,
                    help="path to benchmark manifest (default: %(default)s)")
    ap.add_argument("--root", default="http://127.0.0.1:8080")
    ap.add_argument("--leaf", default="http://127.0.0.1:8081")
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--dry-run", action="store_true",
                    help="read manifest and list tasks without calling any server")
    a = ap.parse_args(args)

    m = BenchmarkManifest.load(a.manifest)

    if a.dry_run:
        # Dry-run: just list tasks without calling any server
        try:
            manifest_display = a.manifest.relative_to(REPO)
        except ValueError:
            manifest_display = a.manifest
        print(f"closed-book probe (dry-run): {len(m.tasks)} tasks from "
              f"{manifest_display}\n")
        for e in m.tasks:
            print(f"  {e.task_id:<12} {e.category:<12}")
        return 0

    print(f"closed-book probe: {len(m.tasks)} tasks x 2 models x {len(SEEDS)} "
          f"seeds = {len(m.tasks) * 2 * len(SEEDS)} calls\n")

    contaminated: list[str] = []
    for e in m.tasks:
        t = Task.from_file(REPO / e.task_file)
        hits = 0
        detail: list[dict] = []
        for role, base in (("root", a.root), ("leaf", a.leaf)):
            for seed in SEEDS:
                got = ask(base, t.text, seed, a.n_predict)
                ok = check(t.checker, got, t.answer)
                hits += ok
                detail.append({"role": role, "seed": seed, "passed": ok,
                               "answer": got[:120]})
        e.closed_book = {"seeds": len(SEEDS), "models": ["root", "leaf"],
                         "passed_without_corpus": hits, "detail": detail}
        flag = ""
        if hits:
            contaminated.append(e.task_id)
            flag = "   <== ANSWERED WITHOUT THE CORPUS"
        print(f"  {e.task_id:<12} {e.category:<12} {hits}/6{flag}")

    m.write(a.manifest)
    try:
        manifest_display = a.manifest.relative_to(REPO)
    except ValueError:
        manifest_display = a.manifest
    print(f"\nwrote probe results into {manifest_display}")
    if contaminated:
        print(f"\n§8 REQUIRES REWRITE OR REPLACEMENT of: {contaminated}")
        print("The freeze is blocked until they are regenerated.")
        return 1
    m.validate(require_closed_book=True)
    print("\nNO TASK is answerable without its corpus. §8 precondition 1 PASSES,")
    print("and the manifest now validates with require_closed_book=True.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
