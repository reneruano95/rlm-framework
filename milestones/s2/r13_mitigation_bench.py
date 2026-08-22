"""Price the R13 mitigations. Measurement only -- no hypothesis in here.

Modes
-----
--pattern      one chunk, N questions, on one slot, cache_prompt on vs off.
               This is the number that prices mitigation 1 (`cache_prompt:false`).
--throughput   cold prefill and decode t/s at a given window size and k
               concurrent streams. Prices mitigations 2/3 and the -np sweep.
--windows      contamination test at PRODUCTION geometry (1024-token windows):
               distinct windows down one shared slot vs one virgin slot each,
               enumeration probe (the most sensitive probe per R13 section 5).

Reuses the transport and the leak oracle from r13_repro.py so the verdicts are
produced by exactly the same code that produced R13.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s2.r13_repro import (  # noqa: E402
    Call,
    Server,
    compose_enum,
    foreign_strings,
    load_fixtures,
)

FIX = Path(__file__).resolve().parent / "fixtures"


def _emit(path: Path, calls: list[Call], tag: str, base: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for c in calls:
            rec = asdict(c)
            rec["label_tag"] = tag
            rec["base_url"] = base
            rec["ts"] = time.time()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def tail_window(srv: Server, text: str, target_tokens: int) -> str:
    """Last ~target_tokens tokens of `text`, by measured tokenisation."""
    lines = text.splitlines()
    lo, hi = 1, len(lines)
    best = "\n".join(lines[-1:])
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = "\n".join(lines[-mid:])
        n = srv.tokenize(cand)
        if n <= target_tokens:
            best, lo = cand, mid + 1
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------- pattern


def mode_pattern(srv: Server, args: argparse.Namespace) -> dict[str, Any]:
    """One chunk, N questions. cache_prompt true vs false, same pinned slot."""
    manifest, texts = load_fixtures(FIX)
    out: dict[str, Any] = {"mode": "pattern", "cells": {}}
    calls: list[Call] = []

    for cell in args.cells.split(","):
        spec = manifest["cells"][cell]
        chunk = texts[cell]
        qs = [spec["questions"][k]["question"] for k in ("literal", "paraphrase", "absent")]
        qs = (qs * 4)[: args.questions]
        cell_out: dict[str, Any] = {"doc_tokens": spec["measured_tokens"], "arms": {}}

        for arm, cache in (("cache_prompt_true", True), ("cache_prompt_false", False)):
            slot = args.slot if arm == "cache_prompt_true" else args.slot + 1
            per: list[dict[str, Any]] = []
            for qi, q in enumerate(qs):
                user = f"DOCUMENT:\n{chunk}\n\nQUESTION: {q}"
                prompt = srv.render(user)
                data, wall = srv.completion(
                    prompt, id_slot=slot, cache_prompt=cache,
                    seed=1, n_predict=args.n_predict, temperature=0.0,
                )
                tm = data.get("timings", {}) or {}
                c = Call(
                    condition=f"pattern:{arm}", trial=qi + 1, label=cell,
                    doc_tokens=spec["measured_tokens"], id_slot_requested=slot,
                    cache_prompt=cache, seed=1,
                )
                c.raw_output = data.get("content", "")
                c.slot_id = data.get("id_slot")
                c.prompt_n, c.cache_n = tm.get("prompt_n"), tm.get("cache_n")
                c.prompt_ms, c.predicted_ms = tm.get("prompt_ms"), tm.get("predicted_ms")
                c.predicted_n = tm.get("predicted_n")
                c.wall_s = round(wall, 4)
                calls.append(c)
                per.append({
                    "q": qi + 1, "wall_s": round(wall, 3),
                    "prompt_n": c.prompt_n, "cache_n": c.cache_n,
                    "prompt_ms": c.prompt_ms, "predicted_n": c.predicted_n,
                    "predicted_ms": c.predicted_ms,
                })
                print(f"  [{cell} {arm}] q{qi+1} wall={wall:.2f}s "
                      f"prompt_n={c.prompt_n} cache_n={c.cache_n} "
                      f"prompt_ms={c.prompt_ms}")
            cell_out["arms"][arm] = {
                "calls": per,
                "total_wall_s": round(sum(p["wall_s"] for p in per), 3),
                "first_wall_s": per[0]["wall_s"],
                "subsequent_wall_s": round(sum(p["wall_s"] for p in per[1:]), 3),
            }
        a = cell_out["arms"]["cache_prompt_true"]["total_wall_s"]
        b = cell_out["arms"]["cache_prompt_false"]["total_wall_s"]
        cell_out["penalty_s"] = round(b - a, 3)
        cell_out["penalty_x"] = round(b / a, 2) if a else None
        out["cells"][cell] = cell_out
        print(f"[pattern {cell}] cache=True {a}s  cache=False {b}s  "
              f"penalty {b-a:+.2f}s ({b/a:.2f}x)")

    _emit(Path(args.out), calls, args.label, srv.base_url)
    return out


# ------------------------------------------------------------- throughput


def mode_throughput(srv: Server, args: argparse.Namespace) -> dict[str, Any]:
    """Cold prefill + decode t/s. k streams fired simultaneously, each on its
    own virgin slot, each a distinct window -> no cache reuse anywhere."""
    _, texts = load_fixtures(FIX)
    cells = sorted(texts)
    windows = [tail_window(srv, texts[c], args.window) for c in cells]
    out: dict[str, Any] = {"mode": "throughput", "window": args.window, "k": {}}
    calls: list[Call] = []

    for k in [int(x) for x in args.kvals.split(",")]:
        # every call gets a slot that has held nothing, and a distinct doc
        jobs = []
        for i in range(k):
            w = windows[i % len(windows)]
            jobs.append((args.slot_base + i, compose_enum(w)))
        rendered = [srv.render(u) for _, u in jobs]

        def one(idx: int) -> tuple[dict[str, Any], float]:
            # ignore_eos pins the decode workload to exactly n_predict tokens
            # in every stream, so wall/window compares like with like.
            body = {
                "prompt": rendered[idx], "n_predict": args.n_predict,
                "temperature": 0.0, "seed": 7, "cache_prompt": False,
                "stream": False, "id_slot": jobs[idx][0],
                "ignore_eos": True,
            }
            t = time.monotonic()
            r = srv.client.post("/completion", json=body)
            w = time.monotonic() - t
            r.raise_for_status()
            return r.json(), w

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=k) as ex:
            res = list(ex.map(one, range(k)))
        wall = time.monotonic() - t0

        pn = sum((r[0].get("timings", {}) or {}).get("prompt_n", 0) for r in res)
        dn = sum((r[0].get("timings", {}) or {}).get("predicted_n", 0) for r in res)
        per_prefill = [
            (r[0]["timings"]["prompt_n"] / (r[0]["timings"]["prompt_ms"] / 1000.0))
            for r in res if (r[0].get("timings") or {}).get("prompt_ms")
        ]
        per_decode = [
            (r[0]["timings"]["predicted_n"] / (r[0]["timings"]["predicted_ms"] / 1000.0))
            for r in res if (r[0].get("timings") or {}).get("predicted_ms")
        ]
        for i, (data, w) in enumerate(res):
            tm = data.get("timings", {}) or {}
            c = Call(condition=f"throughput:k{k}", trial=i, label=f"w{args.window}",
                     doc_tokens=tm.get("prompt_n", 0), id_slot_requested=jobs[i][0],
                     cache_prompt=False, seed=7)
            c.slot_id = data.get("id_slot")
            c.prompt_n, c.cache_n = tm.get("prompt_n"), tm.get("cache_n")
            c.prompt_ms, c.predicted_ms = tm.get("prompt_ms"), tm.get("predicted_ms")
            c.predicted_n, c.wall_s = tm.get("predicted_n"), round(w, 4)
            c.raw_output = data.get("content", "")[:400]
            calls.append(c)

        out["k"][str(k)] = {
            "wall_s": round(wall, 3),
            "prompt_tokens": pn,
            "predicted_tokens": dn,
            "aggregate_prefill_tps": round(pn / wall, 1),
            "aggregate_total_tps": round((pn + dn) / wall, 1),
            "per_stream_prefill_tps_median": round(statistics.median(per_prefill), 1) if per_prefill else None,
            "per_stream_decode_tps_median": round(statistics.median(per_decode), 1) if per_decode else None,
            "wall_per_window_s": round(wall / k, 3),
        }
        print(f"[throughput w={args.window} k={k}] wall={wall:.2f}s "
              f"agg_prefill={pn/wall:.0f} t/s  per-stream prefill "
              f"{statistics.median(per_prefill):.0f} decode "
              f"{statistics.median(per_decode):.1f} t/s  "
              f"{wall/k:.3f} s/window")

    _emit(Path(args.out), calls, args.label, srv.base_url)
    return out


# ---------------------------------------------------------------- windows


def mode_windows(srv: Server, args: argparse.Namespace) -> dict[str, Any]:
    """Contamination at production geometry: N distinct ~1024-token windows.

    Arm A: all windows down ONE shared slot, in order.
    Arm B: each window on a slot that has held nothing else.
    Enumeration probe (R13 section 5: the most sensitive probe available).
    """
    _, texts = load_fixtures(FIX)
    cells = [c for c in args.cells.split(",") if c in texts]
    if args.window_ladder:
        sizes = [int(x) for x in args.window_ladder.split(",")]
        assert len(sizes) == len(cells), "one window size per cell"
        wins = {c: tail_window(srv, texts[c], s) for c, s in zip(cells, sizes)}
    else:
        wins = {c: tail_window(srv, texts[c], args.window) for c in cells}
    for c in cells:
        print(f"  window {c}: {srv.tokenize(wins[c])} tokens")

    calls: list[Call] = []
    res: dict[str, Any] = {"mode": "windows", "window": args.window,
                           "window_ladder": args.window_ladder,
                           "cache_prompt": args.cache_prompt,
                           "cells": cells, "arms": {}}

    for arm in args.arms.split(","):
        leaked = n = 0
        detail = []
        for i, c in enumerate(cells):
            for trial in range(1, args.trials + 1):
                slot = args.slot if arm == "shared_slot" else args.slot_base + i
                user = compose_enum(wins[c])
                prompt = srv.render(user)
                data, wall = srv.completion(
                    prompt, id_slot=slot, cache_prompt=args.cache_prompt, seed=trial,
                    n_predict=args.n_predict, temperature=args.temperature,
                )
                tm = data.get("timings", {}) or {}
                ans = data.get("content", "")
                # own = the FULL source chunk (stricter than the window alone),
                # so a string from this document's own head is never counted.
                fg = foreign_strings(ans, texts[c], texts, c)
                call = Call(condition=f"windows:{arm}", trial=trial, label=c,
                            doc_tokens=srv.tokenize(wins[c]), id_slot_requested=slot,
                            cache_prompt=args.cache_prompt, seed=trial)
                call.slot_id = data.get("id_slot")
                call.prompt_n, call.cache_n = tm.get("prompt_n"), tm.get("cache_n")
                call.prompt_ms, call.predicted_ms = tm.get("prompt_ms"), tm.get("predicted_ms")
                call.predicted_n, call.wall_s = tm.get("predicted_n"), round(wall, 4)
                call.raw_output = ans
                call.verdict = "FOREIGN" if fg else "OWN_OR_NONE"
                calls.append(call)
                n += 1
                if fg:
                    leaked += 1
                    detail.append({"cell": c, "trial": trial, "cache_n": call.cache_n,
                                   "foreign": fg, "output": ans[:220]})
                    print(f"  FOREIGN [{arm}] {c}/t{trial} cache_n={call.cache_n} {fg}")
            print(f"  [{arm}] {c} done ({leaked}/{n})")
        res["arms"][arm] = {"n": n, "leaked": leaked,
                            "rate": round(leaked / n, 3) if n else None,
                            "detail": detail}
        print(f"[windows {arm}] {leaked}/{n}")

    _emit(Path(args.out), calls, args.label, srv.base_url)
    return res



# ---------------------------------------------------------------- episode


def mode_episode(srv: Server, args: argparse.Namespace) -> dict[str, Any]:
    """A realistic episode: N windows x Q questions, end to end, wall-clock.

    Slot policy is the variable being priced:
      virgin   -- one never-reused slot per window; both questions on it
                  (same document, so the second question is warm and legal)
      shared   -- one pinned slot for the whole episode (what production does
                  today, and what `--parallel 1` forces)
    """
    _, texts = load_fixtures(FIX)
    cells = sorted(texts)
    # N distinct windows carved from the corpus, all at production geometry
    base = [tail_window(srv, texts[c], args.window) for c in cells]
    wins = [base[i % len(base)] for i in range(args.windows)]
    manifest, _ = load_fixtures(FIX)
    qs = [manifest["cells"]["s2-4096-p50"]["questions"][k]["question"]
          for k in ("literal", "paraphrase", "absent")]
    qs = (qs * 4)[: args.questions]

    prompts: list[list[str]] = []
    for w in wins:
        prompts.append([srv.render(f"DOCUMENT:\n{w}\n\nQUESTION: {q}") for q in qs])

    mismatches: list[tuple[int, int]] = []

    def do_window(i: int) -> float:
        slot = (args.slot_base + i) if args.policy == "virgin" else args.slot
        t = time.monotonic()
        for qi in range(len(qs)):
            body = {
                "prompt": prompts[i][qi], "n_predict": args.n_predict,
                "temperature": 0.0, "seed": 1,
                "cache_prompt": args.cache_prompt, "stream": False,
                "id_slot": slot,
            }
            if args.force_decode:
                body["ignore_eos"] = True
                body["n_predict"] = args.force_decode
            r = srv.client.post("/completion", json=body)
            r.raise_for_status()
            got = r.json().get("id_slot")
            if got != slot:
                mismatches.append((slot, got))
        return time.monotonic() - t

    t0 = time.monotonic()
    if args.concurrency <= 1:
        per = [do_window(i) for i in range(args.windows)]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            per = list(ex.map(do_window, range(args.windows)))
    wall = time.monotonic() - t0

    out = {
        "mode": "episode", "policy": args.policy, "cache_prompt": args.cache_prompt,
        "windows": args.windows, "questions": args.questions,
        "window_tokens": args.window, "concurrency": args.concurrency,
        "n_predict": args.n_predict, "force_decode": args.force_decode,
        "episode_wall_s": round(wall, 2),
        "per_window_wall_s_median": round(statistics.median(per), 3),
        "s_per_window": round(wall / args.windows, 3),
        "s_per_call": round(wall / (args.windows * args.questions), 3),
        "slot_mismatches": len(mismatches),
        "slot_mismatch_examples": mismatches[:8],
    }
    print(f"[episode policy={args.policy} cache={args.cache_prompt} "
          f"k={args.concurrency}] {args.windows}x{args.questions} -> "
          f"{wall:.1f}s total, {wall/args.windows:.2f} s/window")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--mode", required=True, choices=["pattern", "throughput", "windows", "episode"])
    ap.add_argument("--cells", default="s2-1024-p50,s2-2048-p50,s2-4096-p10,s2-8192-p50,"
                                       "s2-16384-p95,s2-32768-p97,s2-2048-p10,s2-4096-p90")
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--no-cache-prompt", dest="cache_prompt", action="store_false",
                    default=True, help="send cache_prompt:false on every call")
    ap.add_argument("--arms", default="shared_slot,virgin_slot")
    ap.add_argument("--window-ladder", default=None,
                    help="comma list of per-cell window sizes; isolates size from growth")
    ap.add_argument("--kvals", default="1,2,4,8")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--slot-base", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--force-decode", type=int, default=0,
                    help="pin every answer to exactly N decoded tokens (ignore_eos)")
    ap.add_argument("--policy", default="virgin", choices=["virgin", "shared"])
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="milestones/s2/results/r13_mitigation.jsonl")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    srv = Server(args.base_url or f"http://127.0.0.1:{args.port}")
    srv.wait_healthy()
    fn = {"pattern": mode_pattern, "throughput": mode_throughput,
          "windows": mode_windows, "episode": mode_episode}[args.mode]
    result = fn(srv, args)
    srv.close()
    blob = json.dumps(result, indent=1)
    print(blob[:4000])
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(blob, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
