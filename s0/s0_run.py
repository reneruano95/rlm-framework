#!/usr/bin/env python3
"""S0 measurement runner (rlm-runtime-spec v0.2.2 §9).

Stdlib only. Each subcommand measures one S0 item against a running
llama-server and appends a JSON record to s0/results/raw.jsonl.
Servers are launched/killed externally; this script only measures.
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

RESULTS = Path(__file__).parent / "results" / "raw.jsonl"

WORDS = (
    "system ledger harbor granite velvet copper meadow signal lantern orchard "
    "timber falcon marble cinder willow beacon quarry ribbon saddle thistle "
    "anchor bramble crystal dynamo ember foxglove gutter hollow ingot juniper "
    "kestrel lattice mortar nectar oakum pallet quiver rudder sextant tallow "
    "umber vellum wicket yarrow zephyr basalt cobalt drizzle estuary fathom"
).split()


def http_json(url, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(base, tries=180):
    for _ in range(tries):
        try:
            http_json(base + "/health", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


def gen_text(n_words, seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n_words):
        w = rng.choice(WORDS)
        if i % 13 == 12:
            w += "."
        out.append(w)
    return " ".join(out)


def tokenize_n(base, text):
    return len(http_json(base + "/tokenize", {"content": text})["tokens"])


def record(kind, payload):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **payload}
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))


def completion(base, prompt, n_predict, cache_prompt, slot=-1, timeout=900):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": cache_prompt,
        "temperature": 0.0,
        "seed": 1,
        "ignore_eos": True,
    }
    if slot >= 0:
        body["id_slot"] = slot
    t0 = time.perf_counter()
    r = http_json(base + "/completion", body, timeout=timeout)
    wall = time.perf_counter() - t0
    t = r.get("timings", {})
    return {
        "wall_s": round(wall, 3),
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": round(t.get("prompt_n", 0) / t["prompt_ms"] * 1000, 2)
        if t.get("prompt_ms")
        else None,
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "predicted_tps": round(
            t.get("predicted_n", 0) / t["predicted_ms"] * 1000, 2
        )
        if t.get("predicted_ms")
        else None,
        "cache_n": t.get("cache_n"),
        "content_head": (r.get("content") or "")[:60],
    }


def cmd_fixture(a):
    # generate text, trim to target token count via tokenize/detokenize
    words = int(a.tokens * 0.9)
    text = gen_text(words, seed=a.seed)
    toks = http_json(a.base + "/tokenize", {"content": text})["tokens"]
    while len(toks) < a.tokens:
        words = int(words * 1.15)
        text = gen_text(words, seed=a.seed)
        toks = http_json(a.base + "/tokenize", {"content": text})["tokens"]
    toks = toks[: a.tokens]
    text = http_json(a.base + "/detokenize", {"tokens": toks})["content"]
    n = tokenize_n(a.base, text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    record("fixture", {"out": a.out, "target": a.tokens, "actual_tokens": n})


def cmd_props(a):
    p = http_json(a.base + "/props", timeout=30)
    keep = {
        k: p.get(k)
        for k in (
            "default_generation_settings",
            "total_slots",
            "model_path",
            "build_info",
        )
    }
    record("props", {"label": a.label, "props": keep})


def cmd_prefill(a):
    text = Path(a.file).read_text(encoding="utf-8")
    runs = []
    for i in range(a.runs):
        tag = f"COLD{i}R{random.randint(10000, 99999)} "
        res = completion(a.base, tag + text, 1, cache_prompt=False)
        runs.append(res)
        print(
            f"run {i}: prompt_n={res['prompt_n']} "
            f"prefill={res['prompt_tps']} t/s",
            file=sys.stderr,
        )
    med = statistics.median(r["prompt_tps"] for r in runs)
    record(
        "prefill",
        {"label": a.label, "runs": runs, "median_prompt_tps": med},
    )


def cmd_decode(a):
    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
        tag = f"DEC{random.randint(10000, 99999)} "
        res = completion(a.base, tag + text, a.n, cache_prompt=False)
    else:
        res = completion(a.base, "The quick brown fox", a.n, cache_prompt=False)
    record("decode", {"label": a.label, "depth_file": a.file, **res})


def cmd_scale(a):
    text = Path(a.file).read_text(encoding="utf-8")
    levels = [int(x) for x in a.levels.split(",")]
    out = []
    for k in levels:
        results = [None] * k
        barrier = threading.Barrier(k + 1)

        def worker(idx):
            tag = f"SC{k}W{idx}R{random.randint(10000, 99999)} "
            barrier.wait()
            results[idx] = completion(
                a.base, tag + text, 1, cache_prompt=False
            )

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(k)
        ]
        for t in threads:
            t.start()
        barrier.wait()
        t0 = time.perf_counter()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total = sum(r["prompt_n"] for r in results)
        agg = round(total / wall, 2)
        solo = [r["prompt_tps"] for r in results]
        out.append(
            {
                "k": k,
                "wall_s": round(wall, 2),
                "total_prompt_tokens": total,
                "aggregate_tps": agg,
                "per_req_tps_min": min(solo),
                "per_req_tps_max": max(solo),
            }
        )
        print(f"k={k}: aggregate {agg} t/s (wall {wall:.1f}s)", file=sys.stderr)
    record("scale", {"label": a.label, "levels": out})


def cmd_warm(a):
    text = Path(a.file).read_text(encoding="utf-8")
    r1 = completion(a.base, text, 1, cache_prompt=True, slot=0)
    r2 = completion(a.base, text, 1, cache_prompt=True, slot=0)
    reuse = (
        round(r2["cache_n"] / r2["prompt_n"], 4)
        if r2["cache_n"] is not None and r2["prompt_n"]
        else None
    )
    record(
        "warm",
        {
            "label": a.label,
            "run1_cache_n": r1["cache_n"],
            "run1_prompt_ms": r1["prompt_ms"],
            "run2_cache_n": r2["cache_n"],
            "run2_prompt_n": r2["prompt_n"],
            "run2_prompt_ms": r2["prompt_ms"],
            "token_weighted_reuse": reuse,
        },
    )


def cmd_multiturn(a):
    # root-turn integrity (§7 #3c): the transcript grows turn by turn and
    # each new turn extends the slot's cached sequence, so per-turn cache_n
    # must approximate the prior transcript's token count.
    turns = []
    transcript = ""
    for i in range(a.turns):
        prior_tokens = tokenize_n(a.base, transcript) if transcript else 0
        chunk = gen_text(a.words, seed=100 + i)
        prompt = transcript + "\n\nUser: " + chunk + "\nAssistant:"
        body = {
            "prompt": prompt,
            "n_predict": a.n,
            "cache_prompt": True,
            "id_slot": 0,
            "temperature": 0.0,
            "seed": 1,
            "ignore_eos": True,
        }
        r = http_json(a.base + "/completion", body)
        t = r.get("timings", {})
        turns.append(
            {
                "turn": i + 1,
                "prompt_n": t.get("prompt_n"),
                "cache_n": t.get("cache_n"),
                "expected_min_cache": prior_tokens,
                "decode_tps": round(
                    t.get("predicted_n", 0) / t["predicted_ms"] * 1000, 2
                )
                if t.get("predicted_ms")
                else None,
            }
        )
        transcript = prompt + (r.get("content") or "")
    record("multiturn", {"label": a.label, "turns": turns})


def cmd_decodescale(a):
    # aggregate decode across k continuous-batching streams (short prompts)
    k = a.k
    results = [None] * k
    barrier = threading.Barrier(k + 1)

    def worker(idx):
        prompt = f"D{idx} " + gen_text(40, seed=500 + idx)
        barrier.wait()
        results[idx] = completion(a.base, prompt, a.n, cache_prompt=False)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    total = sum(r["predicted_n"] for r in results)
    record(
        "decodescale",
        {
            "label": a.label,
            "k": k,
            "wall_s": round(wall, 2),
            "total_predicted": total,
            "aggregate_decode_tps": round(total / wall, 2),
            "per_stream_tps": [r["predicted_tps"] for r in results],
        },
    )


def cmd_soak(a):
    text = Path(a.file).read_text(encoding="utf-8")
    t_end = time.time() + a.seconds
    waves = []
    w = 0
    while time.time() < t_end:
        k = a.k
        results = [None] * k
        barrier = threading.Barrier(k + 1)

        def worker(idx):
            tag = f"SO{w}W{idx}R{random.randint(10000, 99999)} "
            barrier.wait()
            results[idx] = completion(a.base, tag + text, 1, cache_prompt=False)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(k)
        ]
        for t in threads:
            t.start()
        barrier.wait()
        t0 = time.perf_counter()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total = sum(r["prompt_n"] for r in results)
        waves.append(
            {
                "wave": w,
                "t_offset_s": round(time.time() - (t_end - a.seconds), 1),
                "aggregate_tps": round(total / wall, 2),
                "wall_s": round(wall, 2),
            }
        )
        print(f"wave {w}: {waves[-1]['aggregate_tps']} t/s", file=sys.stderr)
        w += 1
    first, last = waves[0]["aggregate_tps"], waves[-1]["aggregate_tps"]
    record(
        "soak",
        {
            "label": a.label,
            "waves": waves,
            "first_wave_tps": first,
            "last_wave_tps": last,
            "drift_pct": round((last - first) / first * 100, 2),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8081")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fixture")
    s.add_argument("--tokens", type=int, default=32768)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", required=True)

    s = sub.add_parser("props")
    s.add_argument("--label", required=True)

    s = sub.add_parser("prefill")
    s.add_argument("--file", required=True)
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--label", required=True)

    s = sub.add_parser("decode")
    s.add_argument("--file", default=None)
    s.add_argument("--n", type=int, default=128)
    s.add_argument("--label", required=True)

    s = sub.add_parser("scale")
    s.add_argument("--file", required=True)
    s.add_argument("--levels", default="1,2,4,8")
    s.add_argument("--label", required=True)

    s = sub.add_parser("warm")
    s.add_argument("--file", required=True)
    s.add_argument("--label", required=True)

    s = sub.add_parser("multiturn")
    s.add_argument("--turns", type=int, default=3)
    s.add_argument("--words", type=int, default=1200)
    s.add_argument("--n", type=int, default=64)
    s.add_argument("--label", required=True)

    s = sub.add_parser("decodescale")
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--n", type=int, default=128)
    s.add_argument("--label", required=True)

    s = sub.add_parser("soak")
    s.add_argument("--file", required=True)
    s.add_argument("--seconds", type=int, default=600)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--label", required=True)

    a = p.parse_args()
    if not wait_health(a.base):
        print("server not healthy", file=sys.stderr)
        sys.exit(1)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
