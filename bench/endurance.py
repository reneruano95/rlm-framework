#!/usr/bin/env python3
"""endurance.py -- a soak driver for any OpenAI-compatible llama-server.

Moved here from the prime-agent spike's tools/ on 2026-09-01 when the rest of that
directory was deleted. It survives because it is stdlib + urllib with zero coupling
to prime-agent, and because the decision rule cites its result -- +9.9% thermal drift
across a 2,000-request run -- as THE REASON wall-clock is recorded but never gated.
That rule now lives in section 13 of the s6-lite spec, `decide.py` having been deleted
2026-09-01 for having no input in the repo; the measurement is still its premise.

Originally: the C1b driver for the prime-agent local spike.

Reading C1b (docs/superpowers/plans/2026-08-26-prime-agent-local-spike.md,
section 2, Phase C):

    "the Gate 0 section 5 design: a scripted loop of 2,000 short, independent
     chat completions (distinct ~200-token user messages, no shared prefix
     beyond the system line) against the same server process, after Phase C;
     alive at the end, RSS <2x, zero non-200 responses"

Runs on the Windows host against the hand-launched llama-server (section 3
step 1).  Standard library only (urllib.request, json, csv, time, statistics).

    python endurance.py --url http://127.0.0.1:8080 --n 2000 --out c1b.csv
    python endurance.py --url http://127.0.0.1:8080 --n 2000 --out c1b.csv \
        --model qwen3.8-27b --sample-metrics 50

Requests are strictly sequential (one in flight at a time), non-streaming,
temperature 0, max_tokens 64, no tools.  Every user message is generated
deterministically from its index, so index i always produces the same text on
every run and no two indices share a prefix: the message opens with the index
itself, followed by a per-index shuffle of a fixed word list.

One CSV row is written and flushed per request, so a server crash or a Ctrl-C
leaves a complete record of everything up to that point.  Non-200 responses
and transport exceptions are recorded and the loop CONTINUES; only 20
consecutive failures stop the run.

RSS is not sampled here -- the plan's host-side sampler (section 3 step 1,
D:\\spike\\rss.csv) owns that reading.

Exit codes: 0 = all requests succeeded, 1 = finished with failures,
2 = aborted on 20 consecutive failures, 130 = interrupted.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

# --- request shape (fixed by the plan) ---------------------------------------

SYSTEM_LINE = "You are a benchmark responder. Answer in at most five words."
MAX_TOKENS = 64
TEMPERATURE = 0.0
CONSECUTIVE_FAILURE_LIMIT = 20
DEFAULT_WORDS_PER_MESSAGE = 128  # ~200 prompt tokens including the system line
TRAILER = "Reply with the single word ACK."

# Fixed vocabulary for the filler text.  Ordinary English words keep the
# tokenizer honest (~1.3 tokens/word) and contain no control-token strings.
# The list never changes, so the per-index shuffle is reproducible.
WORDS = """
account acre afternoon agency anchor angle apple arbor archive arrow autumn
avenue balance barrel basin beacon bearing bedrock bellows binder blanket
border bracket branch bridge bronze bucket bundle burrow cabinet cadence
canvas capital carbon cargo carpet cascade catalog cavern cedar cellar census
chamber channel charter chimney cipher circuit cistern clause clearing cliff
cluster cobalt column compass conduit copper corridor cotton council counter
courtyard crate crescent crossing crystal current curtain cypress dagger dairy
delta depot desert diagram district ditch dock domain drawer driftwood drought
eastern echo edifice ember embankment engine envelope estate estuary factory
fallow fathom feather fence ferry fiber figure filament flagon flint foliage
forge fountain fragment freight frontier funnel furnace gallery garden garland
gateway gauge girder glacier granary granite gravel grotto gutter hamlet
harbor harvest hatchway header hearth hedge helix hillside hinge hollow
horizon hostel hourglass hull ingot inlet iron island ivory jetty journal
junction juniper kernel kettle keystone lantern lattice launch ledger lever
lichen lighthouse limestone lintel lodge lumber machine magnet mantle manual
marble margin market marsh masonry meadow measure meridian metal milestone
mineral mirror module moisture monument mortar mosaic motor mountain nautical
network nickel notch nursery oasis obelisk observatory ochre offset orchard
outpost overhang packet paddock palette panel parapet parcel passage pasture
pathway pavilion pedestal pendulum petal pewter pillar pinion pipeline plateau
platform plaza plinth plumage pocket pollen portal portico pottery prairie
precinct printer prism promenade province pulley pumice quarry quartz quay
quilt radial rafter railing rampart ravine reactor reservoir residue ridge
rigging river roster rotunda rudder runway saddle salvage sandstone sawmill
scaffold schedule seabed seam sediment sentry shale shelter shipment shoreline
shutter signal silo sketch slate sleeve sluice smelter socket solder spindle
spire spillway spool sprocket stable stairwell stanchion station steeple
stipple stockyard stonework storeroom strata stratum stream strut summit
surveyor switch syrup tackle tallow tannery tapestry tavern telegraph tenant
terrace textile thicket threshold timber tinder tollgate topsoil torrent
tower tramway transit trellis tribute trolley trough tunnel turbine turret
twilight umber upland valley vantage vault veneer verge vessel viaduct village
vineyard voltage warehouse waterway weather weir wharf wheelhouse willow
windlass workshop yardarm zenith
""".split()


def build_user_message(index: int, n_words: int = DEFAULT_WORDS_PER_MESSAGE) -> str:
    """Deterministic, index-unique filler of roughly 200 tokens.

    Seeded from the index alone, via a str seed (so it does not depend on
    PYTHONHASHSEED, platform or word order), and opening with the index, so:
      * the same index always yields the same text, and
      * no two messages share anything beyond the fixed system line.
    """
    rnd = random.Random("rlm-halo/c1b-endurance/%d" % index)
    pool: list[str] = []
    while len(pool) < n_words:
        chunk = list(WORDS)
        rnd.shuffle(chunk)
        pool.extend(chunk)
    pool = pool[:n_words]

    sentences: list[str] = []
    pos = 0
    while pos < len(pool):
        words = pool[pos:pos + rnd.randint(8, 13)]
        pos += len(words)
        if not words:
            break
        sentences.append(" ".join([words[0].capitalize()] + words[1:]) + ".")

    header = "Ledger entry %06d (survey batch %d, sequence %d)." % (
        index, index // 100, index % 100,
    )
    return " ".join([header] + sentences + [TRAILER])


# --- HTTP --------------------------------------------------------------------

def post_chat(endpoint: str, payload: dict, timeout: float):
    """POST /v1/chat/completions.  Returns (status, body_text, error_str).

    status == 0 means no HTTP response at all (transport exception/timeout).
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), ""
    except urllib.error.HTTPError as exc:  # the server answered, non-2xx
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return exc.code, body, "http_error: %s" % (exc.reason,)
    except Exception as exc:  # URLError, timeout, connection reset, ...
        return 0, "", "%s: %s" % (type(exc).__name__, exc)


def get_text(base_url: str, path: str, timeout: float = 10.0):
    """GET a text endpoint.  Returns (text, error_str)."""
    req = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace"), ""
    except Exception as exc:
        return "", "%s: %s" % (type(exc).__name__, exc)


def fetch_metrics(base_url: str, timeout: float = 10.0):
    """Scrape /metrics.  Returns ({series_name: float}, error_str).

    Series names keep their labels verbatim, e.g.
    llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"}.
    """
    text, err = get_text(base_url, "/metrics", timeout)
    if err:
        return {}, err
    series: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("llamacpp:"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name, raw = parts
        try:
            series[name] = float(raw)
        except ValueError:
            continue
    return series, ""


# --- small helpers -----------------------------------------------------------

def percentile(values, q: float) -> float:
    """Nearest-rank percentile; q in [0, 1].  Safe for n == 0 and n == 1."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


CSV_FIELDS = [
    "index",
    "ts_iso",
    "http_status",
    "wall_ms",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "prompt_ms",
    "predicted_ms",
    "cache_n",
    "finish_reason",
    "error",
]

METRICS_FIELDS = ["at_request", "ts_iso", "elapsed_s", "metric", "value"]


# --- main --------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="C1b endurance driver: N sequential, prefix-disjoint chat completions.",
    )
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="llama-server base URL (default: http://127.0.0.1:8080)")
    ap.add_argument("--n", type=int, default=2000,
                    help="number of requests (default: 2000)")
    ap.add_argument("--out", required=True,
                    help="output CSV path (one row per request)")
    ap.add_argument("--model", default="qwen3.8-27b",
                    help="model id sent in the request (default: qwen3.8-27b)")
    ap.add_argument("--sample-metrics", type=int, default=0, metavar="N",
                    help="snapshot /metrics every N requests into <out>.metrics.csv")
    ap.add_argument("--words", type=int, default=DEFAULT_WORDS_PER_MESSAGE,
                    help="filler words per user message (default: %d, ~200 tokens)"
                         % DEFAULT_WORDS_PER_MESSAGE)
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-request timeout in seconds (default: 600)")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first message index (default: 0); shift it to resume a run "
                         "without repeating any earlier message text")
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs={'enable_thinking': false}, matching "
                         "the scaffold's decode path (off by default: the plan pins only "
                         "max_tokens/temperature for C1b)")
    args = ap.parse_args(argv)

    if args.n <= 0:
        ap.error("--n must be positive")
    if args.words < 8:
        ap.error("--words must be at least 8")

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    metrics_path = args.out + ".metrics.csv"

    csv_file = open(args.out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    metrics_file = None
    metrics_writer = None
    if args.sample_metrics > 0:
        metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDS)
        metrics_writer.writeheader()
        metrics_file.flush()

    t_start = time.time()

    def snapshot_metrics(at_request: int) -> None:
        if metrics_writer is None:
            return
        series, err = fetch_metrics(args.url)
        ts = now_iso()
        elapsed = round(time.time() - t_start, 3)
        rows = ([{"metric": "_error", "value": err}] if err
                else [{"metric": k, "value": series[k]} for k in sorted(series)])
        for row in rows:
            metrics_writer.writerow({"at_request": at_request, "ts_iso": ts,
                                     "elapsed_s": elapsed, **row})
        metrics_file.flush()

    print("endurance: %d sequential requests -> %s" % (args.n, endpoint))
    print("           model=%s  max_tokens=%d  temperature=%s  words/msg=%d  thinking=%s"
          % (args.model, MAX_TOKENS, TEMPERATURE, args.words,
             "off (chat_template_kwargs)" if args.no_think else "server default"))
    print("           rows -> %s%s" % (
        args.out,
        ("   metrics -> %s (every %d)" % (metrics_path, args.sample_metrics))
        if metrics_writer is not None else ""))
    sys.stdout.flush()

    snapshot_metrics(0)

    ok_walls: list[float] = []
    attempted = 0
    ok_2xx = 0
    non_2xx = 0
    exceptions = 0
    malformed = 0
    consecutive_failures = 0
    first_failure_index = None
    interrupted = False
    aborted = False
    stop_reason = "completed"

    try:
        for offset in range(args.n):
            index = args.start_index + offset
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_LINE},
                    {"role": "user", "content": build_user_message(index, args.words)},
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stream": False,
            }
            if args.no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            attempted += 1
            ts = now_iso()
            t0 = time.perf_counter()
            status, body, error = post_chat(endpoint, payload, args.timeout)
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            row = {field: "" for field in CSV_FIELDS}
            row["index"] = index
            row["ts_iso"] = ts
            row["http_status"] = status
            row["wall_ms"] = wall_ms
            row["error"] = error

            is_2xx = 200 <= status < 300
            if is_2xx:
                try:
                    doc = json.loads(body)
                except Exception as exc:
                    doc = None
                    row["error"] = "bad_json: %s" % (exc,)
                if isinstance(doc, dict):
                    usage = doc.get("usage") or {}
                    row["prompt_tokens"] = usage.get("prompt_tokens", "")
                    row["completion_tokens"] = usage.get("completion_tokens", "")
                    details = usage.get("prompt_tokens_details") or {}
                    row["cached_tokens"] = details.get("cached_tokens", "")
                    timings = doc.get("timings") or {}
                    row["prompt_ms"] = timings.get("prompt_ms", "")
                    row["predicted_ms"] = timings.get("predicted_ms", "")
                    row["cache_n"] = timings.get("cache_n", "")
                    choices = doc.get("choices") or []
                    if choices:
                        row["finish_reason"] = choices[0].get("finish_reason", "")
                elif doc is not None:
                    row["error"] = "bad_json: response was not a JSON object"
            else:
                snippet = " ".join(body.split())[:300]
                if snippet:
                    row["error"] = (row["error"] + " | " if row["error"] else "") + snippet

            writer.writerow(row)
            csv_file.flush()

            failed = (not is_2xx) or bool(row["error"])
            if failed:
                if first_failure_index is None:
                    first_failure_index = index
                consecutive_failures += 1
                if not is_2xx:
                    if status == 0:
                        exceptions += 1
                    else:
                        non_2xx += 1
                else:
                    ok_2xx += 1      # the server did answer 2xx ...
                    malformed += 1   # ... but the body was unusable
                print("  ! request %d: status=%s wall_ms=%.0f  %s"
                      % (index, status, wall_ms, str(row["error"])[:160]))
                sys.stdout.flush()
                if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                    aborted = True
                    stop_reason = ("ABORTED after %d consecutive failures at index %d"
                                   % (consecutive_failures, index))
                    break
            else:
                ok_2xx += 1
                consecutive_failures = 0
                ok_walls.append(wall_ms)

            done = offset + 1
            if args.sample_metrics > 0 and done % args.sample_metrics == 0:
                snapshot_metrics(index + 1)
            if done % 50 == 0 or done == args.n:
                print("  ... %d/%d done  ok=%d  fail=%d  elapsed=%.1fs"
                      % (done, args.n, ok_2xx - malformed,
                         non_2xx + exceptions + malformed, time.time() - t_start))
                sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "INTERRUPTED by user (KeyboardInterrupt)"
    finally:
        snapshot_metrics(attempted)
        csv_file.close()
        if metrics_file is not None:
            metrics_file.close()

    total_wall = time.time() - t_start
    failures = non_2xx + exceptions + malformed
    health, health_err = get_text(args.url, "/health")

    print("")
    print("=" * 70)
    print("C1b endurance summary")
    print("=" * 70)
    print("  endpoint              : %s" % endpoint)
    print("  model                 : %s" % args.model)
    print("  requests attempted    : %d (of %d requested)" % (attempted, args.n))
    print("  2xx responses         : %d" % ok_2xx)
    print("  non-2xx responses     : %d" % non_2xx)
    print("  exceptions (no reply) : %d" % exceptions)
    print("  malformed 2xx bodies  : %d" % malformed)
    print("  total failures        : %d" % failures)
    print("  total wall            : %.1f s (%.2f min)" % (total_wall, total_wall / 60.0))
    if ok_walls:
        print("  wall_ms median (ok)   : %.1f" % statistics.median(ok_walls))
        print("  wall_ms p95    (ok)   : %.1f" % percentile(ok_walls, 0.95))
        print("  wall_ms min/max (ok)  : %.1f / %.1f" % (min(ok_walls), max(ok_walls)))
    else:
        print("  wall_ms median (ok)   : n/a (no successful request)")
        print("  wall_ms p95    (ok)   : n/a")
    print("  first failing index   : %s"
          % ("none" if first_failure_index is None else first_failure_index))
    print("  outcome               : %s" % stop_reason)
    print("  /health after run     : %s"
          % (" ".join(health.split()) if health else "UNREACHABLE (%s)" % health_err))
    print("  row CSV               : %s" % args.out)
    if metrics_writer is not None:
        print("  metrics CSV           : %s" % metrics_path)
    print("  C1b verdict           : %s"
          % ("PASS (server alive, zero failures in %d requests)" % attempted
             if (failures == 0 and health and not aborted and not interrupted
                 and attempted == args.n)
             else "FAIL / incomplete -- see the counts above"))
    print("=" * 70)
    sys.stdout.flush()

    if interrupted:
        return 130
    if aborted:
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
