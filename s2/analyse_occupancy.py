"""Turn `s2/results/occupancy.jsonl` into the tables `s2/OCCUPANCY.md` reports.

Kept separate from the runner so the report can be regenerated without a GPU,
and so the arithmetic that prices the geometry decision is in one readable
place rather than inside a measurement loop.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "s2" / "results" / "occupancy.jsonl"

BUCKETS = ("send_ms", "queue_ms", "prefill_ms", "decode_ms", "tail_ms")


def evictions_per_call(log: Path) -> dict[int, int]:
    """How many `making room for prompt cache entry` lines the server emitted
    between one `launch_slot_` and the next, keyed by the slot launched.

    This is the mechanism read straight off the server's own log: it is the
    only per-request work in the system that scales with the number of
    OCCUPIED slots, and it happens after the slot is launched and before
    prompt processing starts, which is why no `timings` field can see it.
    """
    if not log.exists():
        return {}
    out: dict[int, int] = {}
    slot: int | None = None
    n = 0
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "launch_slot_" in line:
            if slot is not None:
                out[slot] = n
            try:
                slot = int(line.split("id")[1].split("|")[0].strip())
            except (IndexError, ValueError):
                slot = None
            n = 0
        elif "making room for prompt cache entry" in line:
            n += 1
    if slot is not None:
        out[slot] = n
    return out


def load(path: Path = RUNS) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def by_condition(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        out[r["condition"]].append(r)
    return dict(out)


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def band(recs: list[dict[str, Any]], lo: int, hi: int,
         key: str = "occupancy_before") -> list[dict[str, Any]]:
    return [r for r in recs if r["status"] == "ok" and lo <= r[key] <= hi]


def decomposition_table(recs: list[dict[str, Any]], *, width: int = 16) -> str:
    """§1 — the per-bucket decomposition of one condition, banded on occupancy."""
    ok = [r for r in recs if r["status"] == "ok"]
    bands: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        bands[int(r["occupancy_before"]) // width].append(r)
    lines = ["| occupancy | n | wall s | send ms | queue ms | prefill ms | "
             "decode ms | tail ms | residual ms | /health ms | /slots ms | "
             "private MiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in sorted(bands):
        rs = bands[b]
        def m(k: str) -> float:
            return med([r[k] for r in rs if r.get(k) is not None])
        lines.append(
            f"| {b*width}-{b*width+width-1} | {len(rs)} | "
            f"**{m('wall_ms')/1000:.2f}** | **{m('send_ms'):.0f}** | "
            f"{m('queue_ms'):.0f} | {m('prefill_ms'):.0f} | {m('decode_ms'):.0f} | "
            f"{m('tail_ms'):.0f} | {m('residual_ms'):.0f} | "
            f"{m('health_rtt_ms'):.1f} | {m('slots_rtt_ms'):.1f} | "
            f"{m('private_mib'):.0f} |")
    return "\n".join(lines)


def condition_table(conds: dict[str, list[dict[str, Any]]],
                    names: list[str], logs: Path) -> str:
    """§2 — one row per condition: the low band against the high band."""
    lines = ["| condition | flag varied | np | order | n | wall s, first 16 | "
             "wall s, last 16 | ratio | send ms first | send ms last | "
             "prefill ms first | prefill ms last | evictions/call, last |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        recs = conds.get(name)
        if not recs:
            continue
        n_calls = len([r for r in recs if r["status"] == "ok"])
        low = band(recs, 0, 15)
        high = band(recs, max(0, n_calls - 16), n_calls - 1)
        if not low or not high:
            continue
        wl = med([r["wall_ms"] for r in low]) / 1000
        wh = med([r["wall_ms"] for r in high]) / 1000
        ev = evictions_per_call(logs / f"occ-{name}.log")
        ev_high = [ev.get(r["id_slot"], 0) for r in high if r["id_slot"] in ev]
        lines.append(
            f"| `{name}` | `{recs[0]['extra'] or '(none)'}` | {recs[0]['np']} | "
            f"{recs[0]['order']} | {len(recs)} | "
            f"{wl:.2f} | {wh:.2f} | **{wh/wl:.2f}x** | "
            f"{med([r['send_ms'] for r in low]):.0f} | "
            f"{med([r['send_ms'] for r in high]):.0f} | "
            f"{med([r['prefill_ms'] for r in low]):.0f} | "
            f"{med([r['prefill_ms'] for r in high]):.0f} | "
            f"{int(med(ev_high)) if ev_high else '-'} |")
    return "\n".join(lines)


def slot_vs_ordinal(recs: list[dict[str, Any]]) -> str:
    """§3 — for a shuffled run, does `send_ms` track the slot INDEX or the
    number of calls already served? Pearson r against both."""
    ok = [r for r in recs if r["status"] == "ok"]
    if len(ok) < 8:
        return "(not enough calls)"
    def r_of(key: str) -> float:
        xs = [float(r[key]) for r in ok]
        ys = [float(r["send_ms"]) for r in ok]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        return num / den if den else float("nan")
    return (f"send_ms vs CALL ORDINAL (= slots in use): r = {r_of('ordinal'):.3f}; "
            f"send_ms vs SLOT INDEX: r = {r_of('id_slot'):.3f}")


def eviction_slope(recs: list[dict[str, Any]]) -> str:
    """§4 — ms of overhead per occupied slot, from the linear region."""
    ok = [r for r in recs if r["status"] == "ok" and r["occupancy_before"] >= 48]
    if not ok:
        return "(no linear region in this run)"
    slopes = [r["send_ms"] / r["occupancy_before"] for r in ok]
    return (f"median send_ms / occupancy over occ>=48: "
            f"**{med(slopes):.1f} ms per occupied slot** (n={len(ok)})")


def project(base_wall_s: float, evict_ms: float, cache_entries: int,
            pool: int, windows: int, questions: int = 2) -> dict[str, Any]:
    """The decision arithmetic.

    Under R13 never-reuse the pool drains monotonically and is rotated when it
    is spent, so occupancy is a SAWTOOTH from 0 to `pool`, not a monotonic ramp
    over the whole episode. Both questions about a window share that window's
    slot, so occupancy is counted in WINDOWS and the per-call overhead is paid
    by every question at that window's occupancy.
    """
    def overhead_ms(occ: int) -> float:
        return 0.0 if occ <= cache_entries else occ * evict_ms

    total_ms = 0.0
    peak = 0
    for w in range(windows):
        occ = w % pool
        peak = max(peak, occ)
        total_ms += questions * (base_wall_s * 1000 + overhead_ms(occ))
    calls = windows * questions
    return {
        "windows": windows,
        "calls": calls,
        "peak_occupancy": peak,
        "mean_wall_s": round(total_ms / calls / 1000, 3),
        "wall_at_peak_s": round(base_wall_s + overhead_ms(peak) / 1000, 3),
        "episode_serial_s": round(total_ms / 1000, 1),
        "rotations": windows // pool,
    }


ORDER = ["baseline", "cram0", "nocacheidle", "sps0", "shuffle",
         "np8", "np32", "np128", "conc8-keepalive20", "conc8-keepalive1",
         "w640", "cram0-w640"]


def main() -> None:
    recs = load()
    conds = by_condition(recs)
    logs = REPO_ROOT / "traces" / "logs"
    print(f"conditions: {sorted(conds)}\n")
    print("### CONDITION TABLE")
    print(condition_table(conds, [c for c in ORDER if c in conds]
                          + [c for c in sorted(conds) if c not in ORDER], logs))
    print()
    for name, rs in conds.items():
        ok = [r for r in rs if r["status"] == "ok"]
        print(f"## {name}  (n={len(rs)}, ok={len(ok)}, extra={rs[0]['extra']!r}, "
              f"np={rs[0]['np']}, order={rs[0]['order']}, "
              f"chunk={rs[0]['chunk_tokens']}, conc={rs[0]['concurrency']}, "
              f"keepalive={rs[0]['max_keepalive']})")
        print(f"  leaks={sum(1 for r in ok if r.get('leak_detected'))} "
              f"mismatches={sum(1 for r in ok if r.get('slot_mismatch'))} "
              f"correct={sum(1 for r in ok if r.get('answer_correct'))}/{len(ok)}")
        print(decomposition_table(rs))
        print("  " + slot_vs_ordinal(rs))
        print("  " + eviction_slope(rs))
        print()


if __name__ == "__main__":
    sys.exit(main())
