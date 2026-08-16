"""Price the three ways out of the §8 aggregation deadlock.

THE DEADLOCK (ARCHITECTURE.md §8, "Corpus-size ceiling per category"):
  * the aggregation cap is 200K tokens => <=463 windows => <=926 sub-calls;
  * but "the wall clock binds well before the sub-call budget does: at the
    measured ~2.78 s per window serial, max_wall_clock_s: 900 supports ~323
    windows ~= 139K tokens ~= 525K chars before any root turn is charged";
  * so §8 rules "no aggregation task may be authored against the 200K figure
    until [concurrent dispatch is usable]" and max_wall_clock_s "must be
    re-derived per size class ... once concurrent dispatch is usable";
  * R14 says concurrent dispatch corrupts answers (31/32 correct serial, 7/32 at
    2 in flight) and dispatch_concurrency is pinned at 1, possibly permanently.

Every number below is a PROJECTION from measured inputs, not a measurement.
Inputs are named so each can be challenged on its own.
"""
from __future__ import annotations

# ---- measured inputs (source in the comment) --------------------------------
S_PER_WINDOW = 2.78        # §8, serial, includes both sub-calls per window
CHARS_PER_TOKEN = 3.7727   # §8, measured
SNAP_STRIDE = 432          # §8: int(stride 480 * 0.9), the snap bound
WALL_CAP_NOW = 900         # config.yaml max_wall_clock_s
S4_BUDGET_H = 60           # §8 pre-registered wall budget for a full S4
ARMS = 4                   # RLM, B1, B2, B3
SEEDS = 3

# ---- assumptions (stated, challengeable) ------------------------------------
# Only RLM and B2 read the corpus chunk-by-chunk; B1 and B3 are single-shot
# (§8 "two chunked-and-exposed (RLM, B2) versus two single-shot-and-spared").
EXPENSIVE_ARMS = 2
CHEAP_ARM_S = 60           # a single-shot arm: one big call + scoring
NON_AGG_EXPENSIVE_S = 450  # needle/synthesis/codeQA on an expensive arm; these
                           # prescan rather than cover, so they are far cheaper
                           # than an aggregation sweep. Conservative guess.
ROOT_OVERHEAD_FRAC = 0.30  # share of an aggregation episode NOT spent in leaf
                           # windows (root turns, REPL, scoring)


def windows_for(tokens: int) -> int:
    return -(-tokens * 1 // SNAP_STRIDE)      # ceil(T / 432), §8's snap bound


def leaf_seconds(tokens: int) -> float:
    return windows_for(tokens) * S_PER_WINDOW


def episode_seconds(tokens: int) -> float:
    return leaf_seconds(tokens) / (1 - ROOT_OVERHEAD_FRAC)


def tokens_affordable(wall_cap: int) -> int:
    usable = wall_cap * (1 - ROOT_OVERHEAD_FRAC)
    return int(usable / S_PER_WINDOW) * SNAP_STRIDE


def s4_hours(n_tasks: int, n_agg: int, agg_tokens: int, wall_cap: int) -> float:
    agg_ep = min(episode_seconds(agg_tokens), wall_cap)
    agg = n_agg * SEEDS * EXPENSIVE_ARMS * agg_ep
    other = (n_tasks - n_agg) * SEEDS * EXPENSIVE_ARMS * NON_AGG_EXPENSIVE_S
    cheap = n_tasks * SEEDS * (ARMS - EXPENSIVE_ARMS) * CHEAP_ARM_S
    escalation = 32 * agg_ep * 0.5        # §8: "typically 8-32 extra episodes"
    return (agg + other + cheap + escalation) / 3600


ROOT_CTX = 32_768          # config.yaml servers.root.ctx -- all B1 can ever see


def row(name: str, n_tasks: int, n_agg: int, agg_tokens: int, wall_cap: int):
    w = windows_for(agg_tokens)
    ep = episode_seconds(agg_tokens)
    h = s4_hours(n_tasks, n_agg, agg_tokens, wall_cap)
    # The number that decides what the CATEGORY still tests: aggregation exists
    # to make single-shot reading impossible. Shrinking the corpus hands B1 back
    # a larger share of it, and B1 is the baseline RLM must beat by +2.
    b1_sees = ROOT_CTX / agg_tokens
    fits_ep = "OK" if ep <= wall_cap else f"BREACH ({ep:.0f}s > {wall_cap}s)"
    fits_s4 = "OK" if h <= S4_BUDGET_H else "OVER"
    print(f"{name:<34} {n_tasks:>3} {n_agg:>4} {agg_tokens:>8,} {w:>7} "
          f"{2*w:>8} {ep:>9.0f} {wall_cap:>7} {h:>7.1f} {b1_sees:>7.0%}  "
          f"{fits_ep:<22} {fits_s4}")


def main() -> None:
    print(f"snap-bound windows = ceil(tokens / {SNAP_STRIDE}); "
          f"{S_PER_WINDOW}s per window serial; "
          f"{ROOT_OVERHEAD_FRAC:.0%} of an episode is non-leaf\n")
    print(f"{'option':<34} {'N':>3} {'agg':>4} {'agg tok':>8} {'windows':>7} "
          f"{'subcalls':>8} {'ep secs':>9} {'cap':>7} {'S4 h':>7} {'B1 sees':>7}  "
          f"{'per-episode':<22} 60h")
    print("-" * 130)

    print(">>> the status quo, which is why nothing can be authored:")
    row("as-written (200K @ cap 900)", 20, 5, 200_000, 900)
    print()

    print(">>> A: re-derive max_wall_clock_s for SERIAL dispatch")
    need = episode_seconds(200_000)
    row("A  cap 1900, agg 200K", 20, 5, 200_000, 1900)
    row("A  cap 1900, agg 200K, N=30", 30, 7, 200_000, 1900)
    print(f"    (200K needs {need:.0f}s per aggregation episode, so the cap has "
          f"to more than DOUBLE from {WALL_CAP_NOW}s -- §8's own '~1,560s' "
          f"figure charged no root overhead)")
    print()

    print(">>> RULED 2026-08-15: hybrid, aggregation 130K, N=30, cap 1300")
    row("RULING cap 1300, agg 130K, N=30", 30, 7, 130_464, 1300)
    print("    The cap is 1300 and not 1200: a 130,464-token corpus needs 1,199 s")
    print("    per aggregation episode, so a 1,200 s cap leaves ONE SECOND of")
    print("    margin, which is not a margin. 1,300 gives 8.4%. It costs nothing")
    print("    in the S4 projection because episodes do not run to the cap --")
    print("    the projected total is identical either way.")
    print()

    print(">>> H: hybrid -- raise the cap somewhat AND shrink the corpus")
    for cap in (1200, 1500):
        fit = tokens_affordable(cap)
        row(f"H  cap {cap}, agg {fit//1000}K", 20, 5, fit, cap)
    fit = tokens_affordable(1200)
    row(f"H  cap 1200, agg {fit//1000}K, N=30", 30, 7, fit, 1200)
    print()

    print(">>> B: shrink the aggregation corpus to fit the EXISTING cap")
    fit = tokens_affordable(WALL_CAP_NOW)
    row(f"B1 cap 900, agg {fit//1000}K", 20, 5, fit, 900)
    row(f"B2 cap 900, agg {fit//1000}K, N=30", 30, 7, fit, 900)
    print(f"    (largest corpus that fits 900s with {ROOT_OVERHEAD_FRAC:.0%} "
          f"root overhead = {fit:,} tokens = {fit*CHARS_PER_TOKEN/1000:.0f}K chars)")
    print()

    print(">>> C: fewer aggregation tasks (does NOT resolve the deadlock --")
    print("      each REMAINING aggregation task still must be sized to fit)")
    row("C  200K @ cap 900, only 3 agg", 20, 3, 200_000, 900)
    print()

    print("Sanity checks against §8's own numbers:")
    print(f"  200K tokens -> {windows_for(200_000)} windows "
          f"(§8 says <=463)   {2*windows_for(200_000)} sub-calls (§8 says <=926)")
    print(f"  cap 900 with NO root overhead -> "
          f"{int(900/S_PER_WINDOW)} windows (§8 says ~323)")
    print(f"  that is {int(900/S_PER_WINDOW)*SNAP_STRIDE:,} tokens "
          f"(§8 says ~139K) = "
          f"{int(900/S_PER_WINDOW)*SNAP_STRIDE*CHARS_PER_TOKEN/1000:.0f}K chars "
          f"(§8 says ~525K)")


if __name__ == "__main__":
    main()
