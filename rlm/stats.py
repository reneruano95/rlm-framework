"""§8's inference layer, exactly as pre-registered (ARCHITECTURE.md:339-343).

Dependency-free on purpose: the repo has no scipy, and a p-value that cannot
be recomputed from the record is not evidence (s2/run_distance.py precedent).
Changing any rule here after benchmark runs exist is p-hacking.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign (McNemar) test over discordant tasks, null p=0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(deltas: Sequence[float], *, resamples: int = 10_000,
                        seed: int = 8, alpha: float = 0.05) -> tuple[float, float]:
    """Task-level paired percentile bootstrap CI on the mean per-task delta."""
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples))
    lo_idx = int((alpha / 2) * resamples)
    hi_idx = min(resamples - 1, int((1 - alpha / 2) * resamples))
    return means[lo_idx], means[hi_idx]


def task_passes(seed_results: Sequence[bool]) -> bool:
    """>=2/3 pre-escalation; >=3/5 after seeds {4,5} were added."""
    return sum(seed_results) >= (2 if len(seed_results) <= 3 else 3)


def fractional_score(seed_results: Sequence[bool]) -> float:
    return sum(seed_results) / len(seed_results)


def needs_escalation(margin: int) -> bool:
    """Escalate iff the net margin lands in {+1,+2,+3} (ARCHITECTURE.md:343)."""
    return margin in (1, 2, 3)
