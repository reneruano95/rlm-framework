"""S2 — the chunk-size QUALITY sweep (spec §7 #2, promoted first in v0.2.4).

A package, not a loose script directory, for the same reason `s1` is one:
`tests/test_s2_sweep.py` imports `s2.make_sweep_fixtures` / `s2.run_sweep` /
`s2.run_ub_experiment`, and those must be importable from the repo root
exactly the way the tests import them. Nothing in `rlm/` may ever import
from here — this is measurement apparatus, not scaffold.
"""
