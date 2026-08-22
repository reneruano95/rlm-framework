"""S1 — the minimal-loop slice (spec §9 S1).

A package, not a loose script directory, for one reason: `tests/test_s1_
fixtures.py` imports `s1.make_fixtures`, and the S1 gate runner has to be
importable from the repo root the same way the tests are. Nothing in `rlm/`
may ever import from here — this is measurement apparatus, not scaffold.
"""
