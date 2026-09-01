"""The package's own contract suite. It ships inside the copy unit.

Everything here must run from a bare copy of `rlm/` with only the declared
dependencies installed: no repo-root file is read, no path is walked upward out
of the package, and nothing imports `bench`, which is deliberately not in the
wheel (see `pyproject.toml`). Tests that need the repo -- the benchmark, the
gate, the citation archive -- live in the repo-level `tests/` and stay there.

Run it the way a consumer would:  pytest --pyargs rlm
"""
