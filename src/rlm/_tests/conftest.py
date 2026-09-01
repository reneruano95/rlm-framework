"""Configuration for the shipped contract suite.

Deliberately empty of hooks, fixtures and ini assumptions. Two reasons, both
because this file runs inside somebody else's project:

1. **It changes nothing globally.** In particular it does NOT set
   `filterwarnings = ["error"]`. The repo's own `pyproject.toml` does, which is
   right for this repo and wrong for a consumer: a library whose bundled tests
   turn every warning in the host's session into an error is a library that
   breaks its host's suite. Warnings this suite cares about are asserted locally
   with `pytest.warns`.

2. **It assumes no ini options.** The repo sets `asyncio_mode = "auto"`; a
   consumer running `pytest --pyargs rlm` has no such setting. The file that
   needs it carries a module-level `pytestmark = pytest.mark.asyncio`, applied at
   import time. A `pytest_collection_modifyitems` hook was tried first and does
   not work: pytest-asyncio decides during collection, so the marker arrives
   after the decision and the tests fail with "async def functions are not
   natively supported". Measured 2026-09-01 from a bare copy of the package.
"""
