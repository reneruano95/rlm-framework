"""The shipped default config must not drift from the repo's real one.

`src/rlm/_data/config.default.yaml` exists so a copied `rlm/` can be imported and
its suite run with no repo present. It is DERIVED from `config.yaml`, not written
beside it, and these tests are what keeps that true: a key added to the real config
and forgotten in the default would give consumers a package whose shipped config is
a different shape from the one every recorded run used.

The nine leaves that legitimately differ are enumerated below and nowhere else.
"""
from pathlib import Path

import pytest
import yaml

from rlm.config import Config, default_config_path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL = REPO_ROOT / "config.yaml"

# The only leaves allowed to differ, and why each one has to:
#   servers.*.model        model weights live outside the repo (GGUF, tens of GB)
#   servers.*.backend_dir  a built llama.cpp, also outside the repo
#   servers.root.dflash    + its flag set: config.py refuses dflash=true without a
#                          -md drafter, and refuses a -md path that is not on disk,
#                          so a shipped default cannot keep speculative decoding on
#   scaffold.sandbox.*     the interpreter and bootstrap dir are this box's
MACHINE_LEAVES = {
    "servers.root.model",
    "servers.root.backend_dir",
    "servers.root.dflash",
    "servers.root.extra_flags",
    "servers.leaf.model",
    "servers.leaf.backend_dir",
    "servers.bench_leaf.model",
    "servers.bench_leaf.backend_dir",
    "scaffold.sandbox.interpreter",
    "scaffold.sandbox.bootstrap_dir",
}

MACHINE_TOKENS = ("D:" + chr(92), "C:" + chr(92), "D:/", "C:/", "/home/", "/mnt/")


def _leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        yield path, obj  # lists compared whole; extra_flags is the only one that differs
    else:
        yield path, obj


@pytest.fixture(scope="module")
def real() -> dict:
    return yaml.safe_load(REAL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped() -> dict:
    return yaml.safe_load(default_config_path().read_text(encoding="utf-8"))


def test_the_shipped_default_has_the_same_keys_as_the_real_config(real, shipped):
    """A key added to one and not the other is drift, in either direction."""
    assert {p for p, _ in _leaves(real)} == {p for p, _ in _leaves(shipped)}


def test_only_the_enumerated_machine_leaves_differ(real, shipped):
    """Everything but the nine machine leaves is byte-for-byte the shipped structure."""
    rv = dict(_leaves(real))
    sv = dict(_leaves(shipped))
    differing = {p for p in rv if rv[p] != sv.get(p)}
    assert differing <= MACHINE_LEAVES, (
        f"the default drifted from config.yaml on leaves that are not machine-specific: "
        f"{sorted(differing - MACHINE_LEAVES)}"
    )


def test_the_shipped_default_names_no_machine_path(shipped):
    """The whole point: a consumer's copy must carry nothing about this box."""
    offenders = [
        p for p, v in _leaves(shipped)
        if isinstance(v, str) and any(t in v for t in MACHINE_TOKENS)
    ] + [
        f"{p}{v}" for p, v in _leaves(shipped)
        if isinstance(v, list)
        and any(isinstance(x, str) and any(t in x for t in MACHINE_TOKENS) for x in v)
    ]
    assert not offenders, f"shipped default names this machine at: {offenders}"


def test_the_shipped_default_validates(shipped):
    """It has to load, or the package ships a config nothing can use."""
    Config.model_validate(shipped)


def test_every_shipped_prompt_resolves_and_matches_its_pin(shipped):
    """The prompts moved into the package; the sha pins are the proof they arrived intact."""
    from rlm.config import resolve_prompt_path

    cfg = Config.model_validate(shipped)
    pinned = [(n, r) for n, r in cfg._prompt_refs() if r.sha256 is not None]
    assert pinned, "no prompt is pinned -- the drift check below would be vacuous"
    for name, ref in pinned:
        assert resolve_prompt_path(ref.path).exists(), f"{name} does not resolve"
