"""Regenerate src/rlm/_data/config.default.yaml from the repo's config.yaml.

Not a test -- a generator, kept beside the test that guards its output
(`test_default_config.py`). Run it after changing config.yaml; the guard is what
tells you that you needed to.

Only the leaves that name this box are replaced. Everything else -- budgets, chunk
sizes, prompt slots, checkers, dispatch policy -- is the shipped structure, so the
default cannot drift from the real config in shape.
"""
import io
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))

# The nine machine leaves, replaced with portable placeholders. A consumer
# overrides them; the shipped suite never launches a server, so they only have to
# validate.
for name in ("root", "leaf", "bench_leaf"):
    s = raw["servers"][name]
    s["model"] = f"models/{name}.gguf"
    s["backend_dir"] = "llama.cpp"
    # DFlash off, and its whole flag set with it. Three interlocking validators in
    # config.py make partial removal impossible: dflash=true demands a -md flag
    # (:557 area), a -md path must exist on disk (:557), and a '--spec-type
    # draft-dflash' flag demands dflash=true. Only the complete set can leave. Off
    # is also the honest default -- speculative decoding needs a drafter GGUF the
    # consumer supplies, so a shipped config must not assume one.
    if s.get("dflash"):
        s["dflash"] = False
    flags = s.get("extra_flags")
    if isinstance(flags, list):
        s["extra_flags"] = [
            f for f in flags
            if not (isinstance(f, str) and f.startswith(("-md ", "--spec-type", "--spec-draft-")))
        ]

sb = raw["scaffold"]["sandbox"]
sb["interpreter"] = "python"
sb["bootstrap_dir"] = "sandbox_bootstrap"

out = REPO / "src" / "rlm" / "_data" / "config.default.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
header = (
    "# Derived from the repo's config.yaml -- do not hand-edit.\n"
    "#\n"
    "# This is the config the PACKAGE ships with, so a copied `rlm/` can be imported\n"
    "# and its suite run with no repo present. Nine leaves differ from the real\n"
    "# config, and only those nine: the three server `model` paths, their\n"
    "# `backend_dir`s, one `-md` draft-model flag, and the sandbox `interpreter` and\n"
    "# `bootstrap_dir`. Everything else -- budgets, chunk sizes, prompt slots,\n"
    "# checkers, dispatch policy -- is byte-for-byte the shipped structure, which is\n"
    "# what `tests/test_default_config.py` asserts so the two cannot drift apart.\n"
    "#\n"
    "# The placeholders are not runnable. Nothing in the shipped suite launches a\n"
    "# server; a consumer that wants to run one overrides these nine values.\n"
)
io.open(out, "w", encoding="utf-8", newline="\n").write(
    header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)
)
print("wrote", out, out.stat().st_size, "bytes")
