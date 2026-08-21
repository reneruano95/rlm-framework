import copy
import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from rlm.config import Config, PromptRegistry, config_snapshot, load_config
from rlm.errors import ConfigError


def write_prompt(tmp_path, name, body):
    p = tmp_path / "prompts"
    p.mkdir(exist_ok=True)
    f = p / name
    f.write_text(f"<!-- changelog\nv1 initial\n-->\n{body}", encoding="utf-8")
    return f


def test_extra_keys_are_forbidden(tmp_path):
    with pytest.raises(ConfigError, match="max_subcals"):
        Config.model_validate({"scaffold": {"budgets": {"max_subcals": 32}}})


def test_a_leaf_slot_too_small_for_one_window_is_refused(minimal_cfg_dict):
    bad = minimal_cfg_dict
    bad["servers"]["leaf"]["ctx"] = 12345          # 1,543 tokens per slot
    with pytest.raises(ConfigError, match="ctx"):
        Config.model_validate(bad)


def test_an_over_provisioned_leaf_slot_is_allowed(minimal_cfg_dict):
    """The rule WAS `leaf.ctx == parallel * (size + overhead)`, which was right
    while chunk size was the parallelism lever. Under R13 the lever is slot
    COUNT (§4/§5 C4: `-np 128` -> 2,560 tok/slot), and §7 #2 fixed the window
    at 1,024 -- so equality would now force either a 32K window (falsified) or
    a 39,936-token "overhead" (fiction). What must hold is that a slot can hold
    a window plus its overhead; anything above that is measured headroom, and
    `-np` is pending measurement (§4)."""
    cfg = Config.model_validate(minimal_cfg_dict)
    leaf = cfg.servers.leaf
    chunk = cfg.scaffold.chunk
    assert leaf.ctx // leaf.parallel >= chunk.size_tokens + chunk.overhead_tokens


def test_the_shipped_chunk_geometry_is_the_measured_one(valid_cfg):
    """§7 #2 (v0.3.0): window 640 / stride 480, superseding 1,024/768.

    ONE horizon governs both distances. Facts beyond ~1,000 tokens from the
    question become unfindable (bracket [967 pass, 1022 fail]) and INSTRUCTIONS
    decay over the same distance -- instruction-to-generation distance is
    (window + question), so a 1,024 window puts the system prefix ~47 tokens
    past the measured fail point: 30/30 false positives at 1,024, 0/45 with
    45/45 literal recall at 640. The overhead is re-derived 1,536 -> 1,920 so
    window + overhead stays 2,560, R13's measured dense-slot budget, which
    keeps 128 * 2,560 == leaf.ctx exactly."""
    chunk = valid_cfg.scaffold.chunk
    assert (chunk.size_tokens, chunk.stride_tokens) == (640, 480)
    assert chunk.size_tokens + chunk.overhead_tokens == 2560
    assert 128 * (chunk.size_tokens + chunk.overhead_tokens) == valid_cfg.servers.leaf.ctx


def test_the_leaf_launch_line_disables_the_host_prompt_cache(valid_cfg):
    """`--cache-ram 0` is load-bearing twice over, and neither reason is taste.

    LATENCY (`s2/OCCUPANCY.md`): the default 8,192 MiB host prompt cache holds
    41 entries of this config's 202.80 MiB slot state against a 128-slot pool,
    so past 41 occupied slots the evictions per request equal the occupancy
    exactly, at 37.5 ms each -- 272 s of a 543 s run, per-call wall 2.18 ->
    6.69 s with `timings.prompt_ms` flat. MEASUREMENT (`s2/CACHE-INSTRUMENT.md`):
    the same subsystem restores an idle slot's state onto a DIFFERENT slot, so
    under the default `cache_n` is not a function of the prompts and not
    reproducible run to run -- it is the precondition for §7 #3's gates being
    exact (239/239, 0 tokens of error) rather than history-dependent."""
    assert "--cache-ram 0" in valid_cfg.servers.leaf.extra_flags


def test_a_stride_longer_than_the_window_is_refused(minimal_cfg_dict):
    """A stride above the window leaves tokens in NO window -- silent partial
    coverage, which is how §7 #2's `max_subcalls` shortfall failed before."""
    minimal_cfg_dict["scaffold"]["chunk"]["stride_tokens"] = 2048
    with pytest.raises(ConfigError, match="stride"):
        Config.model_validate(minimal_cfg_dict)


def test_max_subcalls_covers_a_full_pass_at_the_shipped_geometry(valid_cfg):
    """§7 #2/§8.3: the budget is spent per QUESTION, two per window, and a
    budget below full coverage breaks quietly rather than loudly.

    The bound is NOT `ceil((200,000 - size) / stride) + 1`. That assumes every
    window end lands exactly one stride after the last; C2's `_snap_back` moves
    ends BACKWARD to a boundary within the tolerance, so the shortest possible
    gap is `int(stride * (1 - snap_tolerance))` and the window count RISES.
    This asserts the budget against the bound that no corpus can beat."""
    chunk = valid_cfg.scaffold.chunk
    naive = -(-(200_000 - chunk.size_tokens) // chunk.stride_tokens) + 1
    min_gap = int(chunk.stride_tokens * (1 - chunk.snap_tolerance))
    bound = -(-200_000 // min_gap)
    assert (naive, min_gap, bound) == (417, 432, 463)
    assert valid_cfg.scaffold.budgets.max_subcalls >= bound * 2


def test_the_measured_slot_pool_is_pinned(valid_cfg):
    """`s2/R13-slotcount.md` §7: `-np 128` with `-c 327680` retained. Measured
    62.8125 MiB of per-slot recurrent state (priced by slot COUNT, not by the
    token budget -- KV stayed at 3,400 MiB for every `-np`), 32.244 GiB leaf
    residency, 51.53 GiB dual-resident of the 64 GiB carve (19.5% margin) and
    -1.37% prefill. It is the largest pool that fits AND the largest the 1024/768
    window geometry can use: 327,680/128 = 2,560 tok/slot against the ~1,900 the
    geometry needs, while `-np 192` gives 1,706."""
    leaf = valid_cfg.servers.leaf
    assert leaf.parallel == 128
    assert leaf.ctx == 327_680
    assert leaf.ctx // leaf.parallel == 2560


def test_dispatch_concurrency_is_not_the_slot_pool_size(valid_cfg):
    """The equality `dispatch_concurrency == leaf.parallel` was true while
    `--parallel` meant "how many calls this server serves at once". Under
    never-reuse it means "how many WINDOWS one process can serve before it is
    rotated" -- a pool size, measured against memory (`s2/R13-slotcount.md`).
    Concurrency is a throughput lever. Coupling them would put 128 leaf calls
    in flight at once."""
    assert valid_cfg.scaffold.dispatch_concurrency != valid_cfg.servers.leaf.parallel


def test_dispatch_concurrency_is_pinned_to_serial_by_R14(valid_cfg):
    """R14 (spec v0.3.1): concurrent leaf dispatch corrupts the decode.
    Measured serial 31/32 correct, 9-10/32 at two calls in flight, 5/32 at
    four -- and fan-out at two buys 13% of wall-clock while costing 68% of the
    answers, so serial wins on correct-answers-per-second by 2.7x-110x.

    This assertion is a tripwire, not a preference. Raising it silently would
    change what every leaf measurement in `s2/` means. Raise it only when the
    `--no-cont-batching` lead is characterised or fixed upstream, and re-measure
    on correct-answers-per-second -- never on wall-clock alone."""
    assert valid_cfg.scaffold.dispatch_concurrency == 1


def test_concurrency_above_the_pool_size_is_refused(minimal_cfg_dict):
    """Decoupled is not unconstrained: every in-flight call holds a slot, so
    more concurrent calls than slots can only end in an exhaustion the pool
    was never sized for."""
    minimal_cfg_dict["scaffold"]["dispatch_concurrency"] = (
        minimal_cfg_dict["servers"]["leaf"]["parallel"] + 1)
    with pytest.raises(ConfigError, match="dispatch_concurrency"):
        Config.model_validate(minimal_cfg_dict)


def test_zero_concurrency_is_refused(minimal_cfg_dict):
    minimal_cfg_dict["scaffold"]["dispatch_concurrency"] = 0
    with pytest.raises(ConfigError, match="dispatch_concurrency"):
        Config.model_validate(minimal_cfg_dict)


def test_mtp_forces_root_single_slot(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["root"]["mtp"] = True
    minimal_cfg_dict["servers"]["root"]["dflash"] = False
    minimal_cfg_dict["servers"]["root"]["parallel"] = 2
    with pytest.raises(ConfigError, match="mtp"):
        Config.model_validate(minimal_cfg_dict)


def test_prompt_sha256_mismatch_refuses_to_load(tmp_path, valid_cfg):
    f = write_prompt(tmp_path, "root.v1.md", "body")
    reg = PromptRegistry(root_path=f, root_sha256="0" * 64, leaf_prefix_path=f,
                         leaf_prefix_sha256="0" * 64, strategy_paths={},
                         strategy_sha256={})
    with pytest.raises(ConfigError, match="sha256"):
        reg.load()


def test_registry_strips_changelog_header_but_hashes_both(tmp_path):
    f = write_prompt(tmp_path, "root.v1.md", "REAL BODY")
    reg = PromptRegistry.from_files(root_path=f, leaf_prefix_path=f,
                                    strategy_paths={"default": f})
    reg.load()
    assert "changelog" not in reg.render_root("default")
    assert "REAL BODY" in reg.render_root("default")
    h = reg.hashes()
    assert h["root.file"] != h["root.body"]  # both recorded, and they differ


def test_strategy_selection_is_by_declared_category_only(tmp_path):
    a = write_prompt(tmp_path, "strat-needle.v1.md", "NEEDLE BLOCK")
    b = write_prompt(tmp_path, "strat-default.v1.md", "DEFAULT BLOCK")
    root = write_prompt(tmp_path, "root.v1.md", "ROOT")
    reg = PromptRegistry.from_files(root_path=root, leaf_prefix_path=root,
                                    strategy_paths={"needle": a, "default": b})
    reg.load()
    assert "NEEDLE BLOCK" in reg.render_root("needle")
    assert "DEFAULT BLOCK" in reg.render_root("default")
    with pytest.raises(ConfigError):
        reg.render_root("category-the-model-invented")


def test_config_snapshot_is_stable_and_json_serialisable(valid_cfg):
    a = json.dumps(config_snapshot(valid_cfg, {}), sort_keys=False)
    b = json.dumps(config_snapshot(valid_cfg, {}), sort_keys=False)
    assert a == b  # stable field order => stable hashing


def test_snapshot_scrubs_lone_surrogates_before_serialising(valid_cfg):
    snap = config_snapshot(valid_cfg, {"note": "bad" + chr(0xDCFF)})
    json.dumps(snap).encode("utf-8")  # must not raise (D19)


def test_cell_extraction_defaults_match_prompt_promise(valid_cfg):
    assert valid_cfg.scaffold.cell_extraction.select == "first"
    assert valid_cfg.scaffold.cell_extraction.languages[0] == "repl"


# --- Controller rulings (beyond the brief's verbatim test block) ---
#
# Ruling 1: Config.pinned_prompt_hashes() -> dict[str, str] (path -> pinned
# sha256); Task 14's tests call it. Originally config.yaml shipped sha256:
# null for every prompt entry (files didn't exist until Task 14), so
# pinned_prompt_hashes had to skip nulls -- exercised below against an
# explicitly-unpinned entry now that Task 14 has pinned the real config.yaml.
# Ruling 2: scaffold.truncation_cap_chars >= rlm.truncate.MIN_MARKER_CAP is a
# cross-field validator (below that cap the truncator can't emit its marker).


def test_pinned_prompt_hashes_skips_unpinned(minimal_cfg_dict):
    # Task 14 pinned every entry in the real config.yaml, so exercise the
    # skip-when-null behavior directly rather than against valid_cfg.
    minimal_cfg_dict["scaffold"]["prompts"]["leaf_prefix"]["sha256"] = None
    cfg = Config.model_validate(minimal_cfg_dict)
    assert str(cfg.scaffold.prompts.leaf_prefix.path) not in cfg.pinned_prompt_hashes()
    assert str(cfg.scaffold.prompts.root.path) in cfg.pinned_prompt_hashes()  # still pinned


def test_pinned_prompt_hashes_includes_only_pinned(tmp_path, minimal_cfg_dict):
    # A pinned entry must now point at a real, matching file (fix round 1:
    # the Config-level pin check below), so this uses a genuine temp file
    # rather than a fake sha256 against a nonexistent path. leaf_prefix is
    # explicitly unpinned here so the "only pinned" half of the assertion
    # is meaningful even though Task 14 pins it in the real config.yaml.
    f = write_prompt(tmp_path, "root.v1.md", "ROOT")
    real_hash = hashlib.sha256(f.read_bytes()).hexdigest()
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["path"] = str(f)
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["sha256"] = real_hash
    minimal_cfg_dict["scaffold"]["prompts"]["leaf_prefix"]["sha256"] = None
    cfg = Config.model_validate(minimal_cfg_dict)
    hashes = cfg.pinned_prompt_hashes()
    assert hashes[str(cfg.scaffold.prompts.root.path)] == real_hash
    assert str(cfg.scaffold.prompts.leaf_prefix.path) not in hashes


def test_truncation_cap_below_marker_floor_is_refused(minimal_cfg_dict):
    from rlm.truncate import MIN_MARKER_CAP

    minimal_cfg_dict["scaffold"]["truncation_cap_chars"] = MIN_MARKER_CAP - 1
    with pytest.raises(ConfigError, match="truncation_cap_chars"):
        Config.model_validate(minimal_cfg_dict)


# --- Fix round 1/5: Config-level "every prompt path exists and its sha256
# matches the pinned value (when pinned)" validator. PromptRegistry.load()
# alone was opt-in and nothing called it, so a drifted/deleted pinned prompt
# passed load_config() silently. This is a Config-construction-time check. ---


def test_all_null_sha256_validates_fine_even_though_files_dont_exist(tmp_path, minimal_cfg_dict):
    # Proves the no-op property: an entry with sha256: null is never checked
    # for existence, regardless of pinned entries elsewhere. Task 14 pinned
    # every entry in the real config.yaml (files genuinely exist there now),
    # so this points every prompt ref at a path under tmp_path that is never
    # created, with every sha256 explicitly nulled, to exercise the same
    # "unpinned means unchecked" property the real config.yaml demonstrated
    # before Task 14 ran.
    missing = tmp_path / "prompts" / "does-not-exist.md"
    prompts = minimal_cfg_dict["scaffold"]["prompts"]
    prompts["root"] = {"path": str(missing), "sha256": None}
    prompts["leaf_prefix"] = {"path": str(missing), "sha256": None}
    prompts["leaf_envelope"] = {"path": str(missing), "sha256": None}
    # The delegation arm's root prompt is a prompt ref like any other, so the
    # "unpinned means unchecked" property has to hold for it too.
    prompts["root_restricted"] = {"path": str(missing), "sha256": None}
    for cat in prompts["strategy_templates"]:
        prompts["strategy_templates"][cat] = {"path": str(missing), "sha256": None}
    # S4's baseline slots are prompt refs like any other, so they get the same
    # treatment -- and the loop below is the reason this test is the one that
    # notices a new slot enumeration going in unaccompanied.
    for name in prompts.get("baselines", {}):
        prompts["baselines"][name] = {"path": str(missing), "sha256": None}
    cfg = Config.model_validate(minimal_cfg_dict)  # must not raise
    for _, ref in cfg._prompt_refs():
        assert ref.sha256 is None
        assert not ref.path.exists()


def test_pinned_prompt_with_missing_file_is_refused(tmp_path, minimal_cfg_dict):
    missing = tmp_path / "prompts" / "does-not-exist.md"
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["path"] = str(missing)
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["sha256"] = "a" * 64
    with pytest.raises(ConfigError, match="does not exist"):
        Config.model_validate(minimal_cfg_dict)


def test_pinned_prompt_hash_mismatch_is_refused(tmp_path, minimal_cfg_dict):
    f = write_prompt(tmp_path, "root.v1.md", "REAL BODY")
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["path"] = str(f)
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["sha256"] = "0" * 64  # wrong
    with pytest.raises(ConfigError, match="sha256 mismatch"):
        Config.model_validate(minimal_cfg_dict)


def test_pinned_prompt_matching_hash_validates_cleanly(tmp_path, minimal_cfg_dict):
    f = write_prompt(tmp_path, "root.v1.md", "REAL BODY")
    real_hash = hashlib.sha256(f.read_bytes()).hexdigest()
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["path"] = str(f)
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["sha256"] = real_hash
    cfg = Config.model_validate(minimal_cfg_dict)
    assert cfg.scaffold.prompts.root.sha256 == real_hash


# Minor finding, same round: Config.prompt_registry() had no test covering it.
# Kept (rather than deleted) because it is the natural integration point
# Task 14/16 need — a Config already declares everything a PromptRegistry
# needs, so requiring callers to hand-assemble one from cfg.scaffold.prompts
# would just duplicate this mapping at every call site. Covered end-to-end
# below instead of removed.


def test_config_prompt_registry_builds_a_loadable_registry(tmp_path, minimal_cfg_dict):
    root = write_prompt(tmp_path, "root.v1.md", "ROOT BODY")
    leaf_prefix = write_prompt(tmp_path, "leaf-prefix.v1.md", "LEAF PREFIX BODY")
    needle = write_prompt(tmp_path, "strat-needle.v1.md", "NEEDLE BLOCK")
    aggregation = write_prompt(tmp_path, "strat-aggregation.v1.md", "AGG BLOCK")
    synthesis = write_prompt(tmp_path, "strat-synthesis.v1.md", "SYN BLOCK")
    code_qa = write_prompt(tmp_path, "strat-codeqa.v1.md", "CODEQA BLOCK")
    default = write_prompt(tmp_path, "strat-default.v1.md", "DEFAULT BLOCK")

    minimal_cfg_dict["scaffold"]["prompts"] = {
        "root": {"path": str(root), "sha256": None},
        "leaf_prefix": {"path": str(leaf_prefix), "sha256": None},
        "strategy_templates": {
            "needle": {"path": str(needle), "sha256": None},
            "aggregation": {"path": str(aggregation), "sha256": None},
            "synthesis": {"path": str(synthesis), "sha256": None},
            "code_qa": {"path": str(code_qa), "sha256": None},
            "default": {"path": str(default), "sha256": None},
        },
    }
    cfg = Config.model_validate(minimal_cfg_dict)
    reg = cfg.prompt_registry()
    reg.load()
    assert "ROOT BODY" in reg.render_root("default")
    assert "DEFAULT BLOCK" in reg.render_root("default")
    assert reg.leaf_prefix() == "LEAF PREFIX BODY"


# --------------------------------------------------------------------------- #
# R13 (spec v0.2.6 §10): slot policy and the subcall budget it exposed.
# --------------------------------------------------------------------------- #


def test_the_leaf_declares_its_slot_policy_explicitly(valid_cfg):
    """One supported value, declared rather than implied, so the choice is
    greppable: `never_reuse` is the ONLY configuration measured clean
    (virgin slot 0/54 against a shared slot's 24/54, p = 4.4e-9). The options
    it replaces both leak -- `cache_prompt: false` 15/18, `--parallel 1`
    4/18 -- so there is nothing else to offer."""
    assert valid_cfg.servers.leaf.slot_policy == "never_reuse"


def test_an_unmeasured_slot_policy_is_refused(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["leaf"]["slot_policy"] = "reuse_freely"
    with pytest.raises(ConfigError, match="slot_policy"):
        Config.model_validate(minimal_cfg_dict)


def test_the_root_has_no_slot_policy_to_set(minimal_cfg_dict):
    """Slot discipline is the leaf's: the root is one conversation on one
    slot of a `--parallel 1` server, and a policy key there would suggest a
    knob that does nothing."""
    minimal_cfg_dict["servers"]["root"]["slot_policy"] = "never_reuse"
    with pytest.raises(ConfigError, match="slot_policy"):
        Config.model_validate(minimal_cfg_dict)


def test_max_subcalls_covers_a_full_200k_corpus(valid_cfg):
    """§7 #2 (v0.3.0): at window 640 / stride 480 the naive formula gives
    ceil((200,000 - 640) / 480) + 1 = 417 windows = 834 calls, and MEASURING
    the real chunker on 200,000 tokens of fixture prose gives 424 = 848 -- the
    snap shortens gaps. The pinned value is the snap-bounded worst case,
    ceil(200,000 / 432) = 463 windows = 926 calls, so no corpus can turn the
    budget into silent partial coverage. (The same measurement retired the
    previous value: 1,024/768 measured 268 windows against its formula's 261,
    so `max_subcalls: 522` was already 14 calls short of its own geometry.)"""
    assert valid_cfg.scaffold.budgets.max_subcalls == 926
    assert 926 == 2 * -(-200_000 // int(480 * 0.9))


# --------------------------------------------------------------------------- #
# S5/§7 #4: `servers.root.mtp` is a DECLARATION; the flags that actually turn
# MTP on live in extra_flags, because serverproc refuses to invent flags in
# code. The two must not be able to disagree.


def test_mtp_declared_without_the_flag_is_refused(minimal_cfg_dict):
    """A `mtp: true` that emitted nothing at launch would be a silent lie in
    every config_snapshot -- §8's whole comparison rests on the snapshot
    describing the run that actually happened (R11)."""
    minimal_cfg_dict["servers"]["root"]["mtp"] = True
    minimal_cfg_dict["servers"]["root"]["dflash"] = False
    minimal_cfg_dict["servers"]["root"]["parallel"] = 1
    minimal_cfg_dict["servers"]["root"]["extra_flags"] = ["-lm none"]
    with pytest.raises(ConfigError, match="draft-mtp"):
        Config.model_validate(minimal_cfg_dict)


def test_the_flag_without_the_declaration_is_refused(minimal_cfg_dict):
    """The other direction matters too: MTP running while `mtp: false` would
    make §7 #4's optimization invisible to the trace and to scoring."""
    minimal_cfg_dict["servers"]["root"]["mtp"] = False
    minimal_cfg_dict["servers"]["root"]["dflash"] = False
    minimal_cfg_dict["servers"]["root"]["parallel"] = 1
    minimal_cfg_dict["servers"]["root"]["extra_flags"] = [
        "-lm none", "--spec-type draft-mtp", "--spec-draft-n-max 2"]
    with pytest.raises(ConfigError, match="mtp"):
        Config.model_validate(minimal_cfg_dict)


def test_mtp_declared_with_the_flag_validates(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["root"]["mtp"] = True
    minimal_cfg_dict["servers"]["root"]["dflash"] = False
    minimal_cfg_dict["servers"]["root"]["parallel"] = 1
    minimal_cfg_dict["servers"]["root"]["extra_flags"] = [
        "-lm none", "--spec-type draft-mtp", "--spec-draft-n-max 2"]
    cfg = Config.model_validate(minimal_cfg_dict)
    assert cfg.servers.root.mtp is True


def test_the_shipped_config_declares_dflash_and_emits_it(valid_cfg):
    """SUPERSEDES the MTP tripwire (2026-08-19). The shipped root ran
    `draft-mtp` at 2.11x from the S5 swap until DFlash2 shipped a drafter for
    Qwen3.8-27B specifically; measured 1.46x MTP at production sampling
    (s2/DFLASH2.md). Same tripwire intent as before: dropping either half
    silently gives back a 3x on the serial, decode-bound part of every episode."""
    root = valid_cfg.servers.root
    assert root.dflash is True
    assert root.mtp is False, "the two are alternatives; MTP was superseded"
    assert any("draft-dflash" in f for f in root.extra_flags)
    assert not any("draft-mtp" in f for f in root.extra_flags)
    assert root.parallel == 1, "DFlash2 collapses at -np > 1 (llama.cpp#27342)"


def test_the_shipped_config_pins_an_existing_drafter(valid_cfg):
    """DFlash needs a second GGUF on disk, unlike MTP whose head is inside the
    target quant. A missing `-md` is a launch-time death; a stale path is a
    death AFTER the 15.7 GB target has been read. Both are config errors."""
    root = valid_cfg.servers.root
    tokens = [w for f in root.extra_flags for w in f.split()]
    assert "-md" in tokens
    draft = Path(tokens[tokens.index("-md") + 1])
    assert draft.exists(), f"pinned DFlash drafter is missing: {draft}"


def test_the_shipped_config_uses_the_dflash_capable_build(valid_cfg):
    """b10375/b10488 CANNOT load a DFlash2 checkpoint -- 7 unknown tensors and 4
    unknown KV keys, a hard load failure. Pointing the root back at the release
    zip while `dflash: true` stands would fail at launch, not at load, so this
    keeps the two facts adjacent."""
    root = valid_cfg.servers.root
    assert (root.backend_dir / "llama-server.exe").exists()
    assert root.dflash is (root.backend_dir.name == "llamacpp-vulkan-dflash2")


def test_dflash_declared_without_the_flag_is_refused(minimal_cfg_dict):
    """The `draft-mtp`-only version of this check went vacuous the moment the
    root switched methods: `mtp: false` plus DFlash flags satisfied it while
    reintroducing exactly the silent lie it exists to prevent."""
    minimal_cfg_dict["servers"]["root"]["dflash"] = True
    minimal_cfg_dict["servers"]["root"]["parallel"] = 1
    minimal_cfg_dict["servers"]["root"]["extra_flags"] = ["-lm none"]
    with pytest.raises(ConfigError, match="draft-dflash"):
        Config.model_validate(minimal_cfg_dict)


def test_the_dflash_flag_without_the_declaration_is_refused(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["root"]["dflash"] = False
    minimal_cfg_dict["servers"]["root"]["parallel"] = 1
    minimal_cfg_dict["servers"]["root"]["extra_flags"] = [
        "-lm none", "--spec-type draft-dflash", "--spec-draft-n-max 4"]
    with pytest.raises(ConfigError, match="dflash"):
        Config.model_validate(minimal_cfg_dict)


def test_dflash_without_a_draft_model_is_refused(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["root"]["dflash"] = True
    minimal_cfg_dict["servers"]["root"]["parallel"] = 1
    minimal_cfg_dict["servers"]["root"]["extra_flags"] = [
        "-lm none", "--spec-type draft-dflash", "--spec-draft-n-max 4"]
    with pytest.raises(ConfigError, match="-md"):
        Config.model_validate(minimal_cfg_dict)


def test_dflash_with_a_missing_draft_model_is_refused(minimal_cfg_dict):
    """A stale drafter path must fail at config load, not minutes into a launch."""
    minimal_cfg_dict["servers"]["root"]["dflash"] = True
    minimal_cfg_dict["servers"]["root"]["parallel"] = 1
    minimal_cfg_dict["servers"]["root"]["extra_flags"] = [
        "-lm none", "-md D:\\nope\\not-a-drafter.gguf",
        "--spec-type draft-dflash", "--spec-draft-n-max 4"]
    with pytest.raises(ConfigError, match="does not exist"):
        Config.model_validate(minimal_cfg_dict)


def test_dflash_forces_root_single_slot(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["root"]["dflash"] = True
    minimal_cfg_dict["servers"]["root"]["parallel"] = 2
    with pytest.raises(ConfigError, match="dflash"):
        Config.model_validate(minimal_cfg_dict)


def test_the_wall_clock_budget_carries_the_aggregation_ruling(valid_cfg):
    """§8's aggregation deadlock was ruled 2026-08-15: corpora capped at ~130K
    tokens AND this budget raised 900 -> 1300, jointly.

    A tripwire, not a preference. 130,464 tokens is 302 windows at the snap
    bound and needs 1,199 s per episode at the measured 2.78 s/window serial.
    Dropping this back to 900 silently makes every aggregation task in the
    frozen benchmark unaffordable -- they would `budget_kill`, which §8 scores
    as a FAILURE for every arm, so the benchmark would measure the budget
    rather than the roots. 1300 rather than 1200 because 1,199 against 1,200 is
    a one-second margin. See s2/aggregation_options.py for the full pricing.
    """
    assert valid_cfg.scaffold.budgets.max_wall_clock_s == 1300
    # 302 windows x 2 sub-calls must still fit the sub-call budget, or the
    # ruling would have moved the breach rather than removed it.
    assert valid_cfg.scaffold.budgets.max_subcalls >= 604


# --------------------------------------------------------------------------- #
# S4 (§8): the baseline arms' prompts, the B1/B3 relaunch profile, the frozen
# benchmark's pins, and the launch environment.
# --------------------------------------------------------------------------- #


def test_baseline_prompts_load_pin_and_render(valid_cfg):
    reg = valid_cfg.prompt_registry()
    text = reg.render_baseline("b1_single_shot")
    assert "only the answer value" in text
    # The changelog header is registry bookkeeping and must never reach the
    # model: the first draft of these files opened with a one-line comment the
    # loader does not strip, which put "§8 B1 -- single shot, full context" --
    # the name of the experimental arm -- at the top of every B1 prompt.
    assert "<!--" not in text
    hashes = reg.hashes()
    assert "baselines.b1_single_shot.file" in hashes   # recorded => replay-safe


def test_bench_leaf_profile_validates_and_derives_argv(valid_cfg):
    from rlm.serverproc import launch_argv
    bl = valid_cfg.servers.bench_leaf
    assert bl is not None and bl.parallel == 2 and bl.ctx == 524288
    argv = launch_argv(bl)
    assert "-np" in argv and argv[argv.index("-np") + 1] == "2"
    assert argv[argv.index("-c") + 1] == "524288"


def test_benchmark_pins_present(valid_cfg):
    assert valid_cfg.benchmark.manifest_sha256 is not None
    assert valid_cfg.benchmark.escalation_seeds == [4, 5]


def test_config_without_baselines_still_validates(minimal_cfg_dict):
    raw = copy.deepcopy(minimal_cfg_dict)
    raw["scaffold"]["prompts"].pop("baselines", None)
    raw["servers"].pop("bench_leaf", None)
    Config.model_validate(raw)   # pre-S4 configs and snapshots stay loadable


def test_every_baseline_prompt_is_pinned_and_renders(valid_cfg):
    """All four names, not just B1's: a slot that is declared but unreadable
    (or unpinned) is one whose text `config_snapshot` cannot vouch for, and §8
    scores the baselines against RLM on exactly that text."""
    reg = valid_cfg.prompt_registry().load()
    hashes = reg.hashes()
    for name in ("b1_single_shot", "b2_leaf_summary", "b2_root_final",
                 "b3_single_shot"):
        rendered = reg.render_baseline(name)
        assert rendered.strip()
        assert "<!--" not in rendered and "changelog" not in rendered
        assert f"baselines.{name}.file" in hashes
        # …and both hashes are recorded, and they DIFFER -- the same property
        # `test_registry_strips_changelog_header_but_hashes_both` pins for
        # every other prompt. Equal file/body hashes mean nothing was stripped.
        assert hashes[f"baselines.{name}.file"] != hashes[f"baselines.{name}.body"]
    pinned = valid_cfg.pinned_prompt_hashes()
    for _, ref in valid_cfg._prompt_refs():
        assert ref.sha256 is not None, f"{ref.path} is not pinned"
        assert pinned[str(ref.path)] == hashlib.sha256(
            ref.path.read_bytes()).hexdigest()


def test_render_baseline_without_baselines_configured_is_refused(minimal_cfg_dict):
    """A missing `baselines` block must name itself, not KeyError deep inside an
    arm runner half an hour into a bench block."""
    raw = copy.deepcopy(minimal_cfg_dict)
    raw["scaffold"]["prompts"].pop("baselines", None)
    reg = Config.model_validate(raw).prompt_registry()
    with pytest.raises(ConfigError, match="baselines"):
        reg.render_baseline("b1_single_shot")


def test_the_bench_leaf_profile_is_the_shipped_leaf_with_two_full_slots(valid_cfg):
    """§8:344 (v0.2.6 correction). B1 and B3 each get a slot of the leaf's full
    native 262,144-token window, so neither is measured on a truncation the
    scaffold chose for it, and they stop sharing slot 0 (R13's smallest repro).
    `--cont-batching` is deliberately absent: this profile serves exactly two
    sequential single calls per block, and R14 measured continuous batching as
    the mechanism behind concurrent-dispatch corruption."""
    leaf, bench = valid_cfg.servers.leaf, valid_cfg.servers.bench_leaf
    assert bench.model == leaf.model and bench.backend_dir == leaf.backend_dir
    assert bench.backend == leaf.backend and bench.port == leaf.port
    assert bench.ctx // bench.parallel == 262_144
    assert bench.log_path != leaf.log_path, "start() truncates the log (D27)"
    assert "--cont-batching" not in bench.extra_flags
    assert bench.env.get("ROCBLAS_USE_HIPBLASLT") == "1"
    # A plain ServerConfig: slot discipline is the RLM leaf's, and a
    # `slot_policy` here would advertise a knob nothing reads.
    assert not hasattr(bench, "slot_policy")


def test_the_leaf_carries_the_rocblas_env_the_launch_path_used_to_drop(valid_cfg):
    """`s2/run_occupancy.py:455` sets ROCBLAS_USE_HIPBLASLT for every leaf
    measurement recorded so far; the CLI's own launch path passed `env=None`,
    so `--launch-leaf` would have run the S4 blocks against a differently
    configured BLAS than every S2 number they are compared with."""
    assert valid_cfg.servers.leaf.env == {"ROCBLAS_USE_HIPBLASLT": "1"}


def test_the_benchmark_pin_is_the_frozen_manifests_own_hash(valid_cfg):
    """`benchmark.version` names the manifest; this is the number that MOVES if
    any task, corpus hash, checker or precondition result changes. Pinning it in
    config means `config_snapshot` records which freeze an episode was scored
    against, not merely that one existed."""
    from pathlib import Path
    manifest = Path(__file__).resolve().parents[1] / "bench" / "manifest.json"
    assert valid_cfg.benchmark.manifest_sha256 == hashlib.sha256(
        manifest.read_bytes()).hexdigest()
    # The escalation seeds are DISJOINT from the primary ones, or an escalated
    # block would re-run seeds already spent and report the same draw twice.
    assert not (set(valid_cfg.benchmark.escalation_seeds)
                & set(valid_cfg.benchmark.seeds))


def test_max_identical_turns_ships_at_three_and_refuses_one(minimal_cfg_dict):
    """v0.3.16: the shipped config opts in at 3 (correct at 2, kill at 3);
    1 would kill on the first occurrence of any cell -- 0 (disabled) or >= 2."""
    cfg = Config.model_validate(minimal_cfg_dict)
    assert cfg.scaffold.budgets.max_identical_turns == 3
    raw = copy.deepcopy(minimal_cfg_dict)
    raw["scaffold"]["budgets"]["max_identical_turns"] = 1
    with pytest.raises(ConfigError, match="max_identical_turns"):
        Config.model_validate(raw)          # Config.model_validate re-raises pydantic errors as ConfigError
    raw["scaffold"]["budgets"]["max_identical_turns"] = 0
    assert Config.model_validate(raw).scaffold.budgets.max_identical_turns == 0
