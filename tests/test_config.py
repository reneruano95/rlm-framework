import hashlib
import json
import textwrap

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


def test_leaf_ctx_must_equal_parallel_times_slot(minimal_cfg_dict):
    bad = minimal_cfg_dict
    bad["servers"]["leaf"]["ctx"] = 12345
    with pytest.raises(ConfigError, match="ctx"):
        Config.model_validate(bad)


def test_semaphore_equals_leaf_parallel(valid_cfg):
    assert valid_cfg.scaffold.dispatch_concurrency == valid_cfg.servers.leaf.parallel


def test_mtp_forces_root_single_slot(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["root"]["mtp"] = True
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
    for cat in prompts["strategy_templates"]:
        prompts["strategy_templates"][cat] = {"path": str(missing), "sha256": None}
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
    """s2/R13-mitigations.md §8.3: at window 1,024 / stride 768 a 200K-token
    corpus is ceil((200,000 - 1,024) / 768) + 1 = 261 windows, and the budget
    is spent per QUESTION, not per window -- 2 questions each is 522 calls.
    The old default of 32 covered 1,024 + 31 x 768 = 24,832 tokens, i.e.
    12.4% of the corpus, and coverage broke silently rather than loudly."""
    windows = -(-(200_000 - 1_024) // 768) + 1
    assert windows == 261
    assert valid_cfg.scaffold.budgets.max_subcalls == 2 * windows == 522
