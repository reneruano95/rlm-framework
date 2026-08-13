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
# sha256); Task 14's tests call it. config.yaml ships with sha256: null for
# every prompt entry (files don't exist until Task 14), so pinned_prompt_hashes
# must skip nulls.
# Ruling 2: scaffold.truncation_cap_chars >= rlm.truncate.MIN_MARKER_CAP is a
# cross-field validator (below that cap the truncator can't emit its marker).


def test_pinned_prompt_hashes_skips_unpinned(valid_cfg):
    assert valid_cfg.pinned_prompt_hashes() == {}


def test_pinned_prompt_hashes_includes_only_pinned(tmp_path, minimal_cfg_dict):
    # A pinned entry must now point at a real, matching file (fix round 1:
    # the Config-level pin check below), so this uses a genuine temp file
    # rather than a fake sha256 against a nonexistent path.
    f = write_prompt(tmp_path, "root.v1.md", "ROOT")
    real_hash = hashlib.sha256(f.read_bytes()).hexdigest()
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["path"] = str(f)
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["sha256"] = real_hash
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


def test_all_null_sha256_validates_fine_even_though_files_dont_exist(valid_cfg):
    # Proves the no-op property: today's config.yaml pins nothing, and none
    # of prompts/*.md exist yet (Task 14 writes them) — valid_cfg's successful
    # construction (via the fixture) already demonstrates this, so also assert
    # the precondition directly rather than just trusting the fixture.
    for _, ref in valid_cfg._prompt_refs():
        assert ref.sha256 is None
        assert not ref.path.exists()


def test_pinned_prompt_with_missing_file_is_refused(minimal_cfg_dict):
    minimal_cfg_dict["scaffold"]["prompts"]["root"]["sha256"] = "a" * 64
    # path is left as "prompts/root.v1.md", which does not exist yet.
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
