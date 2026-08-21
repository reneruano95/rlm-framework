# tests/test_rootclient.py
import hashlib

from rlm.rootclient import extract_cell, strip_reasoning


def test_strip_reasoning_keeps_the_tail_after_the_last_close_tag():
    text = "<think>\nplan A\n</think>\nmid<think>more</think>\nFINAL"
    assert strip_reasoning(text).strip() == "FINAL"


def test_strip_reasoning_is_a_noop_without_tags():
    assert strip_reasoning("just text").strip() == "just text"


def test_extract_cell_takes_the_first_block_by_default():
    text = "```repl\nA\n```\nprose\n```repl\nB\n```"
    assert extract_cell(text, ["repl", "python", "py"], "first").strip() == "A"


def test_extract_cell_accepts_every_configured_language():
    for lang in ("repl", "python", "py"):
        assert extract_cell(f"```{lang}\nX\n```", ["repl", "python", "py"],
                            "first").strip() == "X"


def test_extract_cell_returns_none_for_zero_blocks_and_unterminated_fences():
    assert extract_cell("no code here", ["repl"], "first") is None
    assert extract_cell("```repl\nunterminated", ["repl"], "first") is None
    assert extract_cell("```ruby\nputs 1\n```", ["repl"], "first") is None


async def test_view_hash_is_the_sha256_of_the_applied_template(fake_root_server):
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("hello")
    turn = await conv.turn()
    assert turn.view_hash == hashlib.sha256(turn.rendered.encode()).hexdigest()
    assert fake_root_server.last_completion_prompt == turn.rendered  # D14


async def test_conversation_growth_is_append_only(fake_root_server):
    """D26: rewriting a sent message collapses prefix-cache reuse."""
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("turn one")
    first = await conv.turn()
    conv.append_user("turn two")
    second = await conv.turn()
    assert second.rendered.startswith(first.rendered.rstrip("\n")[:200])


async def test_thinking_is_disabled_by_default(fake_root_server):
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("hi")
    await conv.turn()
    kw = fake_root_server.last_template_kwargs
    assert kw["chat_template_kwargs"]["enable_thinking"] is False  # D15


async def test_root_sampling_params_reach_the_server(fake_root_server, minimal_cfg_dict):
    """cfg.scaffold.sampling.root must reach the server on every /completion
    call -- real, non-defaulted config (§6 config_snapshot records it as
    what actually ran)."""
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("hi")
    await conv.turn()
    body = fake_root_server.last_completion_body
    expected = minimal_cfg_dict["scaffold"]["sampling"]["root"]
    assert body["temperature"] == expected["temperature"]
    assert body["top_p"] == expected["top_p"]
    assert body["seed"] == expected["seed"] * 1000 + 1  # per_turn schedule (v0.3.16): turn 1


from rlm.rootclient import turn_seed


def test_turn_seed_is_the_base_when_fixed_and_derived_when_per_turn():
    assert turn_seed(2, 1, "fixed") == 2
    assert turn_seed(2, 7, "fixed") == 2
    assert turn_seed(2, 1, "per_turn") == 2001
    assert turn_seed(2, 7, "per_turn") == 2007
    assert turn_seed(3, 1, "per_turn") != turn_seed(2, 1, "per_turn")


async def test_per_turn_schedule_changes_the_seed_every_turn(fake_root_server, minimal_cfg_dict):
    """v0.3.16: the same seed on every turn makes two near-identical turns
    sample identically, which is how a 64% repeat became 70/70 and 111/111 in
    production. With the shipped schedule each turn gets its own seed."""
    conv = fake_root_server.conversation(system="SYS")
    base = minimal_cfg_dict["scaffold"]["sampling"]["root"]["seed"]
    conv.append_user("one")
    await conv.turn()
    first = fake_root_server.last_completion_body["seed"]
    conv.append_user("two")
    await conv.turn()
    second = fake_root_server.last_completion_body["seed"]
    assert (first, second) == (base * 1000 + 1, base * 1000 + 2)


async def test_fixed_schedule_keeps_the_old_behaviour(fake_root_server, minimal_cfg_dict):
    conv = fake_root_server.conversation(system="SYS", seed_schedule="fixed")
    base = minimal_cfg_dict["scaffold"]["sampling"]["root"]["seed"]
    conv.append_user("one")
    await conv.turn()
    conv.append_user("two")
    await conv.turn()
    assert fake_root_server.last_completion_body["seed"] == base
