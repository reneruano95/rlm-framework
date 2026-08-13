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
