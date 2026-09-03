import re

from bench.adversary import parser_adversary, self_read_adversary
from bench.corpus_v2 import build_linear_semantic, load_trec
from bench.tokens import approx_tokens
from rlm.context.chunker import ChunkConfig

CFG = ChunkConfig(size_tokens=640, overhead_tokens=1920, snap_to_boundary=True,
                  snap_tolerance=0.10, stride_tokens=480)


def test_parser_adversary_scores_every_strategy_and_reports_chance():
    c = build_linear_semantic(9101, 20_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec())
    scores = parser_adversary(c)
    assert "__chance__" in scores and "wh_word_rules" in scores and "label_lexicon" in scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_the_wh_word_rule_is_a_real_adversary_on_trec():
    """If this ever scores at chance the adversary is broken, not TREC: 'who' is HUM."""
    c = build_linear_semantic(9101, 20_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec(), paraphrase=False)
    assert parser_adversary(c)["wh_word_rules"] > 0.5


def test_the_paraphrased_register_takes_the_wh_word_rule_to_chance():
    c = build_linear_semantic(9101, 20_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec(), paraphrase=True)
    scores = parser_adversary(c)
    chance = scores.pop("__chance__")
    beaten = {k: v for k, v in scores.items() if v > chance + 0.02}
    assert not beaten, beaten
    assert not any(q.split()[0].lower() in {"who", "where", "when", "what", "which", "how", "why"}
                   for q in re.findall(r"^Query: (.+)$", c.text, re.M))


def test_self_read_adversary_counts_the_windows_the_answer_needs():
    c = build_linear_semantic(9101, 60_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec())
    n = self_read_adversary(c, CFG, approx_tokens, k=40)
    assert n > 40                                   # ~139 windows at 60K tokens
    small = build_linear_semantic(9101, 6_000, approx_tokens, "approx-offline",
                                  question_kind="count_label", items=load_trec())
    assert self_read_adversary(small, CFG, approx_tokens, k=40) <= 40   # a 6K corpus IS self-readable
