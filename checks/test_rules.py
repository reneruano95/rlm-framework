from bench.rules import BenchmarkRules


class _M:
    def __init__(self, rules): self.rules = rules


def test_v1_defaults_reproduce_the_pre_registered_constants():
    r = BenchmarkRules.for_manifest(_M(None))
    assert r.rlm_arm == "rlm" and r.baselines == ("b1", "b2", "b3")
    assert r.arms == ("rlm", "b1", "b2", "b3")
    assert r.margin == 3 and r.escalation_band == (1, 2, 3) and r.tripwire_floor == 3
    assert r.abstentions == {} and r.scored_stream is None


def test_v2_rules_are_read_from_the_manifest():
    r = BenchmarkRules.for_manifest(_M({
        "rlm_arm": "rlm", "baselines": ["rlm-nosubcalls", "b2"], "margin": 2,
        "escalation_band": [1, 2], "abstentions": {"b2": ["interactive"]},
        "scored_stream": "train", "n_tasks": 16}))
    assert r.baselines == ("rlm-nosubcalls", "b2") and r.margin == 2
    assert r.escalation_band == (1, 2)
    assert r.abstains("b2", "interactive") and not r.abstains("b2", "linear_semantic")
    assert not r.abstains("rlm", "interactive")
    assert r.scored_stream == "train" and r.n_tasks == 16


def test_unknown_rule_keys_are_refused():
    import pytest
    with pytest.raises(ValueError, match="unknown rule"):
        BenchmarkRules.for_manifest(_M({"margn": 2}))
