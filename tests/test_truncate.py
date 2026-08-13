from hypothesis import given, settings
from hypothesis import strategies as st

from rlm.truncate import CellOutput, observation_view, truncate_view

CAP = 2000


def test_labeled_order_is_stdout_stderr_repr_traceback():
    out = CellOutput(stdout="A", stderr="B", repr_="C", traceback="D")
    view = observation_view(out, CAP)
    assert view.index("A") < view.index("B") < view.index("C") < view.index("D")
    assert "[stdout]" in view and "[stderr]" in view
    assert "[repr]" in view and "[traceback]" in view


def test_empty_sections_are_omitted():
    out = CellOutput(stdout="only", stderr="", repr_="", traceback="")
    view = observation_view(out, CAP)
    assert "[stderr]" not in view
    assert "only" in view


def test_marker_reports_true_total_and_cap():
    out = CellOutput(stdout="x" * 184_203, stderr="", repr_="", traceback="")
    view = observation_view(out, CAP)
    assert len(view) <= CAP
    assert "[truncated: showing" in view
    assert "184," in view  # thousands-separated true total appears


def test_truncation_is_applied_to_the_concatenated_unit_not_per_stream():
    """A huge stdout must consume the budget so later streams are cut, not each
    stream getting its own allowance."""
    out = CellOutput(stdout="x" * 10_000, stderr="NEEDLE", repr_="", traceback="")
    view = observation_view(out, CAP)
    assert "NEEDLE" not in view
    assert len(view) <= CAP


@settings(max_examples=300, deadline=None)
@given(
    st.text(max_size=5000),
    st.text(max_size=5000),
    st.text(max_size=5000),
    st.text(max_size=5000),
    st.integers(min_value=50, max_value=4000),
)
def test_property_view_never_exceeds_cap_and_is_deterministic(a, b, c, d, cap):
    out = CellOutput(stdout=a, stderr=b, repr_=c, traceback=d)
    first = observation_view(out, cap)
    second = observation_view(out, cap)
    assert first == second
    assert len(first) <= cap


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=0, max_size=20_000), st.integers(min_value=50, max_value=3000))
def test_property_marker_itself_is_never_truncated(text, cap):
    view = truncate_view(text, cap)
    if len(text) > cap:
        assert view.endswith("chars]")
        assert view.count("[truncated: showing") == 1
    assert len(view) <= cap


def test_pathological_inputs_survive():
    for payload in ["\x00" * 5000, "🧨" * 5000, "x" * 1_000_000, "\r\n" * 5000]:
        out = CellOutput(stdout=payload, stderr="", repr_="", traceback="")
        view = observation_view(out, CAP)
        assert len(view) <= CAP
