from hypothesis import given, settings
from hypothesis import strategies as st

from rlm.truncate import MIN_MARKER_CAP, CellOutput, observation_view, truncate_view

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
    st.integers(min_value=0, max_value=4000),
)
def test_property_view_never_exceeds_cap_and_is_deterministic(a, b, c, d, cap):
    out = CellOutput(stdout=a, stderr=b, repr_=c, traceback=d)
    first = observation_view(out, cap)
    second = observation_view(out, cap)
    assert first == second
    assert len(first) <= cap


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=0, max_size=20_000), st.integers(min_value=0, max_value=3000))
def test_property_marker_itself_is_never_truncated(text, cap):
    view = truncate_view(text, cap)
    assert len(view) <= cap
    if len(text) > cap and cap >= MIN_MARKER_CAP:
        assert view.endswith("chars]")
        assert view.count("[truncated: showing") == 1
        if cap == MIN_MARKER_CAP:  # head budget is zero: view is marker-only
            assert view.startswith("[truncated: showing")
    elif 0 < cap < MIN_MARKER_CAP and len(text) > cap:
        assert "[truncated" not in view
        assert "chars]" not in view


def test_regression_cap_zero_returns_empty():
    assert truncate_view("x" * 100, 0) == ""


def test_regression_cap_ten_is_head_only_no_marker_fragment():
    view = truncate_view("x" * 100, 10)
    assert view == "x" * 10
    assert len(view) <= 10
    assert "[truncated" not in view
    assert "chars]" not in view


def test_regression_cap_thirty_is_head_only_no_marker_fragment():
    view = truncate_view("x" * 100, 30)
    assert view == "x" * 30
    assert len(view) <= 30
    assert "[truncated" not in view
    assert "chars]" not in view


def test_pathological_inputs_survive():
    for payload in ["\x00" * 5000, "🧨" * 5000, "x" * 1_000_000, "\r\n" * 5000]:
        out = CellOutput(stdout=payload, stderr="", repr_="", traceback="")
        view = observation_view(out, CAP)
        assert len(view) <= CAP
