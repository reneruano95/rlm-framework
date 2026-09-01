"""The one seam this reorganization committed to building, and why it is one.

`Dispatcher` was refused three times during planning as speculative generality, and
rightly: a Protocol with one implementation is ceremony. It earns its place here
because it has two, and because writing it EXPOSED that the second did not actually
conform.

Measured 2026-09-01: `MockDispatcher` implemented four of the seven members
`episode.py` calls on `self.dispatcher`. The three it lacked -- `restart_required`,
`rotating`, `rotate_pool` -- are read UNGUARDED at `episode.py:703`, `:718` and
`:722`, so a mock handed to `run_episode` raised AttributeError the moment a leak or
a slot exhaustion took the rotation branch. It looked fine only because no test using
it had ever reached that branch. `MockDispatcher` is exported as public API, so the
package was advertising a substitute that was one four sevenths of the time.
"""
import pytest

from rlm.errors import DispatchError
from rlm.serve.dispatcher import Dispatcher, LLMDispatcher, MockDispatcher

#: Not a wish list: this is what an AST pass over episode.py found it calling on
#: `self.dispatcher`. Adding a name here means episode.py started using it.
EPISODE_REQUIRES = (
    "query", "count_tokens", "set_corpus", "steps",
    "restart_required", "rotating", "rotate_pool",
)


def test_the_protocol_names_exactly_what_episode_uses():
    """A protocol wider than its caller is speculation; narrower is a lie.

    Both halves are read: `dir()` finds the methods, and `__annotations__` finds the
    data members, which are bare annotations with no value and are therefore invisible
    to `dir()`. Enumerating only the first would have silently excused a missing
    `steps` -- exactly the kind of half-check this file exists to prevent.
    """
    declared = {n for n in dir(Dispatcher) if not n.startswith("_")}
    declared |= {n for n in getattr(Dispatcher, "__annotations__", {}) if not n.startswith("_")}
    assert declared == set(EPISODE_REQUIRES)


@pytest.mark.parametrize("name", EPISODE_REQUIRES)
def test_the_mock_carries_every_member(name):
    assert hasattr(MockDispatcher({}), name)


@pytest.mark.parametrize("name", EPISODE_REQUIRES)
def test_the_real_dispatcher_carries_every_member(name):
    # `steps` is set per-instance on both; the rest are class-level.
    assert hasattr(LLMDispatcher, name) or name == "steps"


def test_isinstance_holds_for_the_double():
    """Runtime-checkable so a consumer's third implementation can assert conformance
    instead of discovering a gap by hitting it in production."""
    assert isinstance(MockDispatcher({}), Dispatcher)


def test_the_mock_refuses_to_rotate_rather_than_pretending():
    """The limit of `isinstance` against a Protocol is that it checks names, not
    behaviour. A mock that quietly accepted `rotate_pool` would let a test assert
    R13's never-reuse policy against a pool that never existed -- green about nothing.
    So the double raises, and this pins that it keeps raising."""
    with pytest.raises(DispatchError, match="no slot pool"):
        MockDispatcher({}).rotate_pool()


def test_the_mock_never_requires_a_restart():
    assert MockDispatcher({}).restart_required is False
