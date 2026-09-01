import sys

import pytest

from rlm.power import PowerReading, energy_j_between, parse_counter_line


def test_energy_conversion_is_the_s0_recipe():
    a = PowerReading(ts=0.0, energy_pwh=1_000_000_000, power_mw=117_000.0)
    b = PowerReading(ts=10.0, energy_pwh=1_000_325_000, power_mw=117_000.0)
    # 325_000 pWh * 3.6e-9 J/pWh = 1.17e-3 ... scaled: assert exact formula
    assert energy_j_between(a, b) == (325_000) * 3.6e-9


def test_parse_counter_line_extracts_both_counters():
    r = parse_counter_line("1755366000.5,1234567890,117234.0")
    assert r == PowerReading(ts=1755366000.5, energy_pwh=1234567890,
                              power_mw=117234.0)
    assert parse_counter_line("garbage") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows RAPL counter")
def test_live_sampler_smoke():
    from rlm.power import PowerSampler
    s = PowerSampler(interval_s=1.0)
    s.start()
    try:
        import time
        time.sleep(4)
        if not s.alive():      # counter can be absent on other boxes: tolerate
            pytest.skip("Energy Meter counter not available")
        r = s.reading()
        assert r is not None and r.energy_pwh > 0
    finally:
        s.stop()


# --------------------------------------------------------------------------- #
# The child's stderr is a FILE (S4 fix wave 2).
#
# This module's own docstring names the sampler's failure mode: it dies
# silently at launch. `alive()` reports THAT it is dead and never why -- the
# counter set absent, the counter name localised, the Energy Meter provider
# unregistered, an execution policy. Discarding stderr made all of those look
# identical from the outside, on a 39-hour run whose energy column then goes
# NULL for reasons nobody can reconstruct afterwards.
# --------------------------------------------------------------------------- #


def test_the_sampler_writes_its_childs_stderr_to_a_file(tmp_path, monkeypatch):
    import io

    from rlm import power

    captured: dict = {}

    class _Proc:
        def __init__(self, *_a, **kw) -> None:
            captured.update(kw)
            self.stdout = io.StringIO("")

        def kill(self) -> None:
            pass

        def wait(self, timeout=None) -> None:
            pass

    monkeypatch.setattr(power.subprocess, "Popen", _Proc)
    path = tmp_path / "traces" / "power-sampler.err"
    sampler = power.PowerSampler(stderr_path=path)
    sampler.start()

    assert sampler.stderr_path == path
    assert captured["stderr"] is not power.subprocess.DEVNULL
    captured["stderr"].write("Get-Counter : the counter does not exist\n")
    sampler.stop()
    assert "does not exist" in path.read_text(encoding="utf-8")


def test_a_sampler_with_no_stderr_path_keeps_discarding_it(tmp_path, monkeypatch):
    """The parameter is additive: every pre-S4 caller passes nothing."""
    import io

    from rlm import power

    captured: dict = {}

    class _Proc:
        def __init__(self, *_a, **kw) -> None:
            captured.update(kw)
            self.stdout = io.StringIO("")

        def kill(self) -> None:
            pass

        def wait(self, timeout=None) -> None:
            pass

    monkeypatch.setattr(power.subprocess, "Popen", _Proc)
    sampler = power.PowerSampler()
    sampler.start()
    assert captured["stderr"] is power.subprocess.DEVNULL
    assert sampler.stderr_path is None
    sampler.stop()


def test_an_unopenable_stderr_path_does_not_stop_the_sampler(tmp_path, monkeypatch):
    """A diagnostic that cannot be opened must not take the run with it: the
    sampler is optional instrumentation and this is its log file."""
    import io

    from rlm import power

    captured: dict = {}

    class _Proc:
        def __init__(self, *_a, **kw) -> None:
            captured.update(kw)
            self.stdout = io.StringIO("")

        def kill(self) -> None:
            pass

        def wait(self, timeout=None) -> None:
            pass

    monkeypatch.setattr(power.subprocess, "Popen", _Proc)
    directory = tmp_path / "a-directory"
    directory.mkdir()
    sampler = power.PowerSampler(stderr_path=directory)   # open() will refuse
    sampler.start()
    assert captured["stderr"] is power.subprocess.DEVNULL
    sampler.stop()
