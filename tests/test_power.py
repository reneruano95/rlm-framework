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
