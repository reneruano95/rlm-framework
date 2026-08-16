"""S0 item 8's validated Windows package-power collector (s0/RESULTS.md:73-74):
1 Hz `Get-Counter '\\Energy Meter(rapl_package0_pkg)\\Energy'` — Energy is in
picowatt-hours, Power in mW; energy_j = delta_pWh * 3.6e-9. Overhead measured
+0.56% (noise) => energy_j is ENABLED on this box. The sampler's known failure
mode is dying silently at launch — callers must check alive() and record NULLs,
never fabricated numbers, when it is not."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

_PS_SCRIPT = (
    "$ErrorActionPreference='Stop';"
    "Get-Counter -Counter '\\Energy Meter(rapl_package0_pkg)\\Energy',"
    "'\\Energy Meter(rapl_package0_pkg)\\Power' -SampleInterval 1 -Continuous |"
    " ForEach-Object { $e=$_.CounterSamples[0].CookedValue;"
    " $p=$_.CounterSamples[1].CookedValue;"
    " $ts=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0;"
    " Write-Output ('{0},{1},{2}' -f $ts, [long]$e, $p) }"
)


@dataclass(frozen=True)
class PowerReading:
    ts: float
    energy_pwh: int
    power_mw: float


def parse_counter_line(line: str) -> PowerReading | None:
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        return PowerReading(ts=float(parts[0]), energy_pwh=int(parts[1]),
                             power_mw=float(parts[2]))
    except ValueError:
        return None


def energy_j_between(a: PowerReading, b: PowerReading) -> float:
    return (b.energy_pwh - a.energy_pwh) * 3.6e-9


class PowerSampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self._interval = interval_s
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._latest: PowerReading | None = None
        self._count = 0
        self._last_growth = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", _PS_SCRIPT],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            r = parse_counter_line(line)
            if r is None:
                continue
            with self._lock:
                self._latest, self._count = r, self._count + 1
                self._last_growth = time.monotonic()

    def reading(self) -> PowerReading | None:
        with self._lock:
            return self._latest

    def alive(self) -> bool:
        with self._lock:
            return (self._latest is not None
                    and time.monotonic() - self._last_growth < 5.0)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        if proc.stdout is not None:
            proc.stdout.close()


def read_pkg_temp_c() -> float | None:
    """One-shot ACPI thermal-zone read (R9 per-episode record; not gated on the
    power-overhead check — ARCHITECTURE.md:195)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance -Namespace root/wmi MSAcpi_ThermalZoneTemperature"
             " | Select-Object -First 1).CurrentTemperature"],
            capture_output=True, text=True, timeout=15)
        raw = out.stdout.strip()
        return round(int(raw) / 10.0 - 273.15, 1) if raw.isdigit() else None
    except Exception:  # noqa: BLE001 — a temp read must never kill a bench run
        return None
