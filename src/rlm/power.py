"""S0 item 8's validated Windows package-power collector (milestones/s0/RESULTS.md:73-74):
1 Hz `Get-Counter '\\Energy Meter(rapl_package0_pkg)\\Energy'` — Energy is in
picowatt-hours, Power in mW; energy_j = delta_pWh * 3.6e-9. Overhead measured
+0.56% (noise) => energy_j is ENABLED on this box. The sampler's known failure
mode is dying silently at launch — callers must check alive() and record NULLs,
never fabricated numbers, when it is not."""
from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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
    def __init__(self, interval_s: float = 1.0,
                 stderr_path: str | os.PathLike | None = None) -> None:
        self._interval = interval_s
        #: Where the child's stderr goes. Its KNOWN failure mode is dying
        #: silently at launch (this module's own docstring), and `alive()`
        #: reports THAT it is dead without ever saying why -- the counter set
        #: absent, the counter name localised, the Energy Meter provider not
        #: registered, PowerShell's own execution policy. Discarding stderr
        #: made every one of those look identical from the outside, on a
        #: 39-hour run whose energy column silently becomes NULL. `None` keeps
        #: the old DEVNULL behaviour for callers with nowhere to write.
        self._stderr_path = Path(stderr_path) if stderr_path is not None else None
        self._stderr_file = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._latest: PowerReading | None = None
        self._count = 0
        self._last_growth = 0.0
        self._lock = threading.Lock()

    def _open_stderr(self):
        if self._stderr_path is None:
            return subprocess.DEVNULL
        try:
            self._stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = self._stderr_path.open("w", encoding="utf-8",
                                                        errors="replace")
        except OSError:
            # A diagnostic that cannot be opened must not stop the run: the
            # sampler is optional instrumentation and this is its log file.
            return subprocess.DEVNULL
        return self._stderr_file

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", _PS_SCRIPT],
            stdout=subprocess.PIPE, stderr=self._open_stderr(), text=True)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    @property
    def stderr_path(self) -> "Path | None":
        return self._stderr_path

    def _pump(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            for line in self._proc.stdout:
                r = parse_counter_line(line)
                if r is None:
                    continue
                with self._lock:
                    self._latest, self._count = r, self._count + 1
                    self._last_growth = time.monotonic()
        except (ValueError, OSError):
            # stop() may close stdout out from under a still-blocked
            # readline() if this thread hasn't unblocked within stop()'s
            # join timeout (slow PowerShell teardown). On Windows that
            # surfaces here as "I/O operation on closed file" — expected
            # teardown noise once stop() has been called, not a real
            # failure, so this thread just exits quietly.
            pass

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
        if self._stderr_file is not None:
            with contextlib.suppress(OSError, ValueError):
                self._stderr_file.close()
            self._stderr_file = None
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except (ValueError, OSError):
                # Mirrors the guard in _pump: if the pump thread didn't
                # unblock within the join timeout above, this close() races
                # its still-blocked readline() on the same file object.
                # Closing is still required to eventually unstick a hung
                # reader, so swallow whatever this side of the race raises
                # rather than letting pure teardown timing crash stop().
                pass


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
