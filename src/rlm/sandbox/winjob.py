"""Pure-ctypes Windows Job Object wrapper. No pywin32.

Verified on-box (Recipes §sandbox / Cross-check Conflict 11): a Job memory
cap does NOT kill by itself -- JOB_OBJECT_LIMIT_PROCESS_MEMORY only makes the
child's allocation FAIL (measured: 256 MB cap, child peaked at 232 MB and
kept running). The only thing that turns that into a hard kill is an
IoCompletionPort associated with the job plus TerminateJobObject called from
the notification pump. `Job` sets that pump up unconditionally; per D22 the
pump itself never calls `terminate()` -- it only hands each notification to
whatever callbacks were registered with `.watch()`. The kill decision belongs
one layer up (Task 10 routes it onto the event loop for a single
serialization point with the trace write and the reason string).

KILL_ON_JOB_CLOSE is the other half of the story: closing the job's sole
handle kills the whole process tree unconditionally, which is what makes
`.close()` the one thing a crash-recovery path has to get right.
"""
from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# ---- information-class constants ---------------------------------------- #
JobObjectExtendedLimitInformation = 9
JobObjectAssociateCompletionPortInformation = 7

# ---- JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags ------------------------ #
JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

# completion-port message codes -> legible names (spec §6 outcome_reason)
JOB_MSG = {
    1: "END_OF_JOB_TIME",
    2: "END_OF_PROCESS_TIME",
    3: "ACTIVE_PROCESS_LIMIT",
    4: "ACTIVE_PROCESS_ZERO",
    6: "NEW_PROCESS",
    7: "EXIT_PROCESS",
    8: "ABNORMAL_EXIT_PROCESS",
    9: "PROCESS_MEMORY_LIMIT",
    10: "JOB_MEMORY_LIMIT",
    11: "NOTIFICATION_LIMIT",
}

# Every process in the job raises NEW_PROCESS/EXIT_PROCESS/ACTIVE_PROCESS_ZERO
# as routine lifecycle noise; a watcher exists to react to actionable
# breaches, so only those reach it. (Verified evidence: notifications for a
# 256 MB cap were [('NEW_PROCESS', pid), ('JOB_MEMORY_LIMIT', pid), ...] --
# surfacing NEW_PROCESS too would let a caller's "wait for any notification"
# loop return before the violation it actually cares about ever arrives.)
_ACTIONABLE = frozenset({
    "PROCESS_MEMORY_LIMIT", "JOB_MEMORY_LIMIT", "END_OF_JOB_TIME",
    "END_OF_PROCESS_TIME", "ACTIVE_PROCESS_LIMIT", "ABNORMAL_EXIT_PROCESS",
})


# ---- structs -------------------------------------------------------------- #
class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_ASSOCIATE_COMPLETION_PORT(ctypes.Structure):
    _fields_ = [("CompletionKey", wintypes.LPVOID), ("CompletionPort", wintypes.HANDLE)]


assert ctypes.sizeof(JOBOBJECT_BASIC_LIMIT_INFORMATION) == 64
assert ctypes.sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144

# ---- prototypes ------------------------------------------------------------ #
_LPDW = ctypes.POINTER(wintypes.DWORD)

kernel32.CreateJobObjectW.restype = wintypes.HANDLE
kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]

kernel32.SetInformationJobObject.restype = wintypes.BOOL
kernel32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]

kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

kernel32.TerminateJobObject.restype = wintypes.BOOL
kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]

kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]

kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
kernel32.CreateIoCompletionPort.argtypes = [
    wintypes.HANDLE, wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]

kernel32.GetQueuedCompletionStatus.restype = wintypes.BOOL
kernel32.GetQueuedCompletionStatus.argtypes = [
    wintypes.HANDLE, _LPDW, ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD]


def _check(ok, what: str):
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(err, f"{what} failed: {ctypes.FormatError(err)}")
    return ok


class Job:
    """One Job Object. `.close()` alone kills the whole process tree
    (KILL_ON_JOB_CLOSE); the completion-port pump only ever *notifies* --
    see the module docstring for why terminate() is never called from it.

    `active_process_limit` defaults to 1: the isolation design depends on a
    kernel-level ban on the sandboxed process spawning helpers (closes the
    "spawn a clean interpreter with no audit hook" bypass even if a
    Python-level control were somehow evaded). Pass a higher value or None
    explicitly to relax it; tests that need multiple processes in one job
    do exactly that.
    """

    def __init__(self, *, memory_limit_mb: int | None = None,
                 active_process_limit: int | None = 1,
                 per_process_cpu_s: float | None = None,
                 per_job_cpu_s: float | None = None) -> None:
        self._watchers: list[Callable[[str, float], None]] = []
        self._stop = threading.Event()

        self.handle = kernel32.CreateJobObjectW(None, None)
        _check(self.handle, "CreateJobObjectW")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        if memory_limit_mb is not None:
            limit_bytes = memory_limit_mb * 1024 * 1024
            flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
            info.ProcessMemoryLimit = limit_bytes
            info.JobMemoryLimit = limit_bytes
        if active_process_limit is not None:
            flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = active_process_limit
        if per_process_cpu_s is not None:
            flags |= JOB_OBJECT_LIMIT_PROCESS_TIME
            info.BasicLimitInformation.PerProcessUserTimeLimit = int(
                per_process_cpu_s * 10_000_000)
        if per_job_cpu_s is not None:
            flags |= JOB_OBJECT_LIMIT_JOB_TIME
            info.BasicLimitInformation.PerJobUserTimeLimit = int(
                per_job_cpu_s * 10_000_000)
        info.BasicLimitInformation.LimitFlags = flags
        _check(kernel32.SetInformationJobObject(
            self.handle, JobObjectExtendedLimitInformation, ctypes.byref(info),
            ctypes.sizeof(info)), "SetInformationJobObject")

        self._port = kernel32.CreateIoCompletionPort(INVALID_HANDLE_VALUE, None, None, 0)
        _check(self._port, "CreateIoCompletionPort")
        assoc = JOBOBJECT_ASSOCIATE_COMPLETION_PORT()
        assoc.CompletionKey = ctypes.cast(1, wintypes.LPVOID)
        assoc.CompletionPort = self._port
        _check(kernel32.SetInformationJobObject(
            self.handle, JobObjectAssociateCompletionPortInformation,
            ctypes.byref(assoc), ctypes.sizeof(assoc)), "AssociateCompletionPort")

        self._pump_thread = threading.Thread(target=self._pump, name="job-notify", daemon=True)
        self._pump_thread.start()

    # -- assignment ---------------------------------------------------------- #

    def assign(self, handle: int) -> None:
        """Assign an already-open process HANDLE (e.g. from winproc.spawn's
        SpawnResult). Must be called before the process leaves CREATE_SUSPENDED
        for the assignment to be race-free."""
        _check(kernel32.AssignProcessToJobObject(self.handle, handle),
               "AssignProcessToJobObject")

    def assign_pid(self, pid: int) -> None:
        """Assign by PID (used by tests driving a plain subprocess.Popen)."""
        hp = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION,
            False, pid)
        _check(hp, f"OpenProcess({pid})")
        try:
            self.assign(hp)
        finally:
            kernel32.CloseHandle(hp)

    # -- notifications --------------------------------------------------------- #

    def watch(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback invoked as callback(name, timestamp) on the
        pump thread for every completion-port notification. The pump never
        terminates anything itself (D22) -- that decision belongs to the
        caller, one layer up, at a single serialization point."""
        self._watchers.append(callback)

    def _pump(self) -> None:
        while not self._stop.is_set():
            code = wintypes.DWORD()
            key = ctypes.c_void_p()
            overlapped = ctypes.c_void_p()
            ok = kernel32.GetQueuedCompletionStatus(
                self._port, ctypes.byref(code), ctypes.byref(key),
                ctypes.byref(overlapped), 250)
            if not ok:
                continue
            name = JOB_MSG.get(code.value, str(code.value))
            if name not in _ACTIONABLE:
                continue
            ts = time.time()
            for cb in list(self._watchers):
                try:
                    cb(name, ts)
                except Exception:
                    pass

    # -- lifecycle -------------------------------------------------------------- #

    def terminate(self, exit_code: int = 0xB0DE) -> None:
        """C5's budget-kill primitive: unconditional process-tree kill."""
        if self.handle:
            kernel32.TerminateJobObject(self.handle, exit_code)

    def close(self) -> None:
        """Closing the job's sole handle kills the whole tree
        (KILL_ON_JOB_CLOSE) -- verified: this reaps child AND grandchild, and
        survives a hard TerminateProcess of the scaffold itself."""
        self._stop.set()
        if self._port:
            kernel32.CloseHandle(self._port)
            self._port = None
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "Job":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
