"""AppContainer profile + composite CreateProcessW spawn. Pure ctypes, no
pywin32.

Cross-check Conflict 1 (binding, verified on-box): a Job Object needs the
child suspended before assignment (race-free), and AppContainer needs raw
CreateProcessW to carry PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES.
subprocess.Popen can carry neither. Both attributes DO fit in one
InitializeProcThreadAttributeList(count=2) -- that combination was the one
thing neither probe had run before this plan, and it works: AppContainer
profile create 0.012 s, spawn 0.006 s, full bridge round-trip inside the
container.

CreateProcessW footguns this module exists to hide (measured, not
theorised):
  1. Duplicate handle values in PROC_THREAD_ATTRIBUTE_HANDLE_LIST make
     CreateProcessW fail with ERROR_INVALID_PARAMETER (87). `dedupe_handles`
     is the fix and is public/tested for that reason.
  2. std handles are not appended for you (unlike subprocess.Popen) --
     `spawn`'s caller must build them and pass them in `stdio`, and they
     must also appear in the handle list.
  3. `CreateProcessW` requires the attribute list to stay alive until the
     call returns and to be explicitly deleted afterwards.
  4. The raw Win32 command line is one string, not an argv list: naively
     wrapping only the executable in quotes leaves every other argument
     unquoted, so an argument containing whitespace silently splits into
     multiple argv entries in the child -- corruption, not a crash.
     `subprocess.list2cmdline` implements the MS C runtime's actual
     quoting rules (escaped embedded quotes, doubled trailing backslashes)
     and is used for exactly that reason, not to spawn anything.
  5. If anything after a successful CreateProcessW fails (job assignment,
     ResumeThread), the child is left CREATE_SUSPENDED forever unless it is
     explicitly torn down -- a silent, permanent orphan per failure.
"""
from __future__ import annotations

import ctypes
import datetime as dt
import subprocess  # list2cmdline only: the MS C runtime argv-quoting rules,
                    # not for spawning -- getting this wrong silently
                    # corrupts argv instead of raising.
from ctypes import wintypes
from typing import NamedTuple

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
userenv = ctypes.WinDLL("userenv", use_last_error=True)

# ---- flags/constants ------------------------------------------------------- #
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
STARTF_USESTDHANDLES = 0x00000100

PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001


# ---- structs ----------------------------------------------------------------- #
class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [("AppContainerSid", wintypes.LPVOID),
                ("Capabilities", ctypes.POINTER(SID_AND_ATTRIBUTES)),
                ("CapabilityCount", wintypes.DWORD),
                ("Reserved", wintypes.DWORD)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE)]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", wintypes.LPVOID)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


# ---- prototypes --------------------------------------------------------------- #
kernel32.CreateProcessW.restype = wintypes.BOOL
kernel32.CreateProcessW.argtypes = [
    wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
    wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
    ctypes.POINTER(STARTUPINFOEXW), ctypes.POINTER(PROCESS_INFORMATION)]

kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
kernel32.InitializeProcThreadAttributeList.argtypes = [
    ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)]

kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
kernel32.UpdateProcThreadAttribute.argtypes = [
    ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]

kernel32.DeleteProcThreadAttributeList.restype = None
kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]

kernel32.ResumeThread.restype = wintypes.DWORD
kernel32.ResumeThread.argtypes = [wintypes.HANDLE]

kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]

kernel32.TerminateProcess.restype = wintypes.BOOL
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]

kernel32.GetProcessTimes.restype = wintypes.BOOL
kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4

kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]

kernel32.GetExitCodeProcess.restype = wintypes.BOOL
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]


def _check(ok, what: str):
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(err, f"{what} failed: {ctypes.FormatError(err)}")
    return ok


def dedupe_handles(handles: list[int]) -> list[int]:
    """Order-preserving dedupe. D2: duplicate values in
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST make CreateProcessW fail with
    ERROR_INVALID_PARAMETER (87) -- measured the moment stdout and stderr
    shared one file handle."""
    return list(dict.fromkeys(handles))


class AppContainer:
    """One AppContainer profile, minted fresh per episode. CapabilityCount=0
    (no internetClient etc.) blocks outbound TCP/DNS/raw-FFI networking and
    loopback-to-other-processes, and confines the filesystem -- all
    kernel-enforced, verified on-box."""

    def __init__(self) -> None:
        self._name: str | None = None

    def create(self, name: str) -> int:
        """Create-or-derive the profile's SID. Returns the raw PSID pointer
        value (an int) ready to drop into SECURITY_CAPABILITIES.AppContainerSid
        -- no string round-trip needed. ~0.02 s, no admin required."""
        sid_ptr = wintypes.LPVOID()
        hr = userenv.CreateAppContainerProfile(
            name, name, "rlm-halo episode sandbox", None, 0, ctypes.byref(sid_ptr))
        if hr != 0:
            hr = userenv.DeriveAppContainerSidFromAppContainerName(
                name, ctypes.byref(sid_ptr))
            if hr != 0:
                raise OSError(
                    f"AppContainer profile unavailable for {name!r}: "
                    f"hr=0x{hr & 0xffffffff:08x}")
        self._name = name
        return sid_ptr.value

    def delete(self) -> None:
        """DeleteAppContainerProfile does NOT remove ACEs naming that
        profile's SID -- grant filesystem access to ALL_APPLICATION_PACKAGES
        once at install time, never to a per-episode SID (Cross-check
        Conflict 8), or this accrues one dead ACE per episode."""
        if self._name is not None:
            userenv.DeleteAppContainerProfile(self._name)
            self._name = None


class Stdio(NamedTuple):
    """Raw, already-open, inheritable handle values for the child's stdio.
    The caller creates and owns these (e.g. NUL for stdin, a per-episode log
    file for stdout/stderr) -- `spawn` only wires them in."""

    stdin: int
    stdout: int
    stderr: int


class SpawnResult(NamedTuple):
    pid: int
    hprocess: int
    hthread: int


def spawn(exe: str, args: list[str], handles: list[int],
          appcontainer_sid: int | None, job, env: dict[str, str] | None,
          stdio: Stdio, *, cwd: str | None = None) -> SpawnResult:
    """CreateProcessW(CREATE_SUSPENDED) -> job.assign(hProcess) ->
    ResumeThread. One InitializeProcThreadAttributeList carries BOTH
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST and (if `appcontainer_sid` is given)
    PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES -- verified on-box as the
    only version that carries both attributes at once (Cross-check
    Conflict 1). The suspended create + job-assign-before-resume is what
    makes assignment race-free; never subprocess.Popen here.

    `job.assign(hProcess)` is called BEFORE ResumeThread, not after --
    that ordering is the whole point.

    The caller (not this function) owns creating and closing the pipe/file
    handles that go into `handles`/`stdio`; per the bridge mechanics, the
    caller must close its copies of the child's pipe ends immediately after
    this returns, or it will never observe child EOF.
    """
    n_attr = 2 if appcontainer_sid else 1
    size = ctypes.c_size_t(0)
    kernel32.InitializeProcThreadAttributeList(None, n_attr, 0, ctypes.byref(size))
    buf = (ctypes.c_byte * size.value)()
    _check(kernel32.InitializeProcThreadAttributeList(buf, n_attr, 0, ctypes.byref(size)),
           "InitializeProcThreadAttributeList")

    try:
        dedup = dedupe_handles([*handles, stdio.stdin, stdio.stdout, stdio.stderr])
        hlist = (wintypes.HANDLE * len(dedup))(*dedup)
        _check(kernel32.UpdateProcThreadAttribute(
            buf, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, hlist, ctypes.sizeof(hlist),
            None, None), "UpdateProcThreadAttribute(HANDLE_LIST)")

        caps = None
        if appcontainer_sid:
            caps = SECURITY_CAPABILITIES(appcontainer_sid, None, 0, 0)
            _check(kernel32.UpdateProcThreadAttribute(
                buf, 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, ctypes.byref(caps),
                ctypes.sizeof(caps), None, None),
                "UpdateProcThreadAttribute(SECURITY_CAPABILITIES)")

        si = STARTUPINFOEXW()
        si.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES
        si.StartupInfo.hStdInput = stdio.stdin
        si.StartupInfo.hStdOutput = stdio.stdout
        si.StartupInfo.hStdError = stdio.stderr
        si.lpAttributeList = ctypes.cast(buf, wintypes.LPVOID)

        envblock = None
        envbuf = None
        if env is not None:
            blk = "".join(f"{k}={v}\0" for k, v in env.items()) + "\0"
            envbuf = ctypes.create_unicode_buffer(blk)
            envblock = ctypes.cast(envbuf, ctypes.c_void_p)

        # MS C runtime argv-quoting rules (escaped embedded quotes, doubled
        # trailing backslashes before a closing quote): a naive `f'"{a}"'`
        # wrap only protects args with no embedded quote/backslash, and does
        # nothing for the *other* args in the list at all -- an unquoted arg
        # containing whitespace silently splits into multiple argv entries
        # in the child instead of raising. list2cmdline implements exactly
        # the rules CreateProcessW's own argv parser expects.
        cmdline = ctypes.create_unicode_buffer(subprocess.list2cmdline([exe, *args]))
        pi = PROCESS_INFORMATION()
        flags = (CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT
                 | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW)
        _check(kernel32.CreateProcessW(
            exe, cmdline, None, None, True, flags, envblock, cwd,
            ctypes.byref(si), ctypes.byref(pi)), "CreateProcessW")
    finally:
        kernel32.DeleteProcThreadAttributeList(buf)

    try:
        job.assign(pi.hProcess)  # BEFORE ResumeThread -- race-free by construction
        resumed = kernel32.ResumeThread(pi.hThread)
        _check(resumed != 0xFFFFFFFF, "ResumeThread")
    except BaseException:
        # CreateProcessW succeeded but something after it failed (job.assign
        # raised, or ResumeThread itself failed): the child is suspended and
        # would otherwise be a permanent orphan -- holding two open handles
        # and never running, never exiting, never reaped.
        kernel32.TerminateProcess(pi.hProcess, 1)
        kernel32.CloseHandle(pi.hThread)
        kernel32.CloseHandle(pi.hProcess)
        raise
    return SpawnResult(pid=pi.dwProcessId, hprocess=pi.hProcess, hthread=pi.hThread)


# --------------------------------------------------------------------------- #
# §6 crash recovery
# --------------------------------------------------------------------------- #

def _process_create_time(pid: int) -> dt.datetime | None:
    """UTC creation time of a RUNNING `pid`; None if absent or already
    exited. The exit-time check is load-bearing: while a handle is held,
    OpenProcess keeps succeeding for an already-terminated process, so
    handle-openability alone would report zombies as alive."""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
                h, ctypes.byref(creation), ctypes.byref(exit_),
                ctypes.byref(kernel), ctypes.byref(user)):
            return None
        exit_ticks = (exit_.dwHighDateTime << 32) | exit_.dwLowDateTime
        if exit_ticks:
            return None  # already exited; a live handle does not mean alive
        creation_ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return dt.datetime(1601, 1, 1) + dt.timedelta(microseconds=creation_ticks / 10)
    finally:
        kernel32.CloseHandle(h)


def kill_if_ours(pid: int, started_at: dt.datetime, *, slack_s: float = 120.0) -> bool:
    """§6 recovery-scan primitive. PID reuse is real (post-reboot or a busy
    hour): never kill a bare pid without checking it was actually created
    around `started_at` (naive UTC, matching episodes.started_at) first.

    Returns True if there is nothing left to worry about for this pid --
    either it was not running at all, or it matched and was terminated.
    Returns False if a live process exists at this pid but its creation
    time does not match the episode (pid reused by something unrelated:
    refuse to touch it) or the kill attempt itself failed.
    """
    created = _process_create_time(pid)
    if created is None:
        return True
    if not (started_at - dt.timedelta(seconds=slack_s) <= created
            <= started_at + dt.timedelta(seconds=slack_s)):
        return False
    h = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
    if not h:
        return False
    try:
        return bool(kernel32.TerminateProcess(h, 1))
    finally:
        kernel32.CloseHandle(h)
