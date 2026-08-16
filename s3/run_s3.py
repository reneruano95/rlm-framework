"""S3 gate runner — budgets + tracing (ARCHITECTURE.md §9 S3).

Gate text, applied mechanically:

  "Adversarial self-test with the mock dispatcher: a task that infinite-loops
   in the REPL and one that requests unbounded sub-calls; plus a hard-kill
   mid-episode to verify the C6 durability promise (R10) **and** the
   post-restart tombstone (§6 crash recovery).
   Gate: all terminate deterministically within budget, and `rlm replay`
   verifies the full trajectory of any episode from the trace store alone
   (DuckDB + referenced blob directory, episode-relative paths — per-step
   root_view_hash equality plus transcript render), with the lifecycle log
   deleted — no logs, no stdout."

Plus the check §5 C1 assigns here: the no-egress audit — one
Get-NetTCPConnection query against the live sandbox pid must show nothing
but the sandbox's own self-connected loopback pair.

Design decisions, stated:
- Every episode runs through the REAL operator surface: `rlm run` (cmd_run:
  recover -> run_episode -> print) in a SEPARATE OS process. The hard-kill
  leg requires a separate process anyway (TerminateProcess on the scaffold,
  then the Job Object's KILL_ON_JOB_CLOSE must reap the sandbox), and the
  other legs get CLI fidelity for free.
- The root is the scripted FakeRootServer from tests/conftest.py — the same
  canned-response mechanism the whole episode test suite runs on, §5's
  dry-run mode ("canned responses ... real sandbox, real C1/C2/C3/C5/C6").
  The leaf is MockDispatcher, fed from each task file's own `fixtures` key
  (rlm/cli.py:_build_dispatcher).
- Determinism is asserted by repetition: the two adversarial tasks run
  N_ATTEMPTS times each and every attempt must record the identical
  (outcome, outcome_reason) pair, read back from the trace store, never from
  process memory (s1 convention: if a number is not in the trace it does not
  go in the report).
- An interrupted gate must never cost the attempts that already ran:
  one JSONL record per check appended to s3/results/runs.jsonl; the report
  phase re-reads that file and the trace store.

Phases: --phase {runaway,subcalls,hardkill,replay,report,all}
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
import uuid as uuid_mod
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from conftest import FakeRootServer, _episode_cfg_dict  # noqa: E402  (tests/conftest.py)

S3_DIR = REPO_ROOT / "s3"
TASKS_DIR = S3_DIR / "tasks"
RESULTS_DIR = S3_DIR / "results"
STORE_DIR = RESULTS_DIR / "store"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"
RESULTS_MD = S3_DIR / "RESULTS.md"
DB_PATH = STORE_DIR / "rlm.duckdb"
BLOB_ROOT = STORE_DIR / "blobs"
LIFECYCLE_LOG = STORE_DIR / "lifecycle.jsonl"

PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
N_ATTEMPTS = 3

# Budgets per scenario. Wall clocks are generous enough that the intended
# budget is the one that trips, tight enough that a hung scenario fails the
# runner instead of the operator's patience.
RUNAWAY_WALL_S = 20
SUBCALLS_WALL_S = 120
SUBCALLS_CAP = 8
HARDKILL_WALL_S = 900
# ended_at - started_at may exceed the budget by kill/teardown time (job
# terminate, bridge close, cancelled-step write, close_episode). Measured
# slack on this box is ~2 s; 30 s is the mechanical bound we assert.
KILL_TEARDOWN_SLACK_S = 30

NARRATIVE_MARKER = "<!-- HAND-WRITTEN FINDINGS BELOW — regeneration preserves this -->"


# --------------------------------------------------------------------------- #
# record helpers (s1/run_s1.py conventions)
# --------------------------------------------------------------------------- #


def append_run(record: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with RUNS_PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[s3] {record.get('check')}: {record.get('verdict', record.get('note', ''))}")
    return record


def read_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    return [json.loads(line) for line in RUNS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def db_rows(sql: str, params: list | None = None) -> list[dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cur = con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [{c: (str(v) if isinstance(v, uuid_mod.UUID) else v) for c, v in zip(cols, r)}
                for r in cur.fetchall()]
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# scenario plumbing
# --------------------------------------------------------------------------- #


def base_raw() -> dict:
    return yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))


def write_cfg(name: str, server_port: int, **over) -> Path:
    """The shipped config re-pointed at the scripted root and the s3 store —
    tests/conftest._episode_cfg_dict is the single source of that surgery, so
    this gate and the test suite cannot drift apart."""
    raw = _episode_cfg_dict(base_raw(), tmp_path=STORE_DIR, root_port=server_port, **over)
    raw["trace"]["db_path"] = str(DB_PATH)
    raw["trace"]["blob_root"] = str(BLOB_ROOT)
    # Bundles prove nothing here and the killed process never writes one;
    # keep the store exactly DuckDB + blobs, which is what the gate replays.
    raw["trace"]["export_every_episode"] = False
    path = RESULTS_DIR / f"cfg-{name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def write_tasks() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    (TASKS_DIR / "runaway.json").write_text(json.dumps({
        "task_id": "s3-runaway",
        "text": "S3 adversarial task: the root's first cell never returns.",
        "category": "default",
        "context": "S3 runaway fixture corpus. " * 40,
    }, indent=2), encoding="utf-8")

    # llm_query(f'q{i}') composes to just 'q{i}' (compose_leaf_user with
    # chunk=None), so the MockDispatcher fixture keys are computable here.
    fixtures = {
        f"leaf:{hashlib.sha256(f'q{i}'.encode()).hexdigest()}": f"S3-LEAF-{i}"
        for i in range(100)
    }
    (TASKS_DIR / "subcalls.json").write_text(json.dumps({
        "task_id": "s3-subcalls",
        "text": "S3 adversarial task: the root fans out 100 sub-calls at once.",
        "category": "default",
        "context": "S3 subcall fixture corpus. " * 40,
        "fixtures": fixtures,
    }, indent=2), encoding="utf-8")

    (TASKS_DIR / "hardkill.json").write_text(json.dumps({
        "task_id": "s3-hardkill",
        "text": "S3 adversarial task: the scaffold dies mid-episode.",
        "category": "default",
        "context": "S3 hard-kill fixture corpus. " * 200,
    }, indent=2), encoding="utf-8")

    (TASKS_DIR / "recovery.json").write_text(json.dumps({
        "task_id": "s3-recovery",
        "text": "S3 benign task: its startup recovery tombstones the orphan.",
        "category": "default",
        "context": "S3 recovery fixture corpus. " * 10,
        "answer": "recovered",
    }, indent=2), encoding="utf-8")


ROOT_SCRIPTS = {
    "runaway": ["```repl\nwhile True:\n    pass\n```"],
    "subcalls": ["```repl\nimport asyncio\n"
                  "await asyncio.gather(*[llm_query(f'q{i}') for i in range(100)])\n```"],
    "hardkill": ["```repl\nprint(len(chunks))\n```",
                  "```repl\nimport time\ntime.sleep(600)\n```"],
    "recovery": ["```repl\nfinal_answer('recovered')\n```"],
}


def rlm_argv(*args: str) -> list[str]:
    code = "import sys; from rlm.cli import main; sys.exit(main(sys.argv[1:]))"
    return [PYTHON, "-c", code, *args]


def episode_ids_before() -> set[str]:
    if not DB_PATH.exists():
        return set()
    return {r["episode_id"] for r in db_rows("SELECT episode_id FROM episodes")}


def new_episode_row(before: set[str]) -> dict:
    rows = db_rows("SELECT * FROM episodes")
    new = [r for r in rows if r["episode_id"] not in before]
    assert len(new) == 1, f"expected exactly one new episode, got {len(new)}"
    return new[0]


def run_scenario_once(scenario: str, cfg_over: dict, timeout_s: float) -> dict:
    """One `rlm run` subprocess against a fresh scripted root; returns the
    episode row read back from the store plus the subprocess record."""
    server = FakeRootServer(base_raw(), script=list(ROOT_SCRIPTS[scenario]))
    try:
        cfg_path = write_cfg(scenario, server.port, **cfg_over)
        before = episode_ids_before()
        t0 = time.monotonic()
        proc = subprocess.run(
            rlm_argv("run", str(TASKS_DIR / f"{scenario}.json"),
                     "--config", str(cfg_path),
                     "--lifecycle-log", str(LIFECYCLE_LOG)),
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout_s)
        wall = time.monotonic() - t0
        row = new_episode_row(before)
        dur = (row["ended_at"] - row["started_at"]).total_seconds() \
            if row["ended_at"] is not None else None
        return {"episode_id": row["episode_id"], "outcome": row["outcome"],
                "outcome_reason": row["outcome_reason"], "dry_run": row["dry_run"],
                "episode_duration_s": dur, "subprocess_wall_s": round(wall, 2),
                "rc": proc.returncode, "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:]}
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# phase: runaway  (infinite loop in the REPL -> BUDGET_KILL/wall_clock)
# --------------------------------------------------------------------------- #


def phase_runaway() -> None:
    for attempt in range(1, N_ATTEMPTS + 1):
        r = run_scenario_once("runaway", {"max_wall_clock_s": RUNAWAY_WALL_S},
                              timeout_s=RUNAWAY_WALL_S + 120)
        ok = (r["outcome"] == "budget_kill" and r["outcome_reason"] == "wall_clock"
              and r["rc"] == 0 and r["episode_duration_s"] is not None
              and r["episode_duration_s"] <= RUNAWAY_WALL_S + KILL_TEARDOWN_SLACK_S)
        append_run({"check": "runaway", "attempt": attempt, "verdict": "PASS" if ok else "FAIL",
                    "budget_s": RUNAWAY_WALL_S, **r})


# --------------------------------------------------------------------------- #
# phase: subcalls  (100-wide gather at cap 8 -> BUDGET_KILL/max_subcalls)
# --------------------------------------------------------------------------- #


def phase_subcalls() -> None:
    for attempt in range(1, N_ATTEMPTS + 1):
        r = run_scenario_once("subcalls",
                              {"max_subcalls": SUBCALLS_CAP,
                               "max_wall_clock_s": SUBCALLS_WALL_S},
                              timeout_s=SUBCALLS_WALL_S + 120)
        calls = db_rows(
            "SELECT call_id, status FROM steps "
            "WHERE episode_id = ? AND action_type = 'llm_call'", [r["episode_id"]])
        distinct = len({c["call_id"] for c in calls})
        # <= cap alone passes vacuously at zero calls (a broken dispatch path
        # looks exactly like a perfect cap) — dispatched > 0 is load-bearing.
        ok = (r["outcome"] == "budget_kill" and r["outcome_reason"] == "max_subcalls"
              and r["rc"] == 0 and 0 < len(calls)
              and distinct == len(calls) <= SUBCALLS_CAP
              and r["episode_duration_s"] is not None
              and r["episode_duration_s"] <= SUBCALLS_WALL_S)
        append_run({"check": "subcalls", "attempt": attempt,
                    "verdict": "PASS" if ok else "FAIL", "cap": SUBCALLS_CAP,
                    "llm_call_steps": len(calls), "distinct_call_ids": distinct,
                    "statuses": sorted({c["status"] for c in calls}), **r})


# --------------------------------------------------------------------------- #
# phase: hardkill  (TerminateProcess the scaffold mid-step; durability +
#                   KILL_ON_JOB_CLOSE reap + no-egress audit + tombstone)
# --------------------------------------------------------------------------- #


def _ps(cmd: str) -> str:
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


def _child_pids(parent_pid: int, name: str = "python.exe") -> list[int]:
    out = _ps(f"Get-CimInstance Win32_Process -Filter \"ParentProcessId={parent_pid}\" "
              f"| Where-Object {{ $_.Name -eq '{name}' }} "
              f"| Select-Object -ExpandProperty ProcessId")
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _resolve_tree(stub_pid: int) -> tuple[int | None, int | None]:
    """`.venv\\Scripts\\python.exe` is a trampoline: Popen.pid is the stub, the
    REAL scaffold is its python.exe child, and the sandbox interpreter is one
    level deeper (config.yaml's own warning about the stub, verified here by
    cross-checking both pids against the episode row after the kill)."""
    scaffolds = _child_pids(stub_pid)
    scaffold = scaffolds[0] if scaffolds else None
    sandbox = None
    if scaffold is not None:
        kids = _child_pids(scaffold)
        sandbox = kids[0] if kids else None
    return scaffold, sandbox


def _pid_alive(pid: int) -> bool:
    out = _ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue) -ne $null")
    return out.strip().lower() == "true"


def _egress_audit(pid: int) -> dict:
    """§5 C1's rule, applied to the Windows TCP table as it actually renders:
    the sandbox may hold exactly one self-connected 127.0.0.1 pair owned by
    its own pid (asyncio's self-pipe) and zero other endpoints. A control run
    of a bare asyncio loop shows the pair as TWO crossed loopback Established
    rows plus ONE remote-less `0.0.0.0:<pair-port> -> 0.0.0.0:0 Bound` row —
    that Bound shadow is the pair, not an extra endpoint. Anything else
    (a real remote, a Listen state, a port outside the pair) FAILS."""
    out = _ps(
        f"Get-NetTCPConnection -OwningProcess {pid} -ErrorAction SilentlyContinue "
        f"| ForEach-Object {{ \"$($_.LocalAddress):$($_.LocalPort)->"
        f"$($_.RemoteAddress):$($_.RemotePort) $($_.State)\" }}")
    rows = [ln.strip() for ln in out.splitlines() if ln.strip()]
    est = [r for r in rows if r.endswith("Established")]
    bound = [r for r in rows if r.endswith("Bound")]
    pair_ok = False
    pair_ports: set[str] = set()
    if len(est) == 2:
        import re
        parsed = [re.match(r"127\.0\.0\.1:(\d+)->127\.0\.0\.1:(\d+) ", r) for r in est]
        if all(parsed):
            (a_l, a_r), (b_l, b_r) = [m.groups() for m in parsed]
            pair_ok = (a_l == b_r and a_r == b_l)
            pair_ports = {a_l, a_r}
    bound_ok = all(r.startswith("0.0.0.0:") and "->0.0.0.0:0 " in r
                   and r.split(":")[1].split("-")[0] in pair_ports for r in bound)
    extra = [r for r in rows if r not in est and r not in bound]
    ok = pair_ok and bound_ok and not extra
    return {"rows": rows, "pair_ok": pair_ok, "bound_shadow_ok": bound_ok,
            "extra_rows": extra, "verdict": "PASS" if ok else "FAIL"}


def phase_hardkill() -> None:
    server = FakeRootServer(base_raw(), script=list(ROOT_SCRIPTS["hardkill"]))
    episode_id = None
    try:
        cfg_path = write_cfg("hardkill", server.port,
                             max_wall_clock_s=HARDKILL_WALL_S)
        before = episode_ids_before()
        proc = subprocess.Popen(
            rlm_argv("run", str(TASKS_DIR / "hardkill.json"),
                     "--config", str(cfg_path),
                     "--lifecycle-log", str(LIFECYCLE_LOG)),
            cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)

        # Wait for turn 2 (the sleeping cell) to have been DELIVERED — the
        # scripted root sees each /completion, so the parent knows exactly
        # where the episode is without touching the (locked) DB.
        deadline = time.monotonic() + 120
        while server.turns < 2 and time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"scaffold exited early rc={proc.returncode}: "
                    f"{proc.communicate()[1][-2000:]}")
            time.sleep(0.2)
        assert server.turns >= 2, "root script never reached the sleeping cell"
        time.sleep(3.0)  # let the sleep cell reach the sandbox and turn 1 commit

        scaffold_pid, sandbox_pid = _resolve_tree(proc.pid)
        assert scaffold_pid is not None, "no scaffold child under the venv stub"
        audit = _egress_audit(sandbox_pid) if sandbox_pid else \
            {"rows": [], "verdict": "FAIL", "note": "no sandbox child found"}
        append_run({"check": "egress_audit", "sandbox_pid_live": sandbox_pid,
                    "scaffold_pid": scaffold_pid, **audit})

        # TerminateProcess on the REAL scaffold process, mid-step-2.
        _ps(f"Stop-Process -Id {scaffold_pid} -Force")
        proc.wait(timeout=30)
        append_run({"check": "hardkill_kill", "verdict": "PASS",
                    "scaffold_pid": scaffold_pid,
                    "note": "scaffold terminated mid-step-2"})

        # KILL_ON_JOB_CLOSE: the dying scaffold's job handle must reap the
        # sandbox tree — nobody else is left to do it.
        targets = [p for p in (scaffold_pid, sandbox_pid) if p]
        reap_deadline = time.monotonic() + 20
        while time.monotonic() < reap_deadline and \
                any(_pid_alive(p) for p in targets):
            time.sleep(0.5)
        reaped = not any(_pid_alive(p) for p in targets)
        append_run({"check": "hardkill_job_reap", "verdict": "PASS" if reaped else "FAIL",
                    "scaffold_pid": scaffold_pid, "sandbox_pid": sandbox_pid,
                    "reaped": reaped})

        # C6 durability: the store must open after hard death (WAL replay),
        # the episode row must exist with NULL outcome, turn 1 must be
        # committed, and no row may reference a missing blob.
        con = duckdb.connect(str(DB_PATH))  # rw once: WAL replay happens here
        con.close()
        rows = db_rows("SELECT * FROM episodes")
        orphan = [r for r in rows if r["episode_id"] not in before
                  and r["task_id"] == "s3-hardkill"]
        assert len(orphan) == 1, f"expected 1 hardkill episode, got {len(orphan)}"
        episode_id = orphan[0]["episode_id"]
        steps = db_rows("SELECT step_idx, action_type, status, root_view_hash "
                        "FROM steps WHERE episode_id = ? ORDER BY step_idx",
                        [episode_id])
        pid_match = (orphan[0]["sandbox_pid"] == sandbox_pid
                     and orphan[0]["scaffold_instance_id"] == str(scaffold_pid))
        from rlm.trace import sweep_orphan_blobs
        orphan_blobs = sweep_orphan_blobs(DB_PATH, BLOB_ROOT)  # raises on corrupt
        durable = (orphan[0]["outcome"] is None and len(steps) >= 1
                   and steps[0]["status"] == "ok" and bool(pid_match))
        append_run({"check": "hardkill_durability",
                    "verdict": "PASS" if durable else "FAIL",
                    "episode_id": episode_id,
                    "outcome_at_death": orphan[0]["outcome"],
                    "committed_steps": len(steps),
                    "step0_status": steps[0]["status"] if steps else None,
                    "sandbox_pid_matches_audit": bool(pid_match),
                    "orphan_blobs": orphan_blobs,
                    "sweep_orphan_blobs": "clean (no dangling rows)"})

        # Post-restart tombstone: a benign `rlm run` must recover first.
        server2 = FakeRootServer(base_raw(), script=list(ROOT_SCRIPTS["recovery"]))
        try:
            cfg2 = write_cfg("recovery", server2.port)
            rec = subprocess.run(
                rlm_argv("run", str(TASKS_DIR / "recovery.json"),
                         "--config", str(cfg2),
                         "--lifecycle-log", str(LIFECYCLE_LOG)),
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)
        finally:
            server2.shutdown()
        tomb = db_rows("SELECT outcome, outcome_reason, ended_at, scaffold_instance_id "
                       "FROM episodes WHERE episode_id = ?", [episode_id])[0]
        recovery_row = db_rows(
            "SELECT scaffold_instance_id FROM episodes WHERE task_id = 's3-recovery' "
            "ORDER BY started_at DESC LIMIT 1")[0]
        ok = (tomb["outcome"] == "error"
              and tomb["outcome_reason"] == "orphaned_at_recovery"
              and tomb["ended_at"] is not None
              and "recovery: tombstoned 1 orphaned episode(s)" in rec.stdout
              and rec.returncode == 0)
        append_run({"check": "hardkill_tombstone", "verdict": "PASS" if ok else "FAIL",
                    "episode_id": episode_id, "tombstone": {
                        "outcome": tomb["outcome"],
                        "outcome_reason": tomb["outcome_reason"]},
                    "tombstoned_by_different_instance":
                        recovery_row["scaffold_instance_id"]
                        != tomb["scaffold_instance_id"],
                    "recovery_stdout": rec.stdout[-1500:],
                    "recovery_rc": rec.returncode})
        from rlm.trace import sweep_orphan_blobs as sweep2
        sweep2(DB_PATH, BLOB_ROOT)  # still no dangling rows after recovery
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# phase: replay  (lifecycle log DELETED; every episode, killed ones included)
# --------------------------------------------------------------------------- #


def phase_replay() -> None:
    with contextlib.suppress(FileNotFoundError):
        LIFECYCLE_LOG.unlink()
    for extra in STORE_DIR.glob("*.jsonl"):
        extra.unlink()
    assert not list(STORE_DIR.glob("*.jsonl")), "lifecycle log still present"
    cfg_path = RESULTS_DIR / "cfg-runaway.yaml"  # live cfg only supplies paths
    episodes = db_rows("SELECT episode_id, task_id, outcome, outcome_reason "
                       "FROM episodes ORDER BY started_at")
    for ep in episodes:
        proc = subprocess.run(
            rlm_argv("replay", ep["episode_id"], "--config", str(cfg_path)),
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)
        ok = (proc.returncode == 0
              and "root_view_hash: OK" in proc.stdout
              and "message array: OK" in proc.stdout
              and "--- transcript ---" in proc.stdout)
        append_run({"check": "replay", "episode_id": ep["episode_id"],
                    "task_id": ep["task_id"], "episode_outcome": ep["outcome"],
                    "episode_reason": ep["outcome_reason"],
                    "verdict": "PASS" if ok else "FAIL", "rc": proc.returncode,
                    "stdout_head": proc.stdout[:1200],
                    "stderr": proc.stderr[-800:],
                    "lifecycle_log_deleted": True})


# --------------------------------------------------------------------------- #
# phase: report
# --------------------------------------------------------------------------- #


def _latest(runs: list[dict], check: str) -> list[dict]:
    """All records for a check from the LAST contiguous batch (reruns of a
    phase supersede earlier batches wholesale)."""
    recs = [r for r in runs if r.get("check") == check]
    if check in ("runaway", "subcalls"):
        return recs[-N_ATTEMPTS:]
    if check == "replay":
        seen: dict[str, dict] = {}
        for r in recs:
            seen[r["episode_id"]] = r
        return list(seen.values())
    return recs[-1:] if recs else []


def render_report(runs: list[dict]) -> str:
    runaway = _latest(runs, "runaway")
    subcalls = _latest(runs, "subcalls")
    egress = _latest(runs, "egress_audit")
    reap = _latest(runs, "hardkill_job_reap")
    durable = _latest(runs, "hardkill_durability")
    tomb = _latest(runs, "hardkill_tombstone")
    replay = _latest(runs, "replay")

    def all_pass(recs: list[dict], n: int) -> bool:
        return len(recs) == n and all(r.get("verdict") == "PASS" for r in recs)

    suite = _latest(runs, "component_suite")
    det_runaway = len({(r.get("outcome"), r.get("outcome_reason")) for r in runaway}) == 1
    det_subcalls = len({(r.get("outcome"), r.get("outcome_reason")) for r in subcalls}) == 1
    checks = {
        "component suite green (before and after phases)": all_pass(suite, 1),
        f"runaway {N_ATTEMPTS}/{N_ATTEMPTS} (budget_kill, wall_clock), identical":
            all_pass(runaway, N_ATTEMPTS) and det_runaway,
        f"subcalls {N_ATTEMPTS}/{N_ATTEMPTS} (budget_kill, max_subcalls), identical":
            all_pass(subcalls, N_ATTEMPTS) and det_subcalls,
        "hard-kill: job reaps sandbox": all_pass(reap, 1),
        "hard-kill: C6 durability (row+steps survive, no dangling blob rows)":
            all_pass(durable, 1),
        "hard-kill: post-restart tombstone (error, orphaned_at_recovery)":
            all_pass(tomb, 1),
        "no-egress audit (§5 C1)": all_pass(egress, 1),
        "replay: every episode, lifecycle log deleted":
            len(replay) >= 4 and all(r.get("verdict") == "PASS" for r in replay),
    }
    gate = "PASS" if all(checks.values()) else "FAIL"

    lines = [
        "# S3 RESULTS — budgets + tracing gate",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"by `s3/run_s3.py --phase report` from `s3/results/runs.jsonl` and the "
        f"trace store `s3/results/store/`. Numbers come from the store, not "
        f"process memory._",
        "",
        "## Verdict rule (stated before the numbers)",
        "",
        "The gate passes iff ALL of: (1) the component suite is green; "
        f"(2) the runaway-REPL task terminates as (budget_kill, wall_clock) on "
        f"{N_ATTEMPTS}/{N_ATTEMPTS} attempts with identical (outcome, reason) and "
        f"episode duration <= budget + {KILL_TEARDOWN_SLACK_S}s teardown slack; "
        f"(3) the unbounded-sub-call task terminates as (budget_kill, "
        f"max_subcalls) on {N_ATTEMPTS}/{N_ATTEMPTS} attempts with >0 dispatched "
        f"calls and distinct call_ids <= {SUBCALLS_CAP}; (4) after TerminateProcess "
        "on the scaffold mid-step: the sandbox tree is reaped by the Job Object, "
        "the store reopens with the episode row (NULL outcome) and every "
        "committed step intact, `sweep_orphan_blobs` finds no row referencing a "
        "missing blob, and the next `rlm run` tombstones the orphan as (error, "
        "orphaned_at_recovery); (5) the live sandbox pid shows no non-loopback "
        "endpoint; (6) `rlm replay` exits 0 with root_view_hash OK, message "
        "array OK, and a rendered transcript for EVERY episode in the store — "
        "budget-killed and tombstoned episodes included — with the lifecycle "
        "log deleted.",
        "",
        f"## S3 GATE: {gate}",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for name, ok in checks.items():
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", "## Attempts", ""]
    for r in runaway + subcalls:
        lines.append(
            f"- `{r['check']}` #{r['attempt']}: {r['verdict']} — "
            f"({r.get('outcome')}, {r.get('outcome_reason')}), "
            f"episode {r.get('episode_duration_s', '?')} s"
            + (f", {r.get('llm_call_steps')} calls / "
               f"{r.get('distinct_call_ids')} distinct" if r["check"] == "subcalls" else ""))
    for r in durable:
        lines.append(f"- hard-kill durability: {r['verdict']} — "
                     f"{r.get('committed_steps')} step(s) committed, step0 "
                     f"{r.get('step0_status')}, outcome_at_death="
                     f"{r.get('outcome_at_death')}, orphan blobs "
                     f"{len(r.get('orphan_blobs') or [])}, pid match "
                     f"{r.get('sandbox_pid_matches_audit')}")
    for r in tomb:
        lines.append(f"- tombstone: {r['verdict']} — {r.get('tombstone')}, "
                     f"different instance: {r.get('tombstoned_by_different_instance')}")
    for r in egress:
        lines.append(f"- egress audit: {r['verdict']} — {len(r.get('rows') or [])} "
                     f"loopback row(s): {r.get('rows')}")
    lines += ["", "## Replay (lifecycle log deleted)", ""]
    for r in replay:
        lines.append(f"- `{r.get('task_id')}` ({r.get('episode_outcome')}"
                     f"{', ' + r['episode_reason'] if r.get('episode_reason') else ''}): "
                     f"{r['verdict']} (rc={r.get('rc')})")
    lines += ["", NARRATIVE_MARKER, ""]
    return "\n".join(lines)


def regenerate(runs: list[dict]) -> str:
    body = render_report(runs)
    if RESULTS_MD.exists():
        old = RESULTS_MD.read_text(encoding="utf-8")
        if NARRATIVE_MARKER in old:
            body = body[:body.index(NARRATIVE_MARKER)] + \
                old[old.index(NARRATIVE_MARKER):]
    RESULTS_MD.write_text(body, encoding="utf-8", newline="\n")
    return body


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="all",
                    choices=["runaway", "subcalls", "hardkill", "replay",
                             "report", "all"])
    args = ap.parse_args()

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    write_tasks()
    if args.phase in ("runaway", "subcalls", "hardkill", "all"):
        # One-time explicit sandbox install (tests/conftest.bootstrap_dir does
        # the same): the runtime never grants the ACL itself.
        from rlm.config import Config
        from rlm.sandbox.manager import install_bootstrap
        cfg = Config.model_validate(base_raw())
        install_bootstrap(cfg.scaffold.sandbox, grant_acl=True)

    try:
        if args.phase in ("runaway", "all"):
            phase_runaway()
        if args.phase in ("subcalls", "all"):
            phase_subcalls()
        if args.phase in ("hardkill", "all"):
            phase_hardkill()
        if args.phase in ("replay", "all"):
            phase_replay()
    except Exception as exc:  # noqa: BLE001 — a runner exception IS a result
        append_run({"check": "runner_exception", "verdict": "FAIL",
                    "phase": args.phase, "error": repr(exc)})
        raise
    finally:
        if args.phase in ("report", "all", "replay"):
            report = regenerate(read_runs())
            print(report.split(NARRATIVE_MARKER)[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
