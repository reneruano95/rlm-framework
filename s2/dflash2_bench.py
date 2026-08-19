"""Drive s2/mtp_bench.py across DFlash2 / MTP / no-speculation arms.

Every arm launches ONE server with the production root's pinned flags
(config.yaml servers.root, reproduced via rlm.serverproc.launch_argv) and
differs ONLY in the speculative-decoding flags. The harness itself
(s2/mtp_bench.py) is NOT modified -- acceptance is recovered from the
server's own -lv 4 log, which prints

    draft acceptance = 0.61530 (  123 accepted /   200 generated), mean len =  4.80

A NEW `base` arm is re-measured on THIS build. The Aug-18 numbers in
s2/results/ came from the pinned b10375 clang release binary; this is an
MSVC build of an unmerged PR, so a build-to-build delta would confound the
comparison if the old base were reused as the reference.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(r"D:\PROJECTS\rlm-halo-framework")
# The SHIPPED backend_dir (config.yaml servers.root), not the source tree it was
# built from: re-running these arms must measure what production launches, and a
# rebuild under D:\AI\src must not silently change that.
EXE = PROJECT / "tools" / "llamacpp-vulkan-dflash2" / "llama-server.exe"
ROOT_GGUF = Path(r"D:\AI\models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf")
DRAFT_GGUF = Path(r"D:\AI\models\z-lab\Qwen3.8-27B-DFlash2-GGUF\Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
LOGDIR = PROJECT / "s2" / "results" / "dflash2-logs"
PORT = 8080

# config.yaml servers.root, verbatim minus the spec flags (which each arm sets).
BASE_ARGV = [
    str(EXE), "-m", str(ROOT_GGUF),
    "--host", "127.0.0.1", "--port", str(PORT),
    "-c", "32768", "-np", "1",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "-fa", "on", "-ub", "512", "-b", "2048",
    "-lv", "4",
    "-lm", "none", "--no-context-shift",
]

ARMS = [
    ("d2-base", []),
    ("d2-mtp2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
    ("d2-n2", ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "2"]),
    ("d2-n4", ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "4"]),
    ("d2-n7", ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "7"]),
    # Confirmation pass: the n-max curve peaked sharply at 4 (26.80 / 35.56 /
    # 24.40 for n=2/4/7) on 8 runs per arm. These arms re-measure the peak and
    # its two neighbours independently, at higher reps, with the MTP incumbent
    # re-run alongside so the comparison stays same-session.
    ("d2-n3", ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "3"]),
    ("d2-n5", ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "5"]),
    ("d2-n6", ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "6"]),
    ("d2-n4-r", ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "4"]),
    ("d2-mtp2-r", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
]

ACC = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), "
    r"mean len =\s*([0-9.]+)")


def wait_health(timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(2.0)
    return False


def run_arm(label: str, spec: list[str], reps: int, timeout_s: float,
            temp: float = 0.0) -> dict:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / f"{label}.log"
    argv = BASE_ARGV + spec
    print(f"\n{'='*72}\nARM {label}\n  {' '.join(spec) or '(no speculation)'}\n{'='*72}")
    with log.open("wb") as lf:
        proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(EXE.parent))
        try:
            if not wait_health(timeout_s):
                proc.terminate()
                tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
                return {"label": label, "ok": False, "error": "health timeout",
                        "log_tail": tail}
            print(f"  server up; running s2/mtp_bench.py --label {label}")
            r = subprocess.run(
                ["uv", "run", "--python", "3.12", "--no-project",
                 "s2/mtp_bench.py", "--label", label, "--reps", str(reps),
                 "--temp", str(temp)],
                cwd=str(PROJECT), capture_output=True, text=True, timeout=3600)
            print(r.stdout[-2500:])
            if r.returncode != 0:
                return {"label": label, "ok": False, "error": "bench failed",
                        "stderr": r.stderr[-2000:]}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    text = log.read_text(encoding="utf-8", errors="replace")
    accs = [(float(a), int(b), int(c), float(d)) for a, b, c, d in ACC.findall(text)]
    out = {"label": label, "ok": True, "spec": " ".join(spec), "log": str(log),
           "n_acc_records": len(accs)}
    if accs:
        out["mean_acc_len"] = sum(x[3] for x in accs) / len(accs)
        out["accept_ratio"] = sum(x[1] for x in accs) / max(1, sum(x[2] for x in accs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    arms = [x for x in ARMS if not a.only or x[0] in a.only]
    results = []
    for label, spec in arms:
        results.append(run_arm(label, spec, a.reps, a.timeout, a.temp))
        print(json.dumps(results[-1], indent=2)[:1200])

    out = LOGDIR / "arms_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
