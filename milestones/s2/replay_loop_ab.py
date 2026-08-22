"""Replay A/B for the verbatim-repetition loops seen under the DFlash2 root.

Question (milestones/s4/RESULTS-dflash2-rlm-only.md, Findings): two `rlm` episodes in the
DFlash2 re-validation re-emitted one byte-identical REPL cell 70x and 111x
after an empty/unchanged observation. Zero such loops exist in S4's 90 MTP-root
episodes. Is the speculative drafter the cause, or is it sampling variance?

Design: take the EXACT rendered root request at three points of each loop --
`onset` (the turn that produced the first instance of the cell), `repeat1`
(the first repeat), `established` (several repeats in) -- and replay each one
N times, seeds 1..N, with the bench's own sampling (temperature 0.7, top-p 0.8,
n_predict 1024, cache_prompt on), against the root launched under each arm:

  dflash4       production: PR-#27342 build, -md drafter, draft-dflash n-max 4
  mtp2          same build, draft-mtp n-max 2 (S4's speculation on today's build)
  base          same build, no speculation
  mtp2-b10375   S4's actual binary (tools/llamacpp-vulkan, b10375) + draft-mtp 2
  dflash4-r     production again, last, as an order/thermal replicate

Metric per completion: does the reply's extracted cell equal the loop cell
(exact, and whitespace-normalised)? Secondary: any prose outside the cell,
`final_answer` present, tokens_out, stop_type, truncated.

Everything the server launch needs is copied verbatim from config.yaml
servers.root (via milestones/s2/dflash2_bench.py's BASE_ARGV) so the arms differ only in
the binary and the speculative flags. Results: milestones/s2/results/replay-loop-ab/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(r"D:\PROJECTS\rlm-halo-framework")
sys.path.insert(0, str(PROJECT))
from rlm.rootclient import extract_cell, strip_reasoning  # noqa: E402  (the bench's own parser)

EXE_D2 = PROJECT / "tools" / "llamacpp-vulkan-dflash2" / "llama-server.exe"
EXE_B10375 = PROJECT / "tools" / "llamacpp-vulkan" / "llama-server.exe"
ROOT_GGUF = Path(r"D:\AI\models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf")
DRAFT_GGUF = Path(r"D:\AI\models\z-lab\Qwen3.8-27B-DFlash2-GGUF\Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
OUT = PROJECT / "milestones" / "s2" / "results" / "replay-loop-ab"
STIM = OUT / "stimuli"
PORT = 8080

# config.yaml servers.root, verbatim minus the spec flags (each arm sets them).
def base_argv(exe: Path) -> list[str]:
    return [
        str(exe), "-m", str(ROOT_GGUF),
        "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "32768", "-np", "1",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "-fa", "on", "-ub", "512", "-b", "2048",
        "-lv", "4",
        "-lm", "none", "--no-context-shift",
    ]

ARMS: dict[str, tuple[Path, list[str]]] = {
    "dflash4": (EXE_D2, ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "4"]),
    "mtp2": (EXE_D2, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
    "base": (EXE_D2, []),
    "mtp2-b10375": (EXE_B10375, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
    "dflash4-r": (EXE_D2, ["-md", str(DRAFT_GGUF), "--spec-type", "draft-dflash", "--spec-draft-n-max", "4"]),
}

# The bench's root sampling and cell parser settings (config.yaml scaffold.*).
TEMPERATURE = 0.7
TOP_P = 0.8
N_PREDICT = 1024
LANGS = ["repl", "python", "py"]
SELECT = "first"


def http_json(method: str, path: str, body: dict | None = None, timeout: float = 900.0) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


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


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def load_stimuli() -> list[dict]:
    manifest = json.loads((STIM / "manifest.json").read_text(encoding="utf-8"))
    for st in manifest:
        with open(STIM / f"{st['name']}.rendered.txt", encoding="utf-8", newline="") as f:
            st["rendered"] = f.read()
        with open(STIM / f"{st['episode']}.loopcell.txt", encoding="utf-8", newline="") as f:
            st["loop_cell"] = f.read()
        assert hashlib.sha256(st["rendered"].encode("utf-8")).hexdigest() == st["sha256"], st["name"]
    return manifest


def one_completion(rendered: str, seed: int) -> dict:
    t0 = time.time()
    resp = http_json("POST", "/completion", {
        "prompt": rendered, "n_predict": N_PREDICT, "temperature": TEMPERATURE,
        "top_p": TOP_P, "seed": seed, "cache_prompt": True, "stream": False,
        "return_tokens": False,
    })
    wall = time.time() - t0
    raw = resp.get("content", "")
    stripped = strip_reasoning(raw)
    cell = extract_cell(stripped, LANGS, SELECT)
    prose = re.sub(r"```(?:repl|python|py)\n.*?```", "", stripped, flags=re.S).strip()
    timings = resp.get("timings", {}) or {}
    return {
        "raw": raw, "cell": cell, "prose": prose,
        "tokens_out": timings.get("predicted_n"), "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"), "predicted_ms": timings.get("predicted_ms"),
        "prompt_ms": timings.get("prompt_ms"), "stop_type": resp.get("stop_type"),
        "truncated": bool(resp.get("truncated")), "wall_s": round(wall, 2),
    }


def run_arm(arm: str, stimuli: list[dict], n: int, log) -> list[dict]:
    exe, spec = ARMS[arm]
    argv = base_argv(exe) + spec
    srv_log = OUT / f"{arm}.server.log"
    rows: list[dict] = []
    print(f"ARM {arm} launching: {exe.parent.name} {' '.join(spec) or '(no speculation)'}", file=log, flush=True)
    with srv_log.open("wb") as lf:
        proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, cwd=str(exe.parent))
        try:
            if not wait_health(300):
                print(f"ARM {arm} FAILED: health timeout", file=log, flush=True)
                return rows
            props = http_json("GET", "/props")
            build = props.get("build_info")
            print(f"ARM {arm} up: build {build}", file=log, flush=True)
            out_path = OUT / f"{arm}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for st in stimuli:
                    hits = hits_norm = prose_n = final_n = 0
                    toks = []
                    for seed in range(1, n + 1):
                        try:
                            r = one_completion(st["rendered"], seed)
                        except Exception as e:  # noqa: BLE001 -- record and continue
                            r = {"error": repr(e)}
                        r.update(arm=arm, stimulus=st["name"], episode=st["episode"], turn=st["turn"],
                                 seed=seed, build=build, spec=" ".join(spec))
                        if "cell" in r:
                            r["repeat_exact"] = (r["cell"] or "").strip() == st["loop_cell"].strip()
                            r["repeat_norm"] = norm(r["cell"]) == norm(st["loop_cell"])
                            r["has_prose"] = bool(r["prose"])
                            r["has_final"] = "final_answer" in (r["cell"] or "")
                            hits += r["repeat_exact"]; hits_norm += r["repeat_norm"]
                            prose_n += r["has_prose"]; final_n += r["has_final"]
                            if r["tokens_out"] is not None:
                                toks.append(r["tokens_out"])
                        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                        rows.append(r)
                    med = statistics.median(toks) if toks else None
                    print(f"RESULT {arm} {st['name']}: repeat {hits}/{n} (norm {hits_norm}) "
                          f"prose {prose_n}/{n} final_answer {final_n}/{n} median_tokens {med}",
                          file=log, flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(5)
    return rows


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2x2 table [[a,b],[c,d]]: exact two-sided p via hypergeometric tail summing."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def pmf(x: int) -> float:
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p_obs = pmf(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-12)))


def summarise(arms: list[str], stimuli: list[dict], n: int, log) -> None:
    table: dict[tuple[str, str], dict] = {}
    for arm in arms:
        p = OUT / f"{arm}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "cell" not in r:
                continue
            t = table.setdefault((arm, r["stimulus"]), {"n": 0, "rep": 0, "prose": 0, "final": 0, "toks": []})
            t["n"] += 1; t["rep"] += r["repeat_exact"]; t["prose"] += r["has_prose"]; t["final"] += r["has_final"]
            if r["tokens_out"] is not None:
                t["toks"].append(r["tokens_out"])
    print("\n== SUMMARY: repeat_exact / n  (prose, final_answer, median tokens) ==", file=log)
    names = [s["name"] for s in stimuli]
    print("arm".ljust(14) + "".join(nm.ljust(26) for nm in names) + "pooled", file=log)
    pooled: dict[str, tuple[int, int]] = {}
    for arm in arms:
        cells = []; R = N = 0
        for nm in names:
            t = table.get((arm, nm))
            if not t:
                cells.append("-".ljust(26)); continue
            R += t["rep"]; N += t["n"]
            cells.append(f"{t['rep']}/{t['n']} (p{t['prose']} f{t['final']} t{int(statistics.median(t['toks'])) if t['toks'] else '-'})".ljust(26))
        pooled[arm] = (R, N)
        print(arm.ljust(14) + "".join(cells) + (f"{R}/{N}" if N else "-"), file=log)
    if "dflash4" in pooled and "mtp2" in pooled:
        a, n1 = pooled["dflash4"]; c, n2 = pooled["mtp2"]
        print(f"\nFisher exact, pooled dflash4 vs mtp2: {a}/{n1} vs {c}/{n2}  p={fisher_exact_two_sided(a, n1 - a, c, n2 - c):.4g}", file=log)
        for nm in names:
            t1, t2 = table.get(("dflash4", nm)), table.get(("mtp2", nm))
            if t1 and t2:
                print(f"  {nm}: {t1['rep']}/{t1['n']} vs {t2['rep']}/{t2['n']}  p={fisher_exact_two_sided(t1['rep'], t1['n'] - t1['rep'], t2['rep'], t2['n'] - t2['rep']):.4g}", file=log)
    if "dflash4" in pooled and "base" in pooled:
        a, n1 = pooled["dflash4"]; c, n2 = pooled["base"]
        print(f"Fisher exact, pooled dflash4 vs base: {a}/{n1} vs {c}/{n2}  p={fisher_exact_two_sided(a, n1 - a, c, n2 - c):.4g}", file=log)
    if "dflash4" in pooled and "dflash4-r" in pooled:
        a, n1 = pooled["dflash4"]; c, n2 = pooled["dflash4-r"]
        print(f"Replicate check, dflash4 vs dflash4-r: {a}/{n1} vs {c}/{n2}  p={fisher_exact_two_sided(a, n1 - a, c, n2 - c):.4g}", file=log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="dflash4,mtp2,base,mtp2-b10375,dflash4-r")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    OUT.mkdir(parents=True, exist_ok=True)
    stimuli = load_stimuli()
    log = sys.stdout
    print(f"stimuli: {[s['name'] for s in stimuli]}  n={a.n}  arms={arms}", file=log, flush=True)
    if not a.summary_only:
        for arm in arms:
            run_arm(arm, stimuli, a.n, log)
    summarise(arms, stimuli, a.n, log)
    print("DONE", file=log, flush=True)


if __name__ == "__main__":
    main()
