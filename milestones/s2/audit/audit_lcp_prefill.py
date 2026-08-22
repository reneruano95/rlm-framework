"""Q4: does LCP-similarity slot selection operate when cache_prompt is false,
and is the prefilled token count SHORTER than the rendered prompt?

Walks a llama-server stderr log request-by-request, pairing:
    get_availabl (slot selection, with f_sim_best / f_keep)
    launch_slot_ (task id)
    print_timing (prompt eval time = ... / N tokens)
    release     (n_tokens = total)

and prints one row per request. Offline only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"D:/PROJECTS/rlm-halo-framework")

TS = re.compile(r"^(\S+)\s")
SEL = re.compile(r"get_availabl: id\s+(\d+) \| task (-?\d+) \| selected slot by (.*)$")
LAUNCH = re.compile(r"launch_slot_: id\s+(\d+) \| task (\d+) \| processing task")
PE = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens")
EV = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s*([0-9.]+) ms /\s*(\d+) tokens")
REL = re.compile(r"release: id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+)")


def walk(path: Path):
    reqs = []
    cur = None
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ts = TS.match(ln).group(1) if TS.match(ln) else ""
        m = SEL.search(ln)
        if m:
            kind = m.group(3)
            f = re.search(r"f_sim_best\s*=\s*([0-9.]+)", kind)
            k = re.search(r"f_keep\s*=\s*([0-9.]+)", kind)
            cur = dict(ts=ts, slot=int(m.group(1)), how=kind.split(",")[0].strip(),
                       f_sim=float(f.group(1)) if f else None,
                       f_keep=float(k.group(1)) if k else None,
                       task=None, pe_ms=None, pe_tok=None, dec_tok=None, n_tokens=None)
            reqs.append(cur)
            continue
        m = LAUNCH.search(ln)
        if m and cur is not None and cur["task"] is None:
            cur["task"] = int(m.group(2))
            continue
        m = PE.search(ln)
        if m:
            t = int(m.group(2))
            for r in reversed(reqs):
                if r["task"] == t:
                    r["pe_ms"], r["pe_tok"] = float(m.group(3)), int(m.group(4))
                    break
            continue
        m = EV.search(ln)
        if m:
            t = int(m.group(2))
            for r in reversed(reqs):
                if r["task"] == t:
                    r["dec_tok"] = int(m.group(4))
                    break
            continue
        m = REL.search(ln)
        if m:
            t = int(m.group(2))
            for r in reversed(reqs):
                if r["task"] == t:
                    r["n_tokens"] = int(m.group(3))
                    break
    return reqs


def main() -> None:
    for name in sys.argv[1:]:
        p = ROOT / name
        reqs = walk(p)
        print("=" * 100)
        print(f"{p}   requests={len(reqs)}")
        print(f"{'ts':>14} {'slot':>4} {'task':>5} {'how':<18} {'f_sim':>6} {'f_keep':>6} "
              f"{'pe_tok':>7} {'pe_ms':>9} {'dec':>4} {'n_tok':>6} {'implied_reuse':>13}")
        for r in reqs:
            reuse = ""
            if r["pe_tok"] is not None and r["n_tokens"] is not None and r["dec_tok"] is not None:
                prompt_total = r["n_tokens"] - r["dec_tok"]
                reuse = f"{prompt_total - r['pe_tok']:+d}/{prompt_total}"
            print(f"{r['ts']:>14} {r['slot']:>4} {str(r['task']):>5} {r['how'][:18]:<18} "
                  f"{('' if r['f_sim'] is None else f'{r['f_sim']:.3f}'):>6} "
                  f"{('' if r['f_keep'] is None else f'{r['f_keep']:.3f}'):>6} "
                  f"{str(r['pe_tok']):>7} {('' if r['pe_ms'] is None else f'{r['pe_ms']:.0f}'):>9} "
                  f"{str(r['dec_tok']):>4} {str(r['n_tokens']):>6} {reuse:>13}")


if __name__ == "__main__":
    main()
