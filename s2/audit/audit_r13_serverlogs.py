"""Parse the llama-server stderr logs that accompany the R13 leak-* arms and
the Aug-14 arch_ladder probe.

Extracts, per server process:
  * slot-selection lines  ("selected slot by id", "selected slot by LCP similarity")
  * f_sim_best / f_keep values
  * "prompt eval time = X ms / N tokens" and the matching task id
  * n_slots / n_ctx_slot

Offline only: reads files already on disk.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:/PROJECTS/rlm-halo-framework")

SEL = re.compile(
    r"slot get_availabl: id\s+(\d+) \| task (-?\d+) \| selected slot by (.+)$"
)
LCP = re.compile(r"f_sim_best\s*=\s*([0-9.]+).*?f_keep\s*=\s*([0-9.]+)")
PE = re.compile(
    r"slot print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens"
)
LAUNCH = re.compile(r"slot launch_slot_: id\s+(\d+) \| task (\d+) \| processing task")
INIT = re.compile(r"load_model: initializing, n_slots = (\d+), n_ctx_slot = (\d+)")
REL = re.compile(
    r"slot\s+release: id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+)"
)


def summarise(path: Path, show_lcp: int = 6) -> None:
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    print("=" * 78)
    print(f"{path}  ({len(lines)} lines)")
    for m in INIT.finditer(txt):
        print(f"  init: n_slots={m.group(1)} n_ctx_slot={m.group(2)}")
    sel_kinds = Counter()
    sel_rows = []
    for ln in lines:
        m = SEL.search(ln)
        if m:
            kind = m.group(3)
            k = "LCP" if "similarity" in kind or "LCP" in kind else kind.split("(")[0].strip()
            sel_kinds[k] += 1
            sel_rows.append((int(m.group(1)), int(m.group(2)), kind))
    print("  slot-selection kinds: " + repr(dict(sel_kinds)))
    lcp_lines = [ln for ln in lines if "f_sim_best" in ln or "f_keep" in ln]
    print(f"  lines mentioning f_sim_best/f_keep: {len(lcp_lines)}")
    for ln in lcp_lines[:show_lcp]:
        print("    " + ln.strip()[:200])
    # prompt eval
    pes = [(int(a), int(b), float(c), int(d)) for a, b, c, d in PE.findall(txt)]
    print(f"  prompt-eval records: {len(pes)}")
    if pes:
        tks = Counter(p[3] for p in pes)
        print("    token counts (most common 12): " + repr(tks.most_common(12)))
    rels = [(int(a), int(b), int(c)) for a, b, c in REL.findall(txt)]
    if rels:
        print(f"  release records: {len(rels)}; n_tokens most common: "
              + repr(Counter(r[2] for r in rels).most_common(8)))


if __name__ == "__main__":
    targets = sys.argv[1:] or [
        "traces/logs/leaf-server-ub512nocacheidleslots.log.err",
        "traces/logs/leaf-server-ub512cacheram0ctxcp0.log.err",
        "traces/logs/leaf-server-ub512cacheram0.log.err",
        "traces/logs/leaf-server-ub512ctxcp0.log.err",
        "traces/logs/leaf-server-ub512.log.err",
        "s2/logs/arch-qwen.err",
        "s2/logs/arch-gemma.err",
        "s2/logs/arch-q35.err",
        "s2/logs/bos.err",
    ]
    for t in targets:
        p = ROOT / t
        if p.exists():
            summarise(p)
        else:
            print(f"MISSING {p}")
