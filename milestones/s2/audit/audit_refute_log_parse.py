"""Parse llama-server slot logs into an ordered call table (offline)."""
import re, sys, json
from pathlib import Path

PAT_AVAIL = re.compile(r"^(\S+) I slot get_availabl: id\s+(\d+) \| task -1 \| selected slot by (\S+)(.*)$")
PAT_LAUNCH = re.compile(r"^(\S+) I slot launch_slot_: id\s+(\d+) \| task (\d+) \|")
PAT_PROMPT = re.compile(r"^(\S+) I slot print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
PAT_EVAL = re.compile(r"^(\S+) I slot print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
PAT_REL = re.compile(r"^(\S+) I slot\s+release: id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d+)")

def parse(path):
    calls = {}
    order = []
    avail = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = PAT_AVAIL.match(line)
        if m:
            avail.append((m.group(1), int(m.group(2)), m.group(3), m.group(4).strip()))
            continue
        m = PAT_LAUNCH.match(line)
        if m:
            t = int(m.group(3))
            calls[t] = {"t": m.group(1), "slot": int(m.group(2)), "task": t,
                        "sel": avail[-1][2] if avail else None,
                        "sel_detail": avail[-1][3] if avail else None}
            order.append(t)
            continue
        m = PAT_PROMPT.match(line)
        if m:
            calls.setdefault(int(m.group(3)), {})["prompt_n"] = int(m.group(5))
            continue
        m = PAT_EVAL.match(line)
        if m:
            calls.setdefault(int(m.group(3)), {})["pred_n"] = int(m.group(5))
            continue
        m = PAT_REL.match(line)
        if m:
            c = calls.setdefault(int(m.group(3)), {})
            c["n_tokens"] = int(m.group(4)); c["trunc"] = int(m.group(5))
    return [calls[t] for t in order]

for p in sys.argv[1:]:
    rows = parse(p)
    print(f"\n===== {p} : {len(rows)} completions =====")
    for i, c in enumerate(rows):
        print(f"{i:>3} t={c['t']} task={c.get('task'):>4} slot={c.get('slot')} "
              f"prompt_n={c.get('prompt_n')} pred={c.get('pred_n')} "
              f"ntok={c.get('n_tokens')} trunc={c.get('trunc')} sel={c.get('sel')} {c.get('sel_detail','')}")
