#!/bin/bash
# profile_ep.sh <sessions-dir> <cwd-suffix>  -- turns, tool calls, identical-streak, stops
PY=/home/spike/prime-spike/kernel-venv/bin/python
"$PY" - "$1" "$2" <<'PY'
import json, glob, os, sys, collections
d, suf = sys.argv[1], sys.argv[2]
tgt = None
for f in sorted(glob.glob(os.path.join(d, '*.jsonl')), key=os.path.getmtime):
    try:
        h = json.loads(open(f, encoding='utf-8').readline())
    except Exception:
        continue
    if str(h.get('cwd', '')).endswith(suf):
        tgt = f
if not tgt:
    print('no session for', suf); raise SystemExit(0)
codes, turns, tin, tout = [], 0, 0, 0
stops = collections.Counter()
for line in open(tgt, encoding='utf-8'):
    o = json.loads(line); m = o.get('message') or {}
    if m.get('role') == 'assistant':
        turns += 1
        stops[m.get('stopReason')] += 1
        u = m.get('usage') or {}
        tin += u.get('input') or 0; tout += u.get('output') or 0
        for c in m.get('content') or []:
            if c.get('type') == 'toolCall':
                codes.append((c.get('arguments') or {}).get('code', ''))
best = cur = 1 if codes else 0
for a, b in zip(codes, codes[1:]):
    cur = cur + 1 if a == b else 1
    best = max(best, cur)
print(f'session={os.path.basename(tgt)}')
print(f'turns={turns} tool_calls={len(codes)} max_identical_streak={best} in={tin} out={tout} stops={dict(stops)}')
if codes:
    print('--- last tool call:'); print(codes[-1][:700])
PY
