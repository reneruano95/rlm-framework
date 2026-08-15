"""Slot routing + prefill accounting straight from the llama-server logs.

Tests the report's leak_evidence 2 and 3, and the stronger question:
did cache_prompt:false actually force a FULL re-prefill, or is prompt_eval
short of the rendered prompt (partial KV reuse = the leakage channel)?
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\s2")

LAUNCH = re.compile(r"slot launch_slot_: id\s+(\d+) \| task (\d+)")
SEL = re.compile(r"slot get_availabl: id\s+(\d+) \| task -1 \| selected slot by ([^,]+)")
PEVAL = re.compile(r"slot print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens")

LOGS = {
    "arch-qwen.err   (v0.3.3 hybrid arm)": ("arch_ladder_qwen-hybrid.jsonl",),
    "arch-gemma.err  (v0.3.3 control arm)": ("arch_ladder_gemma-fullattn.jsonl",),
    "slotfix-qwen.err (virgin+shared rerun)": ("arch_ladder_qwen_virgin.jsonl",
                                              "arch_ladder_qwen_shared.jsonl"),
    "origgeom.err": (),
}


def main():
    for name, results in LOGS.items():
        f = S2 / "logs" / name.split()[0]
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        launches = LAUNCH.findall(t)
        sels = SEL.findall(t)
        pevals = PEVAL.findall(t)
        print("=" * 74)
        print(f"{name}")
        init = [l for l in t.split("\n") if "n_slots" in l]
        print(f"  init: {init[0].strip()[-80:] if init else '??'}")
        print(f"  launches: {len(launches)}  per-slot: "
              f"{dict(collections.Counter(s for s, _ in launches))}")
        print(f"  slot-selection reasons: "
              f"{dict(collections.Counter(r.strip() for _, r in sels))}")

        # prompt-eval token counts seen
        pe = collections.Counter(int(n) for _, _, n in pevals)
        big = sorted(pe)[-6:]
        print(f"  distinct prompt_eval token counts: {len(pe)}  largest: {big}")

        # compare against the rendered prompt lengths the client recorded
        rt = set()
        for rf in results:
            p = S2 / "results" / rf
            if p.exists():
                for line in p.open(encoding="utf-8"):
                    if line.strip():
                        r = json.loads(line)
                        if r.get("rendered_tokens"):
                            rt.add(r["rendered_tokens"])
        if rt:
            # a completion call's prompt_eval should equal rendered_tokens (full
            # re-prefill). Count prompt-evals that match NO recorded prompt len.
            comp = [int(n) for _, _, n in pevals if int(n) > 200]
            # tolerance +-3 tokens (BOS / template drift)
            def near(v):
                return any(abs(v - x) <= 3 for x in rt)
            short = [v for v in comp if not near(v)]
            print(f"  completion-sized prompt_evals: {len(comp)}; "
                  f"matching a recorded rendered_tokens (+-3): {len(comp)-len(short)}")
            print(f"  NOT matching (possible partial reuse): {len(short)} "
                  f"{sorted(set(short))[:10]}")
            print(f"  recorded rendered_tokens: {sorted(rt)}")


if __name__ == "__main__":
    main()
