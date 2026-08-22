# tests/fixtures/repetition/extract.py
"""One-off: pull the per-turn (cell, observation_view) history of the two
verbatim-repetition loop episodes out of the trace store into JSON fixtures.

The blob tree is git-ignored (ARCHITECTURE.md §6: run output, never source),
so the fixtures are committed and this script records where they came from.
Run from the repo root with the bench idle:

    .venv/Scripts/python.exe tests/fixtures/repetition/extract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rlm.serve.rootclient import extract_cell, strip_reasoning  # noqa: E402

EPISODES = {
    "9d9e47fb-9501-429f-a05c-31df2e01e158": "synth-01",   # 70x, context_exhausted
    "0c1c397d-9501-41b7-82ac-f6e2e8138ebf": "synth-07",   # 111x, budget_kill
}
LANGS, SELECT = ["repl", "python", "py"], "first"


def main() -> None:
    con = duckdb.connect(str(REPO / "traces" / "rlm.duckdb"), read_only=True)
    for episode_id, task_id in EPISODES.items():
        rows = con.execute(
            "SELECT step_idx, action_payload, observation_view FROM steps "
            "WHERE episode_id = ? AND action_type = 'repl_exec' AND status = 'ok' "
            "ORDER BY step_idx", [episode_id]).fetchall()
        turns = []
        for n, (_, payload, view) in enumerate(rows, start=1):
            cell = extract_cell(strip_reasoning(payload or ""), LANGS, SELECT)
            turns.append({"turn": n, "cell": (cell or "").strip(),
                          "observation_view": view or ""})
        out = REPO / "tests" / "fixtures" / "repetition" / f"{episode_id[:8]}.json"
        out.write_text(json.dumps({"episode_id": episode_id, "task_id": task_id,
                                   "turns": turns}, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        print(out, len(turns), "turns")


if __name__ == "__main__":
    main()
