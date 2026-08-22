# Repetition Guard Bundle (spec v0.3.16) Implementation Plan

> **STATUS (2026-08-22): LANDED as spec v0.3.16** (`f19fdca`).
> Its unchecked `- [ ]` boxes are **not** a to-do list: this project never ticked
> plan checkboxes as work landed. Ground truth for what shipped is `ARCHITECTURE.md`
> §9's gate status lines and `CHANGELOG.md`.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a verbatim-repetition loop cost the root ~30 s instead of 1,154–1,308 s, remove the two scaffold-side contributors to it (same seed on every turn; a doubled empty think block on every past assistant turn), and record all three as a spec amendment with a version bump.

**Architecture:** Three independent, small changes behind config keys, each recorded in `config_snapshot` so `rlm replay` and the verdict stay exact. (1) A new C5 budget `max_identical_turns` in `BudgetEnforcer`: the root's `(cell, observation)` pair repeating consecutively is counted; at `max−1` the scaffold appends a correction to the observation, at `max` the episode ends `budget_kill / max_identical_turns` — the existing outcome and the existing §6 reason convention, so no schema, verdict, or bench change. (2) `scaffold.root.seed_schedule: per_turn` derives the root's sampling seed from the episode seed and the turn index in `RootConversation.turn()`. (3) `scaffold.root.history_mode: raw` stores the assistant history message as the model's raw completion (reasoning, if any, split into `reasoning_content`) instead of `assistant_prefix(rendered) + raw`; replay reads the mode from the episode's snapshot, so pre-amendment episodes still reconstruct under the old rule.

**Tech Stack:** Python 3.12, pydantic config (`rlm/config.py`), asyncio episode runner (`rlm/episode.py`), `BudgetEnforcer` (`rlm/budget.py`), `RootConversation` (`rlm/rootclient.py`), `rlm replay` (`rlm/cli.py`), pytest + the `episode_env` / `fake_root_server` / `mock_episode_env` fixtures in `tests/conftest.py`. Run tests with `.venv/Scripts/python.exe -m pytest`.

**Spec:** `milestones/s2/REPLAY-LOOP-AB.md` §4 (the three changes and why), `milestones/s4/RESULTS-dflash2-rlm-only.md` § Findings items 1–2 (the two loop episodes `9d9e47fb`, `0c1c397d`), ARCHITECTURE.md §5 C5 (budgets), §6 (outcome semantics, state rule), §14 (changelog); D26 (append-only conversation) in `docs/superpowers/plans/2026-08-13-capa1-scaffold.md:675`. The chat template that decides the history-mode design is Qwen3.8-27B's (extracted to `tests/fixtures/repetition/qwen38_chat_template.jinja` in Task 1):

```jinja
{%- elif message.role == "assistant" %}
    {%- set reasoning_content = '' %}
    {%- if message.reasoning_content is string %}
        {%- set reasoning_content = message.reasoning_content %}
    {%- endif %}
    {%- set reasoning_content = reasoning_content|trim %}
    {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
    ...
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
```

Every past assistant turn is rendered with the template's own `<think>\n…\n</think>\n\n` block; `content` is emitted verbatim after it. Today `rootclient.py:172` stores `content = assistant_prefix(rendered) + raw` where `assistant_prefix` is the generation prompt's `<think>\n\n</think>\n\n` — hence two blocks per turn in every stored render (`milestones/s2/results/replay-loop-ab/stimuli/9d9e47fb-onset.rendered.txt`: 17 `<think>` for 9 assistant markers).

## Global Constraints

- **I1 — the scaffold disposes.** All three mechanisms are config + scaffold code; no model output may alter a threshold, a seed, or the reconstruction rule.
- **I4 / state rule (§6).** `rlm replay` must re-derive every root message array from the trace alone and rehash `root_view_hash` OK for (a) every episode already in `traces/rlm.duckdb` (old rule) and (b) new episodes (new rule). Replay reads the rule from the episode's `config_snapshot`, never from the live config.
- **§6 outcome vocabulary is unchanged.** The guard terminates as `budget_kill` with `outcome_reason = "max_identical_turns"`, following the existing convention (`budget_kill / wall_clock`, `budget_kill / max_subcalls`). No new `Outcome` member, no `schema.sql` change, no `rlm/verdict.py` or `rlm/bench.py` change.
- **Amendment rule (ARCHITECTURE.md header):** invariants and gates change only with a version bump and a dated changelog entry. This bundle bumps `rlm-runtime-spec-v0.3.15` → `v0.3.16` and touches §5 C5, §6, §10, §14. No invariant or gate changes. (D26, "append-only root conversation", is recorded in `docs/superpowers/plans/2026-08-13-capa1-scaffold.md:675` and in `rlm/rootclient.py`'s docstrings — **not** in ARCHITECTURE.md; the history-mode rule is written into §6.)
- **Scope of the seed schedule:** it applies to `RootConversation.turn()` — the RLM arm's root loop. One-shot root completions (B2's reduce call, `rlm/arms.py:1307`) keep passing the base seed; they have no turns. `rlm/arms.py:477` builds `BudgetLimits(...)` without `max_identical_turns`, so baseline arms keep it at 0 (disabled) by construction — they have no root turn loop either.
- **Prompt registry untouched.** The correction text is scaffold-generated like `no_cell_observation`, not a registry file; `prompts/*` and their sha256 pins do not change.
- **Defaults are backward-compatible.** `Budgets.max_identical_turns` defaults to `0` (disabled) in the dataclass so existing positional constructors in `tests/test_budget.py` keep working; `history_mode` defaults to the OLD rule (`prefix_plus_raw`) and `seed_schedule` to `fixed` in the pydantic models so every snapshot already in the store validates to the behaviour it ran with. `config.yaml` opts in to the new values.
- **Commits:** one per task, scoped `git add` of the task's files only (the working tree holds uncommitted artifacts from 2026-08-20/21 that are not part of this bundle). End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Platform:** Windows 11, Git Bash; `episode_env` tests are `win32`-only (they spawn the real sandbox). Test command: `.venv/Scripts/python.exe -m pytest tests/<file>.py -q`.

---

## File map

| file | change | responsibility |
|---|---|---|
| `tests/fixtures/repetition/9d9e47fb.json`, `0c1c397d.json` (new) | per-turn `cell` + `observation_view` of the two recorded loops | ground truth the guard is tested against |
| `tests/fixtures/repetition/extract.py` (new) | one-off extractor from `traces/blobs/` | provenance of the fixtures (blobs are git-ignored) |
| `tests/fixtures/repetition/qwen38_chat_template.jinja` (new) | the root model's chat template | provenance of the history-mode design |
| `rlm/budget.py` | `Budgets.max_identical_turns`, `BudgetEnforcer.note_turn()` | the counter and the breach |
| `rlm/config.py` | `Budgets.max_identical_turns`, `RootScaffoldCfg.seed_schedule`, `RootScaffoldCfg.history_mode` + validators | config surface |
| `config.yaml` | the three keys, with the file's comment style | shipped values |
| `rlm/episode.py` | `repetition_observation()`, guard call in `_turn_loop`, history-divergence lifecycle event | loop integration |
| `rlm/rootclient.py` | `split_reasoning()`, `history_message()`, per-turn seed, `RootTurn.prefix_extended` | reconstruction rule + seed schedule |
| `rlm/cli.py` | replay uses the snapshot's `history_mode` | offline state-rule check stays exact |
| `rlm/lifecycle.py` | `"root_history"` added to `ALLOWED_KINDS` | the divergence monitor is a lifecycle event; the allowlist refuses unknown kinds with `ValueError` |
| `tests/conftest.py` | `_render_chatml` faithful to the real template | the fake must double the block the way the real server does |
| `tests/test_budget.py`, `tests/test_episode.py`, `tests/test_rootclient.py`, `tests/test_cli.py`, `tests/test_config.py` | new tests | |
| `ARCHITECTURE.md` | v0.3.16 amendment | the record |

---

### Task 1: Fixtures — the two recorded loops and the chat template

**Files:**
- Create: `tests/fixtures/repetition/extract.py`
- Create: `tests/fixtures/repetition/9d9e47fb.json`, `tests/fixtures/repetition/0c1c397d.json`
- Create: `tests/fixtures/repetition/qwen38_chat_template.jinja`

**Interfaces:**
- Produces: JSON files of shape `{"episode_id": str, "task_id": str, "turns": [{"turn": int, "cell": str, "observation_view": str}]}` where `observation_view` is the pre-trailer view string exactly as stored in `steps.observation_view` (C3 output; for these episodes it never contained a scaffold note). Task 2's tests load them with `json.loads(Path(...).read_text(encoding="utf-8"))`.

- [ ] **Step 1: Write the extractor**

```python
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
from rlm.rootclient import extract_cell, strip_reasoning  # noqa: E402

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
```

- [ ] **Step 2: Run it (servers and bench must be idle — the store is single-writer)**

Run: `.venv/Scripts/python.exe tests/fixtures/repetition/extract.py`
Expected: two lines, `9d9e47fb.json 79 turns` and `0c1c397d.json 115 turns` (`9d9e47fb` ended `context_exhausted` after its 79th turn, so every turn has an `ok` row; `0c1c397d`'s 116th turn was the one the wall clock killed and is `cancelled`, so 115). If the store is locked ("being used by another process"), stop the process holding it first.

- [ ] **Step 3: Sanity-check the fixtures reproduce the loop shape**

Run:
```bash
.venv/Scripts/python.exe - <<'EOF'
import json, hashlib
for ep, first, run in [("9d9e47fb", 9, 70), ("0c1c397d", 5, 111)]:
    t = json.load(open(f"tests/fixtures/repetition/{ep}.json", encoding="utf-8"))["turns"]
    h = [hashlib.sha1(x["cell"].encode()).hexdigest()[:8] for x in t]
    loop = h[first - 1]
    earlier_repeat = any(h[i] == h[i-1] and t[i]["observation_view"] == t[i-1]["observation_view"]
                         for i in range(1, first - 1))
    print(ep, "loop cell first at turn", h.index(loop) + 1, "| occurrences:", h.count(loop),
          "| views identical across repeats:", len({x["observation_view"] for x in t[first-1:]}) == 1,
          "| identical pair before onset:", earlier_repeat)
EOF
```
Expected: `9d9e47fb loop cell first at turn 9 | occurrences: 71 | views identical across repeats: True | identical pair before onset: False` and `0c1c397d … turn 5 … 111 … True … False`. ("70×" in the write-up counts the repeats *after* the first instance, and `9d9e47fb` holds all 71; `0c1c397d`'s 112th occurrence was the turn the wall clock killed, stored `cancelled`, so the fixture holds 111. `identical pair before onset: False` is what lets Task 2's fixture test assert the correction lands exactly at onset + 1.)

- [ ] **Step 4: Save the chat template**

Run:
```bash
.venv/Scripts/python.exe - <<'EOF'
import struct
p = r'D:\AI\models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf'
head = open(p, 'rb').read(64 * 1024 * 1024)
i = head.find(b'tokenizer.chat_template'); pos = i + len(b'tokenizer.chat_template')
typ = struct.unpack_from('<I', head, pos)[0]; assert typ == 8; pos += 4
n = struct.unpack_from('<Q', head, pos)[0]; pos += 8
open('tests/fixtures/repetition/qwen38_chat_template.jinja', 'w', encoding='utf-8', newline='').write(head[pos:pos+n].decode('utf-8'))
print('chars', n)
EOF
```
Expected: `chars 8952`. Confirm with `grep -n "reasoning_content|trim" tests/fixtures/repetition/qwen38_chat_template.jinja` → one match.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/repetition/
git commit -m "fixtures: the two verbatim-repetition loop histories and the root chat template

Per-turn (cell, observation_view) of episodes 9d9e47fb (70x) and 0c1c397d
(111x) from the DFlash2 re-validation, plus Qwen3.8-27B's chat template,
which renders every past assistant turn with its own think block.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `max_identical_turns` in `BudgetEnforcer`

**Files:**
- Modify: `rlm/budget.py:50-57` (Budgets dataclass), `rlm/budget.py:81-135` (enforcer init / `_breach`), add `note_turn` next to `note_root_usage` (`rlm/budget.py:227-235`)
- Modify: `rlm/config.py:155-179` (pydantic `Budgets`)
- Modify: `config.yaml` budgets block (after `max_total_tokens: 1500000`, line ~388)
- Test: `tests/test_budget.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `Budgets.max_identical_turns: int = 0` (dataclass, last field) and `BudgetEnforcer.note_turn(cell: str, view: str) -> bool`. Returns `True` exactly when a correction is due (the pair has now repeated `max_identical_turns − 1` times consecutively); raises `BudgetBreach(Outcome.BUDGET_KILL, "max_identical_turns")` when it has repeated `max_identical_turns` times. `0` disables (always `False`, never raises). Task 3's episode loop calls it once per executed cell with the **un-annotated** view.
- Produces: `rlm.config.Budgets.max_identical_turns: int = 3` with validator `== 0 or >= 2`.

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/test_budget.py`:

```python
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "repetition"


def _guarded(max_identical_turns: int = 3) -> BudgetEnforcer:
    return BudgetEnforcer(Budgets(1, 32, 900, 1_000_000, {"root": 1024, "leaf": 512},
                                  max_identical_turns=max_identical_turns))


def test_identical_turns_correct_at_max_minus_one_and_kill_at_max():
    """Spec §5 C5 (v0.3.16): the SAME (cell, observation) pair repeating is a
    budget. At max-1 consecutive occurrences the scaffold corrects; at max it
    kills as budget_kill/max_identical_turns -- the existing outcome, the
    existing reason convention."""
    b = _guarded(3)
    assert b.note_turn("print(1)", "[stdout]\n1") is False      # first occurrence
    assert b.note_turn("print(1)", "[stdout]\n1") is True       # 2nd == max-1: correct
    with pytest.raises(BudgetBreach) as exc:
        b.note_turn("print(1)", "[stdout]\n1")                  # 3rd == max: kill
    assert exc.value.outcome == Outcome.BUDGET_KILL
    assert exc.value.reason == "max_identical_turns"


def test_a_different_cell_or_a_different_observation_resets_the_count():
    b = _guarded(3)
    b.note_turn("print(1)", "[stdout]\n1")
    assert b.note_turn("print(1)", "[stdout]\n1") is True
    assert b.note_turn("print(2)", "[stdout]\n2") is False       # different cell: reset
    b.note_turn("print(2)", "[stdout]\n2")
    assert b.note_turn("print(2)", "[stdout]\n3") is False       # same cell, new output: reset
    b.note_turn("print(2)", "[stdout]\n3")
    assert b.note_turn("print(2)", "[stdout]\n3") is True        # counting again from the reset


def test_cells_are_compared_stripped_and_views_exactly():
    b = _guarded(3)
    b.note_turn("print(1)\n", "v")
    assert b.note_turn("  print(1)", "v") is True                # whitespace around the cell is not a difference
    b2 = _guarded(3)
    b2.note_turn("print(1)", "v")
    assert b2.note_turn("print(1)", "v ") is False               # the observation is compared byte for byte


def test_zero_disables_the_identical_turns_budget():
    b = _guarded(0)
    for _ in range(50):
        assert b.note_turn("print(1)", "v") is False


def test_no_turns_are_noted_after_a_breach():
    b = _guarded(2)
    b.note_turn("x", "v")
    with pytest.raises(BudgetBreach):
        b.note_turn("x", "v")
    with pytest.raises(BudgetBreach):
        b.note_turn("y", "w")                                    # still refusing, never warn-and-continue


@pytest.mark.parametrize("episode, onset_turn", [("9d9e47fb", 9), ("0c1c397d", 5)])
def test_the_recorded_loops_are_killed_at_the_third_identical_turn(episode, onset_turn):
    """The two production loops (milestones/s4/RESULTS-dflash2-rlm-only.md), replayed
    through the enforcer: the correction lands on the first repeat and the
    kill on the second, i.e. turn onset+2 instead of turn 79 / 116."""
    turns = json.loads((FIXTURES / f"{episode}.json").read_text(encoding="utf-8"))["turns"]
    b = _guarded(3)
    corrected_at = killed_at = None
    for t in turns:
        try:
            if b.note_turn(t["cell"], t["observation_view"]) and corrected_at is None:
                corrected_at = t["turn"]
        except BudgetBreach as exc:
            killed_at = t["turn"]
            assert exc.value.reason == "max_identical_turns"
            break
    assert corrected_at == onset_turn + 1
    assert killed_at == onset_turn + 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_budget.py -q -k "identical_turns or resets_the_count or compared_stripped or disables_the_identical or noted_after_a_breach or recorded_loops"`
Expected: FAIL — `TypeError: Budgets.__init__() got an unexpected keyword argument 'max_identical_turns'`.

- [ ] **Step 3: Implement the budget**

In `rlm/budget.py`, extend the dataclass (keep the new field LAST so the positional constructors in existing tests still work):

```python
@dataclass(frozen=True)
class Budgets:
    """Per-episode limits (spec §5 C5). Built from config, immutable."""

    max_depth: int = 1
    max_subcalls: int = 32
    max_wall_clock_s: int = 900
    max_total_tokens: int = 1_500_000
    max_predict: dict[str, int] = field(default_factory=dict)
    #: v0.3.16: the same (cell, observation) pair repeating on consecutive
    #: root turns is a budget. 0 disables. At max-1 the scaffold corrects,
    #: at max it kills -- measured (milestones/s2/REPLAY-LOOP-AB.md): once a cell has
    #: repeated once the root re-emits it ~64% of the time, once it has
    #: repeated a few times ~92%, and it never calls final_answer from there.
    max_identical_turns: int = 0
```

In `BudgetEnforcer.__init__`, after the existing state initialisation, add:

```python
        self._last_turn_key: tuple[str, str] | None = None
        self._identical_run = 0
```

Add the method after `note_root_usage`:

```python
    def note_turn(self, cell: str, view: str) -> bool:
        """v0.3.16 `max_identical_turns`: count consecutive root turns whose
        (cell, observation) pair is identical. Returns True when a scaffold
        correction is due (the pair has now occurred max-1 times in a row);
        raises BudgetBreach(budget_kill, "max_identical_turns") at max.

        `cell` is compared stripped (fence whitespace is not a decision);
        `view` is the C3 observation BEFORE any scaffold note is appended --
        the caller must pass the un-annotated view, or the note it appended
        last turn would make every repeat look different and the budget would
        never fire. 0 disables.
        """
        self._ensure_not_breached()
        cap = self.budgets.max_identical_turns
        if cap <= 0:
            return False
        key = (cell.strip(), view)
        self._identical_run = self._identical_run + 1 if key == self._last_turn_key else 1
        self._last_turn_key = key
        if self._identical_run >= cap:
            self._breach(Outcome.BUDGET_KILL, "max_identical_turns")
        # The correction exists only when there is a repeat to correct: at
        # cap 2 the first repeat is already the kill, and `run == cap - 1`
        # would otherwise fire on the FIRST occurrence of every pair.
        return cap >= 3 and self._identical_run == cap - 1
```

(`_breach` raises; the `return` after it is reached only below the cap. Review ruling 2026-08-21: the `cap >= 3` guard was added after a reviewer showed the original `run == cap - 1` annotated every fresh pair at cap 2.)

- [ ] **Step 4: Run the budget tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_budget.py -q`
Expected: all PASS, including the pre-existing hypothesis suite (the new field has a default, so `Budgets(1, 32, 900, 1000, {"leaf": 500})` is unchanged).

- [ ] **Step 5: Write the failing config test**

Append to `tests/test_config.py`:

```python
def test_max_identical_turns_ships_at_three_and_refuses_one(minimal_cfg_dict):
    """v0.3.16: the shipped config opts in at 3 (correct at 2, kill at 3);
    1 would kill on the first occurrence of any cell -- 0 (disabled) or >= 2."""
    cfg = Config.model_validate(minimal_cfg_dict)
    assert cfg.scaffold.budgets.max_identical_turns == 3
    raw = copy.deepcopy(minimal_cfg_dict)
    raw["scaffold"]["budgets"]["max_identical_turns"] = 1
    with pytest.raises(ConfigError, match="max_identical_turns"):
        Config.model_validate(raw)          # Config.model_validate re-raises pydantic errors as ConfigError
    raw["scaffold"]["budgets"]["max_identical_turns"] = 0
    assert Config.model_validate(raw).scaffold.budgets.max_identical_turns == 0
```

(`copy`, `pytest`, `Config` and `ConfigError` are already imported at the top of `tests/test_config.py` (lines 1–10); `minimal_cfg_dict` is the shipped `config.yaml` loaded by `tests/conftest.py:31`. `rlm/config.py:375-381` overrides `Config.model_validate` so pydantic's `ValidationError` never escapes — every config test in the file catches `ConfigError`.)

- [ ] **Step 6: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -q -k max_identical_turns`
Expected: FAIL — `AttributeError: 'Budgets' object has no attribute 'max_identical_turns'` (or `ConfigError` on the unknown key once `config.yaml` carries it before the model does).

- [ ] **Step 7: Add the pydantic field, validator, and the config.yaml key**

In `rlm/config.py`, class `Budgets(_Strict)`: insert the new field and validator directly after the `max_total_tokens: int = 1_500_000` line. Leave `restricted_max_wall_clock_s` (above it) and `max_predict: MaxPredict` (below it) exactly as they are:

```python
    max_total_tokens: int = 1_500_000
    #: v0.3.16 (milestones/s2/REPLAY-LOOP-AB.md). The same (cell, observation) pair on
    #: consecutive root turns: correct at max-1, kill at max as
    #: budget_kill/max_identical_turns. 0 disables; 1 is refused because the
    #: first occurrence of any cell already satisfies `run >= 1` and would
    #: kill every turn; 2 kills on the first repeat with no correction.
    max_identical_turns: int = 3
    max_predict: MaxPredict

    @field_validator("max_identical_turns")
    @classmethod
    def _identical_turns_zero_or_at_least_two(cls, v: int) -> int:
        if v == 1 or v < 0:
            raise ValueError("max_identical_turns must be 0 (disabled) or >= 2")
        return v
```

`rlm/config.py:27` currently reads `from pydantic import BaseModel, ConfigDict, ValidationError, model_validator` — change it to `from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator`.

Wire the dataclass in `rlm/episode.py:445-458` — add to the `BudgetLimits(...)` construction:

```python
                max_total_tokens=cfg.scaffold.budgets.max_total_tokens,
                max_identical_turns=cfg.scaffold.budgets.max_identical_turns,
                max_predict={"root": cfg.scaffold.budgets.max_predict.root,
```

(`BudgetLimits` is `rlm.budget.Budgets` imported under that name at the top of `episode.py` — confirm with `grep -n "BudgetLimits" rlm/episode.py`.)

In `config.yaml`, after the `max_total_tokens: 1500000` line in `scaffold.budgets`:

```yaml
    # v0.3.16 (2026-08-21), milestones/s2/REPLAY-LOOP-AB.md. The root re-emitting the SAME
    # cell and getting the SAME observation is a loop the model does not leave
    # on its own: measured, once a cell has repeated once it repeats again ~64%
    # of the time, after a few repeats ~92%, and final_answer is never called
    # from there. Two production episodes ran 70 and 111 identical turns until
    # context_exhausted / wall_clock (milestones/s4/RESULTS-dflash2-rlm-only.md). This is
    # a C5 budget like the others: at max-1 consecutive identical turns the
    # scaffold appends a correction to the observation; at max the episode
    # ends budget_kill / max_identical_turns. 0 disables.
    max_identical_turns: 3
```

- [ ] **Step 8: Run the config and budget tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_budget.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add rlm/budget.py rlm/config.py rlm/episode.py config.yaml tests/test_budget.py tests/test_config.py
git commit -m "C5: max_identical_turns -- the same (cell, observation) pair repeating is a budget

Correct at max-1, kill at max as budget_kill/max_identical_turns (existing
outcome, existing reason convention). Ships at 3. The two recorded loops die
at turn 11 and turn 7 instead of 79 and 116.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: The guard in the episode loop, with the scaffold correction

**Files:**
- Modify: `rlm/episode.py` — add `repetition_observation()` after `no_cell_observation` (line ~316); integrate in `_turn_loop` between `view = observation_view(out, cap)` and `self._put(...)` (lines ~1045-1056); add a reason constant next to `NO_CELL_EXTRACTED` (grep `NO_CELL_EXTRACTED =` near line ~100)
- Test: `tests/test_episode.py`

**Interfaces:**
- Consumes: `BudgetEnforcer.note_turn(cell, view) -> bool` (Task 2).
- Produces: `repetition_observation(cfg: Config) -> str` — the scaffold note appended to the observation when a correction is due. The stored `steps.observation_view` for that turn INCLUDES the note (it is "what the root actually saw", §6), so `rlm replay` needs no change for it. **Do not make replay regenerate this note** the way it regenerates `no_cell_observation` for `REJECTED` steps (`rlm/cli.py:829-831`): for `OK` steps replay reads `observation_view` verbatim, which is exactly why storing the annotated view is sufficient and regenerating would double it.

- [ ] **Step 1: Write the failing episode tests**

Append to `tests/test_episode.py`:

```python
_SAME = "```repl\nprint('same')\n```"


async def test_three_identical_turns_end_as_budget_kill_max_identical_turns(episode_env):
    """v0.3.16: the loop that cost 1,154 s and 1,308 s in production now costs
    three turns. The 2nd identical turn carries the scaffold correction in ITS
    observation; the 3rd is the kill."""
    env = episode_env(root_script=[_SAME, _SAME, _SAME, "```repl\nfinal_answer('x')\n```"])
    res = await env.run()
    assert res.outcome == Outcome.BUDGET_KILL
    assert res.reason == "max_identical_turns"
    turns = [s for s in env.steps() if s["action_type"] == "repl_exec"]
    assert len(turns) == 3, "the kill must land on the third identical turn, not later"
    assert "[scaffold]" not in (turns[0]["observation_view"] or "")
    assert "identical" in turns[1]["observation_view"] and "final_answer" in turns[1]["observation_view"]
    assert env.episode_row()["outcome_reason"] == "max_identical_turns"


async def test_a_root_that_takes_the_correction_finishes_normally(episode_env):
    env = episode_env(root_script=[_SAME, _SAME, "```repl\nfinal_answer('same')\n```"],
                      answer="same")
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    turns = [s for s in env.steps() if s["action_type"] == "repl_exec"]
    assert sum("[scaffold]" in (t["observation_view"] or "") for t in turns) == 1


async def test_the_same_cell_with_a_changing_observation_is_not_a_loop(episode_env):
    """A polling cell is legitimate: same code, new output every time."""
    env = episode_env(root_script=[
        "```repl\nn = 0\n```",
        "```repl\nn += 1\nprint(n)\n```",
        "```repl\nn += 1\nprint(n)\n```",
        "```repl\nn += 1\nprint(n)\n```",
        "```repl\nn += 1\nprint(n)\n```",
        "```repl\nfinal_answer(n)\n```",
    ], answer="4")   # four increments -- the plan originally said "5"; corrected in execution
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    assert all("[scaffold]" not in (s["observation_view"] or "")
               for s in env.steps() if s["action_type"] == "repl_exec")


async def test_the_correction_is_the_observation_the_root_saw(episode_env):
    """State rule: the NEXT root request must contain the stored observation_view
    verbatim (note included) -- replay rebuilds the array from that column."""
    env = episode_env(root_script=[_SAME, _SAME, "```repl\nfinal_answer('same')\n```"])
    await env.run()
    turns = [s for s in env.steps() if s["action_type"] == "repl_exec"]
    corrected_view = turns[1]["observation_view"]
    assert "[scaffold]" in corrected_view                      # the annotated view is the stored one
    next_request = env.blob(turns[2]["root_request_ref"])
    assert corrected_view.encode("utf-8") in next_request        # …and it is what the next request carried
```

(`answer=` is the `Task.answer` the `episode_env` factory already accepts; the shipped default checker for `category="default"` compares exact strings — confirm by reading `Task.check` in `rlm/episode.py:177-200`; if the default category has no checker, pass `category="needle"` and a matching UUID-shaped answer instead, or leave `answer=None` and assert `Outcome.FAIL` with reason `checker_failed` — the point of the second test is only that the episode finished through `final_answer`, so asserting `res.final_answer == "same"` is sufficient if the checker is not available.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_episode.py -q -k "identical_turns or takes_the_correction or changing_observation or correction_is_the_observation"`
Expected: the first test FAILS (the episode runs all four scripted turns and ends `success` — no guard yet); the second and fourth FAIL on their `[scaffold]` assertions; the third PASSES already (no guard, no note) and stays as the regression check that the guard never fires on changing output.

- [ ] **Step 3: Implement the correction text and the loop integration**

In `rlm/episode.py`, next to the other reason constants (find `NO_CELL_EXTRACTED = "no_cell_extracted"`):

```python
MAX_IDENTICAL_TURNS = "max_identical_turns"
```

After `no_cell_observation`:

```python
def repetition_observation(cfg: Config) -> str:
    """The scaffold-authored note appended to the observation when the root
    has just re-run the SAME cell and got the SAME output (v0.3.16, C5
    `max_identical_turns`). It is part of `observation_view` -- what the root
    actually saw -- so `rlm replay` rebuilds the next request from the stored
    column without knowing the guard exists. Generated from config so the
    threshold it names can never disagree with the one that fires.
    """
    cap = cfg.scaffold.budgets.max_identical_turns
    return (
        "[scaffold]\n"
        "That cell is identical to your previous cell and produced identical "
        "output. Re-running it will not change anything, and the episode ends "
        f"without an answer at {cap} identical turns in a row.\n\n"
        "If you have the answer, submit it now with final_answer(value). "
        "Otherwise write a different cell."
    )
```

In `_turn_loop`, replace the block that starts `view = observation_view(out, cap)          # C3, scaffold-side (I1)` through the `self._put(...)` that follows it with:

```python
            view = observation_view(out, cap)          # C3, scaffold-side (I1)
            # v0.3.16 C5 `max_identical_turns`, on the UN-annotated view: the
            # note appended below must not make the next repeat look different.
            correct = False
            if not self._final_emitted:
                try:
                    correct = self.enforcer.note_turn(rt.cell, view)
                except BudgetBreach as breach:
                    self._put({**base, "status": StepStatus.OK, "observation_view": view},
                               {"root_request_ref": request_blob,
                                "observation_full_ref": _full_observation(out)})
                    await self._trip(breach.outcome, breach.reason)
                    return
            if correct:
                view = f"{view}\n\n{repetition_observation(cfg)}"
            self._put({**base, "status": StepStatus.OK, "observation_view": view},
                       {"root_request_ref": request_blob,
                        "observation_full_ref": _full_observation(out)})
```

Everything after (`if self._final_emitted: … _log_final …`, `_note_root_usage`, `conv.append_user(compose_user_message(... observation=view))`) stays as is — `view` now carries the note when one is due, so the next user message and the stored column agree.

`BudgetBreach` is already imported in `episode.py` (used by `_note_root_usage`); confirm with `grep -n "BudgetBreach" rlm/episode.py`.

- [ ] **Step 4: Run the episode tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_episode.py -q`
Expected: all PASS (the whole file — the guard must not disturb the existing scripted episodes, none of which repeats a cell).

- [ ] **Step 5: Replay a guarded episode offline**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -q -k replay`
Expected: PASS (the `mock_episode_env` script is a single `final_answer` turn; this confirms nothing in replay broke). The guard-specific replay check is the `correction_is_the_observation` test above.

- [ ] **Step 6: Commit**

```bash
git add rlm/episode.py tests/test_episode.py
git commit -m "episode: apply max_identical_turns -- correction on the first repeat, kill on the second

The note is part of observation_view (what the root saw), so replay needs
no change; the enforcer sees the un-annotated view so the note cannot mask
the next repeat.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Per-turn root seed (`scaffold.root.seed_schedule`)

**Files:**
- Modify: `rlm/config.py:298-300` (`RootScaffoldCfg`)
- Modify: `rlm/rootclient.py:128-175` (`RootConversation.__init__`, `turn`)
- Modify: `config.yaml` `scaffold.root` block (line ~491)
- Test: `tests/test_rootclient.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `RootScaffoldCfg.seed_schedule: Literal["fixed", "per_turn"] = "fixed"`; `rootclient.turn_seed(base: int, turn: int, schedule: str) -> int` = `base` when `fixed`, `base * 1000 + turn` when `per_turn` (turn is 1-based). No assertion on `turn`: a 32K root window cannot hold 1,000 turns, and raising inside `conv.turn()` would be caught by `_turn_loop`'s blanket `except Exception` and mislabelled `error / server_unreachable`. The bench's `seeded_config` (`rlm/bench.py:141`) keeps patching `sampling.root.seed` per attempt; the schedule composes on top, so an RLM-arm episode's turn seeds are `seed*1000+1, seed*1000+2, …` — distinct across seeds, reproducible, recorded by the snapshot (base seed + schedule). Applies to `RootConversation.turn()` only; B2's one-shot root reduce (`rlm/arms.py:1307`) keeps the base seed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rootclient.py`:

```python
from rlm.rootclient import turn_seed


def test_turn_seed_is_the_base_when_fixed_and_derived_when_per_turn():
    assert turn_seed(2, 1, "fixed") == 2
    assert turn_seed(2, 7, "fixed") == 2
    assert turn_seed(2, 1, "per_turn") == 2001
    assert turn_seed(2, 7, "per_turn") == 2007
    assert turn_seed(3, 1, "per_turn") != turn_seed(2, 1, "per_turn")


async def test_per_turn_schedule_changes_the_seed_every_turn(fake_root_server, minimal_cfg_dict):
    """v0.3.16: the same seed on every turn makes two near-identical turns
    sample identically, which is how a 64% repeat became 70/70 and 111/111 in
    production. With the shipped schedule each turn gets its own seed."""
    conv = fake_root_server.conversation(system="SYS")
    base = minimal_cfg_dict["scaffold"]["sampling"]["root"]["seed"]
    conv.append_user("one")
    await conv.turn()
    first = fake_root_server.last_completion_body["seed"]
    conv.append_user("two")
    await conv.turn()
    second = fake_root_server.last_completion_body["seed"]
    assert (first, second) == (base * 1000 + 1, base * 1000 + 2)


async def test_fixed_schedule_keeps_the_old_behaviour(fake_root_server, minimal_cfg_dict):
    conv = fake_root_server.conversation(system="SYS", seed_schedule="fixed")
    base = minimal_cfg_dict["scaffold"]["sampling"]["root"]["seed"]
    conv.append_user("one")
    await conv.turn()
    conv.append_user("two")
    await conv.turn()
    assert fake_root_server.last_completion_body["seed"] == base
```

Extend `FakeRootServer.conversation` in `tests/conftest.py` (line ~748) to accept the override:

```python
    def conversation(self, *, system: str | None = None, enable_thinking: bool | None = None,
                     seed_schedule: str | None = None, history_mode: str | None = None):
        from rlm.dispatcher import ServerClient
        from rlm.rootclient import RootConversation

        raw = copy.deepcopy(self._base_cfg_dict)
        raw["servers"]["root"]["port"] = self.port
        if enable_thinking is not None:
            raw["scaffold"]["root"]["enable_thinking"] = enable_thinking
        if seed_schedule is not None:
            raw["scaffold"]["root"]["seed_schedule"] = seed_schedule
        if history_mode is not None:
            raw["scaffold"]["root"]["history_mode"] = history_mode
        cfg = Config.model_validate(raw)
        client = ServerClient(self.base_url, timeout=5.0)
        self._clients.append(client)
        return RootConversation(client, cfg, system=system)
```

(`history_mode` is used by Task 6; adding it now keeps the fixture edit in one place.)

Also update the existing `test_root_sampling_params_reach_the_server` (`tests/test_rootclient.py:59`): its `assert body["seed"] == expected["seed"]` becomes `assert body["seed"] == expected["seed"] * 1000 + 1` with the comment `# per_turn schedule (v0.3.16): turn 1`.

Append to `tests/test_config.py`:

```python
def test_root_seed_schedule_ships_per_turn_and_defaults_fixed(minimal_cfg_dict):
    import copy
    from rlm.config import Config
    assert Config.model_validate(minimal_cfg_dict).scaffold.root.seed_schedule == "per_turn"
    raw = copy.deepcopy(minimal_cfg_dict)
    del raw["scaffold"]["root"]["seed_schedule"]          # every pre-v0.3.16 snapshot
    assert Config.model_validate(raw).scaffold.root.seed_schedule == "fixed"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rootclient.py tests/test_config.py -q -k "turn_seed or schedule"`
Expected: FAIL — `ImportError: cannot import name 'turn_seed'` / validation error on the unknown key `seed_schedule` (`_Strict` forbids extras).

- [ ] **Step 3: Implement**

`rlm/config.py`:

```python
class RootScaffoldCfg(_Strict):
    enable_thinking: bool = False
    window_tokens: int
    #: v0.3.16. `fixed` (pre-amendment behaviour, the default so old snapshots
    #: validate to what they ran): the same seed on every turn. `per_turn`:
    #: seed*1000 + turn, distinct per turn, reproducible from the snapshot.
    seed_schedule: Literal["fixed", "per_turn"] = "fixed"
```

`rlm/rootclient.py` — module-level function after `assistant_prefix`:

```python
def turn_seed(base: int, turn: int, schedule: str) -> int:
    """The sampling seed for root turn `turn` (1-based) under
    `scaffold.root.seed_schedule`. `fixed` reproduces the pre-v0.3.16
    behaviour; `per_turn` gives every turn its own seed while staying a pure
    function of (episode seed, turn index), so the snapshot still determines
    the run. The stride is 1,000: turns beyond it would collide with the next
    base seed's schedule, and a 32K root window cannot hold 1,000 turns --
    deliberately not asserted, because an exception here would surface as a
    mislabelled `server_unreachable` in the turn loop."""
    if schedule == "fixed":
        return base
    return base * 1000 + turn
```

In `RootConversation.__init__` after `self._seed = root_sampling.seed`:

```python
        self._seed_schedule = cfg.scaffold.root.seed_schedule
        self._turns = 0
```

In `RootConversation.turn()`, replace the `completion(...)` call's `seed=self._seed` with a derived seed:

```python
        self._turns += 1
        seed = turn_seed(self._seed, self._turns, self._seed_schedule)
        result = await self._client.completion(
            rendered, n_predict=self._max_predict, temperature=self._temperature,
            top_p=self._top_p, seed=seed, stream=True)
```

`config.yaml`, in `scaffold.root`:

```yaml
  root:
    enable_thinking: false
    window_tokens: 32768
    # v0.3.16 (2026-08-21), milestones/s2/REPLAY-LOOP-AB.md §4. Until now the root was
    # sampled with the SAME seed on every turn of an episode. When two
    # consecutive turns see near-identical histories -- a cell re-run against
    # an unchanged observation -- the sampler then makes the same choices, and
    # a per-turn repeat probability of ~64% (measured with varied seeds)
    # became 70/70 and 111/111 in production. `per_turn` = seed*1000 + turn:
    # distinct per turn, still a pure function of the episode seed the bench
    # assigns, still recorded by config_snapshot. `fixed` is the old rule.
    seed_schedule: per_turn
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rootclient.py tests/test_config.py tests/test_episode.py -q`
Expected: PASS. (`tests/test_episode.py::test_every_leaf_call_carries_this_episodes_seed` is about the LEAF seed and is untouched.)

- [ ] **Step 5: Commit**

```bash
git add rlm/config.py rlm/rootclient.py config.yaml tests/test_rootclient.py tests/test_config.py tests/conftest.py
git commit -m "root: per-turn seed schedule -- seed*1000+turn instead of the same seed every turn

Breaks the lock-step that turned a ~64% per-turn repeat into 70/70 and
111/111; still a pure function of the episode seed, still in the snapshot.
Old snapshots validate to 'fixed'.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Make the test fake render history like the real template

**Files:**
- Modify: `tests/conftest.py:207-217` (`_render_chatml`)
- Test: `tests/test_rootclient.py`

**Interfaces:**
- Produces: `_render_chatml(messages, enable_thinking)` renders assistant messages as `<|im_start|>assistant\n<think>\n{reasoning_content|trim}\n</think>\n\n{content}<|im_end|>\n` — the exact branch of `tests/fixtures/repetition/qwen38_chat_template.jinja` (with `preserve_thinking` undefined). User/system messages and the generation prompt are unchanged.

This task deliberately lands BEFORE the history-mode fix: its test documents the defect under the current rule (two think blocks) so Task 6's fix has a failing test to turn green.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rootclient.py`:

```python
@pytest.mark.xfail(strict=True, reason="doubled think block until history_mode raw lands (Task 6)")
async def test_history_renders_one_think_block_per_past_turn(fake_root_server):
    """Qwen3.8's template emits its OWN empty think block in front of every
    past assistant turn (tests/fixtures/repetition/qwen38_chat_template.jinja).
    Storing assistant_prefix(rendered) + raw therefore rendered TWO blocks per
    turn in every S4 and re-validation request (v0.3.16 finding). After the
    fix the history carries exactly one, and the next render is the previous
    render + the raw completion, byte for byte (D26 as intended)."""
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("one")
    first = await conv.turn()
    conv.append_user("two")
    second = await conv.turn()
    past = second.rendered.split("<|im_start|>assistant\n")[1]      # the first turn, as history
    assert past.count("<think>") == 1, past[:120]
    assert second.rendered.startswith(first.rendered + first.raw.strip() + "<|im_end|>
")   # the template trims content
```

- [ ] **Step 2: Make the fake faithful**

Replace `_render_chatml` in `tests/conftest.py`:

```python
def _render_chatml(messages: list[dict], enable_thinking: bool) -> str:
    """Qwen3.8-27B's ChatML shape (tests/fixtures/repetition/qwen38_chat_template.jinja),
    the branches the scaffold exercises. Shared by both fake servers: since the
    S2 leaf-template fix, the LEAF renders through /apply-template too (D14).

    The load-bearing detail: every PAST assistant message is rendered with the
    template's own `<think>\\n{reasoning_content}\\n</think>\\n\\n` in front of
    `content` (preserve_thinking undefined -> the preserving branch), and
    `content` is emitted verbatim -- the template never parses think tags out
    of it. The generation prompt is `<think>\\n\\n</think>\\n\\n` with thinking
    off and `<think>\\n` with it on."""
    parts = []
    for m in messages:
        if m["role"] == "assistant":
            reasoning = (m.get("reasoning_content") or "").strip()
            parts.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n"
                         f"{m['content']}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)
```

- [ ] **Step 3: Run the suite's fake-server consumers and the new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rootclient.py tests/test_dispatcher.py tests/test_episode.py tests/test_cli.py -q`
Expected: everything that passed before still passes (no existing test asserts the exact history shape — `test_conversation_growth_is_append_only` checks only a 200-char prefix), and the NEW test is reported XFAIL; run it once with `--runxfail` to see it fail at `past.count("<think>") == 1` with `2` — the defect, reproduced by the fake. (Execution ruling 2026-08-21: the test is committed as a strict xfail rather than a plain failure so every commit keeps a green suite; Task 6 removes the decorator.)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_rootclient.py
git commit -m "tests: the fake root renders history like Qwen3.8's template, exposing the doubled think block

The template prepends its own think block to every past assistant turn and
emits content verbatim; the fake now does the same, and the new test shows
the stored history carrying two blocks per turn.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `scaffold.root.history_mode: raw` — store what the model generated, not the prompt's tail

**Files:**
- Modify: `rlm/config.py` (`RootScaffoldCfg`)
- Modify: `rlm/rootclient.py` (`split_reasoning`, `history_message`, `RootConversation.turn`, `RootTurn`)
- Modify: `rlm/cli.py:836-843` (replay reconstruction)
- Modify: `rlm/episode.py` (one lifecycle event)
- Modify: `rlm/lifecycle.py:15-25` (`ALLOWED_KINDS` gains `"root_history"` — `Lifecycle.event` raises `ValueError` for any kind not in the set, which would crash the episode task on the first divergence)
- Modify: `config.yaml` `scaffold.root`
- Test: `tests/test_rootclient.py`, `tests/test_cli.py`, `tests/test_config.py`, `tests/test_lifecycle.py`

**Interfaces:**
- Produces: `RootScaffoldCfg.history_mode: Literal["prefix_plus_raw", "raw"] = "prefix_plus_raw"`.
- Produces: `rootclient.split_reasoning(raw: str, *, open_block: bool = False) -> tuple[str, str]` → `(reasoning, content)`. `history_message` passes `open_block=assistant_prefix(rendered).rstrip().endswith("<think>")`: with thinking ON the generation prompt ends in an open `<think>
` and the completion carries only the closing tag, so the split is at the first `</think>` (reasoning stripped, content after the template's `

`); otherwise a LEADING `<think>…</think>` in the completion is split off; else `("", raw)`. (Execution ruling 2026-08-21 — the first draft assumed a leading tag, which the thinking-ON shape never has.)
- Produces: `rootclient.history_message(rendered: str, raw: str, mode: str) -> dict` → `{"role": "assistant", "content": assistant_prefix(rendered) + raw}` for `prefix_plus_raw`; for `raw`: `{"role": "assistant", "content": content}` plus `"reasoning_content": reasoning` only when reasoning is non-empty. **Both `RootConversation.turn()` and `rlm replay` call this one function** — that is what keeps them identical.
- Produces: `RootTurn.prefix_extended: bool | None` — `None` on turn 1, else whether `rendered` started with the previous turn's `rendered + raw.strip()` — the template renders every message's content through `|trim` (`tests/fixtures/repetition/qwen38_chat_template.jinja:103`), so a completion's leading/trailing whitespace never reaches the history. The episode logs ONE lifecycle event `{"kind": "root_history", "state": "diverged"}` the first time it is `False` (a monitor for §7 #3c, never a failure — with thinking on, Qwen's template keeps reasoning but re-trims whitespace, so divergence there is expected and documented).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rootclient.py`:

```python
from rlm.rootclient import history_message, split_reasoning


def test_split_reasoning_separates_a_leading_think_block():
    assert split_reasoning("<think>\nplan\n</think>\n\n```repl\nx\n```") == ("plan", "```repl\nx\n```")
    assert split_reasoning("```repl\nx\n```") == ("", "```repl\nx\n```")
    assert split_reasoning("<think>\n\n</think>\n\nA") == ("", "A")


def test_history_message_under_each_mode():
    rendered = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    old = history_message(rendered, "```repl\nx\n```", "prefix_plus_raw")
    assert old == {"role": "assistant", "content": "<think>\n\n</think>\n\n```repl\nx\n```"}
    new = history_message(rendered, "```repl\nx\n```", "raw")
    assert new == {"role": "assistant", "content": "```repl\nx\n```"}
    thinking = history_message(rendered, "<think>\nplan\n</think>\n\nA", "raw")
    assert thinking == {"role": "assistant", "content": "A", "reasoning_content": "plan"}


async def test_prefix_plus_raw_mode_still_doubles_the_block(fake_root_server):
    """The OLD rule, kept selectable because every episode in the store was
    recorded under it and replay must reproduce their arrays exactly."""
    conv = fake_root_server.conversation(system="SYS", history_mode="prefix_plus_raw")
    conv.append_user("one"); first = await conv.turn()
    conv.append_user("two"); second = await conv.turn()
    past = second.rendered.split("<|im_start|>assistant\n")[1]
    assert past.count("<think>") == 2
    assert second.prefix_extended is False and first.prefix_extended is None


async def test_raw_mode_extends_the_previous_render_byte_for_byte(fake_root_server):
    conv = fake_root_server.conversation(system="SYS", history_mode="raw")
    conv.append_user("one"); first = await conv.turn()
    conv.append_user("two"); second = await conv.turn()
    assert second.rendered.startswith(first.rendered + first.raw.strip() + "<|im_end|>
")   # the template trims content
    assert second.prefix_extended is True
```

Append to `tests/test_config.py`:

```python
def test_history_mode_ships_raw_and_defaults_to_the_old_rule(minimal_cfg_dict):
    import copy
    from rlm.config import Config
    assert Config.model_validate(minimal_cfg_dict).scaffold.root.history_mode == "raw"
    raw = copy.deepcopy(minimal_cfg_dict)
    del raw["scaffold"]["root"]["history_mode"]           # every pre-v0.3.16 snapshot
    assert Config.model_validate(raw).scaffold.root.history_mode == "prefix_plus_raw"
```

Append to `tests/test_cli.py` (next to the other replay tests; `mock_episode_env` runs the shipped config, i.e. `history_mode: raw` after this task):

```python
def test_replay_reconstructs_under_the_snapshots_history_mode(mock_episode_env, capsys, tmp_path):
    """The reconstruction rule is read from the EPISODE's config_snapshot:
    an episode recorded under prefix_plus_raw replays under prefix_plus_raw
    even when the live config says raw, and vice versa. Two turns, so the
    second turn's stored array actually contains an assistant message built
    by history_message() -- a one-turn episode would never compare one."""
    import yaml
    two_turn = ["```repl\nprint(1)\n```", "```repl\nfinal_answer('42')\n```"]

    def run_with(config_file: Path) -> str:
        # FakeRootServer serves script[turns] and never resets `turns` on its
        # own; a second `rlm run` against a spent script gets HTTP 500.
        mock_episode_env.server.script = list(two_turn)
        mock_episode_env.server.turns = 0
        main(["run", str(mock_episode_env.task_file), "--config", str(config_file)])
        return mock_episode_env.last_episode_id()

    raw_cfg = yaml.safe_load(Path(mock_episode_env.config_file).read_text(encoding="utf-8"))
    raw_cfg["scaffold"]["root"]["history_mode"] = "prefix_plus_raw"
    old_cfg = tmp_path / "config-old-history.yaml"
    old_cfg.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
    old_episode = run_with(old_cfg)
    new_episode = run_with(Path(mock_episode_env.config_file))
    for episode_id in (old_episode, new_episode):
        # replay always takes the LIVE config file on the command line; the
        # rule must come from the snapshot regardless of which file that is
        rc = main(["replay", episode_id, "--config", str(mock_episode_env.config_file)])
        assert rc == 0
        assert "message array: OK" in capsys.readouterr().out
```

(`mock_episode_env.server` is the `FakeRootServer`; `script` and `turns` are public attributes, and `tests/test_arms.py` already assigns `fake_root_server.script = [...]` the same way. `main` and `Path` are already imported in `tests/test_cli.py`.)

Append to `tests/test_lifecycle.py`:

```python
def test_root_history_is_a_lifecycle_kind():
    """v0.3.16: the history-divergence monitor is a lifecycle event, and
    Lifecycle.event refuses unknown kinds with ValueError."""
    from rlm.lifecycle import ALLOWED_KINDS
    assert "root_history" in ALLOWED_KINDS
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rootclient.py tests/test_config.py tests/test_cli.py tests/test_lifecycle.py -q -k "split_reasoning or history_message or history_mode or prefix_plus_raw or raw_mode or one_think_block or root_history"`
Expected: FAIL — import errors for `split_reasoning` / `history_message`, unknown config key, `"root_history"` not in `ALLOWED_KINDS`.

- [ ] **Step 3: Implement**

`rlm/config.py` — `RootScaffoldCfg` gains:

```python
    #: v0.3.16. How the root's own reply is stored in its conversation history.
    #: `prefix_plus_raw` (the pre-amendment rule, the default so old snapshots
    #: replay exactly): assistant_prefix(rendered) + raw -- which, because the
    #: chat template prepends its OWN think block to every past assistant turn,
    #: rendered two empty think blocks per turn. `raw`: the model's completion,
    #: with any reasoning split into `reasoning_content`; the template then
    #: renders exactly what the model saw and generated.
    history_mode: Literal["prefix_plus_raw", "raw"] = "prefix_plus_raw"
```

`rlm/rootclient.py` — after `assistant_prefix`:

```python
_LEADING_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def split_reasoning(raw: str, *, open_block: bool = False) -> tuple[str, str]:
    """(reasoning, content) for a raw completion.

    With thinking OFF the prompt closes the think block before the model
    speaks, the model emits no tags, and reasoning is ''. With thinking ON
    the prompt ends in an OPEN `<think>
`, so the completion carries the
    reasoning, then `</think>`, then the answer -- and never an opening
    tag; `open_block=True` says so and the split happens at the first
    `</think>`. A leading `<think>...</think>` in the completion itself is
    also honoured (belt and braces; the template would double it otherwise).
    The template re-renders reasoning inside its own think block and trims
    both parts, so with the model's `
</think>

` the re-render is a
    byte-for-byte extension either way."""
    if open_block:
        head, sep, tail = raw.partition("</think>")
        if sep:
            return head.strip(), tail.lstrip("
")
        return "", raw
    m = _LEADING_THINK_RE.match(raw)
    if not m:
        return "", raw
    return m.group(1).strip(), raw[m.end():]



def history_message(rendered: str, raw: str, mode: str) -> dict[str, str]:
    """The assistant message appended to the root's history for the turn whose
    request rendered as `rendered` and whose completion was `raw`, under
    `scaffold.root.history_mode`. THE one definition: `RootConversation.turn`
    and `rlm replay` both call it, which is what makes the offline
    re-derivation exact (§6 state rule)."""
    if mode == "prefix_plus_raw":
        return {"role": "assistant", "content": assistant_prefix(rendered) + raw}
    if mode != "raw":
        raise ValueError(f"unknown history_mode {mode!r}")
    reasoning, content = split_reasoning(
        raw, open_block=assistant_prefix(rendered).rstrip().endswith("<think>"))
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg
```

`RootTurn` gains a field (keep `slots=True`; add at the end with a default):

```python
    usage: CompletionResult
    prefix_extended: bool | None = None
```

`RootConversation.__init__` gains:

```python
        self._history_mode = cfg.scaffold.root.history_mode
        self._prev_rendered: str | None = None
        self._prev_raw: str | None = None
```

`RootConversation.turn()` — replace the `self.messages.append({"role": "assistant", "content": assistant_prefix(rendered) + raw})` block with:

```python
        extended = None
        if self._prev_rendered is not None:
            # The template renders every message's content through `|trim`
            # (Qwen3.8 template line 103), so the previous turn's completion
            # reappears stripped -- that is the byte-for-byte contract.
            extended = rendered.startswith(self._prev_rendered + self._prev_raw.strip())
        self.messages.append(history_message(rendered, raw, self._history_mode))
        self._prev_rendered, self._prev_raw = rendered, raw

        return RootTurn(raw=raw, cell=cell, view_hash=view_hash,
                         rendered=rendered, usage=result, prefix_extended=extended)
```

Update the D26 comment above it to say: "D26: the history message is `history_message(rendered, raw, mode)` — under `raw`, the template supplies the think block and the next render is the previous render + raw, byte for byte (v0.3.16); under `prefix_plus_raw` the pre-amendment rule is reproduced for old episodes."

`rlm/cli.py` replay (line ~836): replace the assistant append with

```python
        rendered = _rendered(blob_root, step["root_request_ref"])
        messages.append(history_message(
            rendered, step["action_payload"] or "", cfg.scaffold.root.history_mode))
```

and change the import at `rlm/cli.py:95` to `from rlm.rootclient import extract_cell, history_message` (drop `assistant_prefix` if nothing else in the file uses it — `grep -n assistant_prefix rlm/cli.py`). `cfg` here is the EPISODE's config built from its snapshot (see the docstring at `rlm/cli.py:800-805`), so an old episode gets `prefix_plus_raw` by default.

`rlm/episode.py` — in `_turn_loop`, right after `rt = await conv.turn()` succeeds (before `idx = self._alloc()`), add the monitor:

```python
            if rt.prefix_extended is False and not self._history_diverged:
                self._history_diverged = True
                self.lifecycle.event("root_history", state="diverged", turn=turn,
                                      history_mode=cfg.scaffold.root.history_mode)
```

and initialise `self._history_diverged = False` in `__init__` next to `self._final_emitted = False`.

`tests/conftest.py` — make the fake trim content for every role, as the real template does (Task 5's reviewer caught the gap; `qwen38_chat_template.jinja:103`): in `_render_chatml`, bind `content = (m.get("content") or "").strip()` at the top of the loop body and use `content` in place of `m['content']` in both the assistant branch and the user/system branch. Task 5's test `test_history_renders_one_think_block_per_past_turn` asserts with `first.raw.strip()` per the ruling above — update it if it still reads `first.raw`.

`rlm/lifecycle.py` — add the kind to the allowlist (the set at lines 15–25):

```python
ALLOWED_KINDS = frozenset({
    "trace_write_failure",
    "config_refused",
    "handshake_refused",
    "server_health",
    "quiesce_wait",
    "recovery_action",
    "sandbox_spawn",
    "sandbox_death",
    "operator_abort",
    # v0.3.16: a root render that is not a byte-for-byte extension of the
    # previous one (scaffold.root.history_mode monitor); once per episode.
    "root_history",
})
```

`config.yaml`, `scaffold.root`:

```yaml
    # v0.3.16 (2026-08-21). The chat template renders EVERY past assistant turn
    # with its own `<think>\n\n</think>\n\n` before the content; storing
    # assistant_prefix(rendered) + raw put a second one in front of every past
    # turn in every S4 and re-validation request (stimuli in
    # milestones/s2/results/replay-loop-ab/: 17 think blocks for 9 assistant markers).
    # `raw` stores the completion itself, so the next render is the previous
    # render + the model's tokens, byte for byte, and the history is what the
    # model actually saw. `prefix_plus_raw` is the old rule, which `rlm replay`
    # still applies to every episode whose snapshot carries it (or no key).
    history_mode: raw
```

- [ ] **Step 4: Run the full suite**

First remove the `@pytest.mark.xfail(...)` decorator Task 5 placed on `test_history_renders_one_think_block_per_past_turn` (strict xfail would otherwise FAIL the run once the test passes).

Run: `.venv/Scripts/python.exe -m pytest tests -q -x`
Expected: PASS, including Task 5's `test_history_renders_one_think_block_per_past_turn` (now green under the shipped `raw` mode).

- [ ] **Step 5: Replay the two recorded loops from the real store under the new code (old rule via their snapshots)**

Run (bench idle):
```bash
.venv/Scripts/rlm.exe replay 9d9e47fb-9501-429f-a05c-31df2e01e158 --config config.yaml | head -5
.venv/Scripts/rlm.exe replay 0c1c397d-9501-41b7-82ac-f6e2e8138ebf --config config.yaml | head -5
```
Expected: both print `root_view_hash: OK (… stored requests rehashed offline)` and `message array: OK` — their snapshots have no `history_mode`, so they reconstruct under `prefix_plus_raw`. If `message array` mismatches, the default of the pydantic field is wrong (it must be `prefix_plus_raw`).

- [ ] **Step 6: Commit**

```bash
git add rlm/config.py rlm/rootclient.py rlm/cli.py rlm/episode.py rlm/lifecycle.py config.yaml tests/test_rootclient.py tests/test_config.py tests/test_cli.py tests/test_lifecycle.py
git commit -m "root: history_mode raw -- store the completion, let the template supply the think block

history_message() is the one definition both the live turn and rlm replay
use; the mode comes from the episode's snapshot, so the ~700 episodes
recorded under prefix_plus_raw still replay exactly. Adds a once-per-episode
lifecycle event when a render is not a prefix extension of the previous one.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: On-box verification with the real servers

**Files:** none modified (a smoke, recorded in the commit message of Task 8).

- [ ] **Step 1: Validate and run one real episode**

Run (servers down first — `tasklist | grep llama-server` must be empty):
```bash
.venv/Scripts/rlm.exe validate --no-server-probe --config config.yaml   # validate probes live servers otherwise; rlm run does the handshake itself
.venv/Scripts/rlm.exe bench --arm rlm --smoke --tasks synth-01 --config config.yaml --ledger traces/smoke-v0316-ledger.jsonl --report traces/smoke-v0316.md
```
(Execution ruling 2026-08-21: `rlm run` never launches the root — both servers are operator-managed for it — so the smoke uses `rlm bench --smoke`, which owns both servers from `config.yaml`'s launch lines with a throwaway run_id; the scratch `--ledger` keeps §8's pre-registered ledger untouched. Take the episode id from the scratch ledger.)
Expected: `validate` passes; the episode ends `success` (synth-01 solved in 70–90 s in every non-loop seed) and its lifecycle log (`traces/lifecycle.jsonl`, last lines) contains **no** `"kind": "root_history", "state": "diverged"` event — i.e. with the real Qwen3.8 template, every render was a byte-for-byte extension of the previous one under `raw`.

- [ ] **Step 2: Inspect the stored render**

Run:
```bash
.venv/Scripts/python.exe - <<'EOF'
import glob, os, json
D = max(glob.glob('traces/blobs/*'), key=os.path.getmtime)
req = sorted(glob.glob(D + '/step-*.root_request_ref.blob'))[-1]
raw = open(req, 'rb').read(); parts = raw.split(b'\n', 2); st = dict(json.loads(parts[1])['streams'])
rendered = parts[2][st.get('messages', 0):st.get('messages', 0) + st['rendered']].decode('utf-8') if list(st)[0] == 'messages' else parts[2][:st['rendered']].decode('utf-8')
print('assistant markers', rendered.count('<|im_start|>assistant'), '| think blocks', rendered.count('<think>'))
EOF
```
Expected: `think blocks == assistant markers` (one per past turn plus one for the live generation prompt), where the old rule gave `2 × past + 1`.

- [ ] **Step 3: Replay it**

Run: `.venv/Scripts/rlm.exe replay <episode id printed by rlm run> --config config.yaml`
Expected: `root_view_hash: OK`, `message array: OK`.

---

### Task 8: The spec amendment — ARCHITECTURE.md v0.3.16

**Files:**
- Modify: `ARCHITECTURE.md:3` (version), `:4` (status line — one clause), `:143` (§5 C5 budgets), `:189` (§6 outcome_reason conventions), `:175` (§6 final-answer channel — unchanged; add the history-mode paragraph right after it), §10 (new row R15 after R12, line ~446), `:501` (§14 new entry). ARCHITECTURE.md is 595 lines; D26 is not in it (see Global Constraints) — the history-mode rule lives in §6.

- [ ] **Step 1: Version and status**

Line 3: `**Spec version:** \`rlm-runtime-spec-v0.3.16\` (changelog: §14)`.
Line 4, append to the end of the status sentence before "§7 numbers are on-box measurements": ` **v0.3.16 (2026-08-21): C5 gains \`max_identical_turns\`; the RLM arm's root conversation samples per-turn seeds; the root history is stored as the raw completion (\`history_mode: raw\`) — after the DFlash2 re-validation's two 70- and 111-turn repetition loops (\`milestones/s4/RESULTS-dflash2-rlm-only.md\`, \`milestones/s2/REPLAY-LOOP-AB.md\`).**`

- [ ] **Step 2: §5 C5 budgets line (line 143)**

After `` `max_predict` per call, **per role** (root 1024; leaf 512 default, …phantom tokens across 32 subcalls). `` append:

```
**`max_identical_turns` (default 3; 0 disables; 1 refused — v0.3.16).** A root turn whose cell (stripped) and C3 observation are byte-identical to the previous turn's is counted; at `max − 1` consecutive identical turns the scaffold appends `repetition_observation()` to the observation (part of `observation_view`, so replay needs nothing); at `max` the episode ends `budget_kill / max_identical_turns`. Measured basis (`milestones/s2/REPLAY-LOOP-AB.md`): given one repeat the root repeats again ≈64 % of the time, given several ≈92 %, and it never reaches `final_answer` from there; the enforcer sees the un-annotated view so the note cannot mask the next repeat.
```

- [ ] **Step 3: §6 outcome_reason conventions (line 189)**

Change `conventions: which budget breached; …` to `conventions: which budget breached (`wall_clock`, `max_subcalls`, `max_total_tokens`, `max_identical_turns`); …`.

After the final-answer-channel paragraph (line 175) add:

```
**The root history (v0.3.16).** The assistant message stored for a root turn is `history_message(rendered, raw, scaffold.root.history_mode)`, one function used by the live loop and by `rlm replay`. Under `raw` (shipped) it is the model's completion, with any reasoning split into `reasoning_content`; the chat template supplies the think block it renders for every past assistant turn, so the next request is the previous request + the model's tokens, byte for byte — the append-only conversation (D26, `docs/superpowers/plans/2026-08-13-capa1-scaffold.md`) as intended, now through the model's own tokens and not only the prompt. Under `prefix_plus_raw` (every episode before v0.3.16; the default a snapshot without the key validates to) it was `assistant_prefix(rendered) + raw`, which rendered two empty think blocks per past turn in every S4 and re-validation request. Replay reads the mode from the episode's `config_snapshot`. A render that is not a prefix extension of the previous one is a lifecycle event (`root_history / diverged`, added to the §5 lifecycle-log kinds), never a failure. The RLM arm's root conversation also samples a per-turn seed (`scaffold.root.seed_schedule: per_turn`, `seed × 1000 + turn`); one-shot root completions (B2's reduce) keep the base seed.
```

- [ ] **Step 4: §10 — new risk row**

Append to the §10 table after R12:

```
| R15 | **Verbatim-repetition attractor in the root** (v0.3.16): after an empty or unchanged observation following a prose-free cell, Qwen3.8-27B re-emits the identical cell — ≈6 % at onset, ≈64 % after one repeat, ≈92 % after several, `final_answer` never — independent of speculative drafter or build (`milestones/s2/REPLAY-LOOP-AB.md`: DFlash2 61/120, MTP 65/120, none 67/120; two production episodes ran 70 and 111 turns). The same seed on every turn (pre-v0.3.16) turned the per-turn rate into certainty. Entry into the state was ~10× more frequent under the DFlash2 root in the re-validation (11/88 vs 1/90 episodes, code QA only) — unexplained. | C5 `max_identical_turns` (correct at 2, kill at 3) bounds the cost at three turns; `seed_schedule: per_turn` removes the lock-step; `history_mode: raw` removes the doubled think block from the state the attractor lives in. **Owed:** an entry-rate A/B (code QA × 3 seeds, `dflash` vs `mtp`, interleaved) before R4's "success unchanged" is clean; a prompt A/B on non-benchmark fixtures asking for one line of intent before each cell. |
```

- [ ] **Step 5: (removed) — D26 is not in ARCHITECTURE.md.** Its sentence is carried by the §6 paragraph in Step 3. Nothing to do.

- [ ] **Step 6: §14 changelog — new first entry**

Insert above the v0.3.15 entry:

```
- **v0.3.16 — 2026-08-21. Repetition guard bundle (`milestones/s2/REPLAY-LOOP-AB.md`; plan `docs/superpowers/plans/2026-08-21-repetition-guard-bundle.md`).** The DFlash2 root-only re-validation passed 30/30 (`milestones/s4/RESULTS-dflash2-rlm-only.md`) but two `synth` episodes re-emitted one byte-identical cell 70× and 111× until killed. A five-arm replay A/B showed repetition *given the state* is the model's (≈64 % after one repeat, ≈92 % established, in every arm) and an adversarial review showed *entry* was ~10× more frequent under DFlash2 in code QA — recorded as R15, entry-rate A/B owed. Three scaffold changes, all config-recorded: **C5 `max_identical_turns`** (correct at 2, kill at 3 as `budget_kill / max_identical_turns` — no new outcome, no schema or verdict change; the recorded loops now die at turns 11 and 7); **`scaffold.root.seed_schedule: per_turn`** (the RLM arm's root had sampled with one seed on every turn, turning a 64 % repeat into 70/70 and 111/111; B2's one-shot reduce is unchanged); **`scaffold.root.history_mode: raw`** (the stored history carried two empty think blocks per past turn because `assistant_prefix + raw` was re-rendered behind the template's own block; `history_message()` is now the one rule the loop and `rlm replay` share, old snapshots replay under the old rule, and a `root_history / diverged` lifecycle kind monitors prefix extension). §5 C5, §6, §10 R15 amended; no invariant or gate changes; on-box smoke: one real episode under `raw` rendered one think block per turn with zero `root_history / diverged` events and replayed OK.
```

- [ ] **Step 7: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "spec v0.3.16: max_identical_turns, per-turn root seeds, history_mode raw; R15 repetition attractor

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** `milestones/s2/REPLAY-LOOP-AB.md` §4 asks for: a C5 repetition guard with a correction then a named termination (Tasks 2–3 — named by reason, not by a new outcome; the plan says why), a unit test on the two recorded histories (Task 2 step 1, last test), per-turn seed derivation (Task 4), the doubled-think-block fix with `rlm replay` kept exact (Tasks 5–6, with the store's own loop episodes replayed in Task 6 step 5), and a spec amendment with a version bump (Task 8). The entry-rate A/B and the prompt A/B are recorded as owed in R15, not built here — they are measurements, not scaffold changes, and belong to their own plan.

**Placeholder scan.** Every code step carries the code; every run step carries the command and the expected output. Two steps defer to a quick `grep` to confirm an import or a name already in the file (`BudgetLimits`, `BudgetBreach`, `assistant_prefix` in `cli.py`) — those are confirmations, not design left open.

**Pre-execution review (2026-08-21).** Two independent reviewers traced every code block against the working tree. Fixed from their findings: `Config.model_validate` re-raises as `ConfigError` (Task 2 test); `Lifecycle.event` refuses kinds outside `ALLOWED_KINDS` (Task 6 adds `root_history`); `FakeRootServer` never resets its script between `rlm run`s and a one-turn episode never compares an assistant message (Task 6 cli test now two-turn with a reset); D26 is not in ARCHITECTURE.md (Task 8 anchor removed); fixture counts (79 turns / 71 occurrences); the fourth Task 3 test needed its `[scaffold]` assertion to be red first; `turn_seed` must not raise inside `conv.turn()`; the seed schedule's scope (RLM root loop only). Verified by them as correct: dataclass/positional-constructor compatibility, `_breach` raising semantics, scope of every name at the Task 3 insertion point, `RootTurn` slots with a trailing default, replay's use of the episode snapshot, `_Strict` defaults for old snapshots, the blob framing in Task 7, the faithful fake not breaking any existing render-shape assertion, and that no existing scripted test repeats an identical (cell, observation) pair.

**Type consistency.** `note_turn(cell: str, view: str) -> bool` (Task 2) is what Task 3 calls; `history_message(rendered, raw, mode) -> dict` (Task 6) is what `cli.py` and `turn()` both call; `turn_seed(base, turn, schedule) -> int` (Task 4); `RootTurn.prefix_extended: bool | None` (Task 6) is what `episode.py` reads; config keys `scaffold.budgets.max_identical_turns`, `scaffold.root.seed_schedule`, `scaffold.root.history_mode` match between `config.py`, `config.yaml`, the fixture override in `conftest.py`, and the tests.

**Order dependency.** Task 5 must precede Task 6 (its test is the one Task 6 turns green). Task 4's change to `test_root_sampling_params_reach_the_server` depends on `config.yaml` shipping `per_turn` in the same task. Tasks 2→3 are sequential; Task 4 and Tasks 5–6 are independent of 2–3 and of each other.
