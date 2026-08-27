# Prime Agent on the local root — a three-day spike

**Date:** 2026-08-26 · **Status:** PLANNED, not run · **Genre note:** an experiment plan. Its output is a research doc (`docs/research/2026-08-2x-prime-agent-spike.md`) and a results folder; it changes nothing under `src/`, `config.yaml`, or `traces/`.
**Owner's question (2026-08-26):** "Prime Agent built a self-improving RLM that improved scores with frontier models, runs long tasks with its own autonomy — that is what I want." The paper (arXiv:2608.23552) evaluated **only** frontier APIs (Claude Opus 5 / Sonnet 5, GPT-5.6, GLM-5.x, DeepSeek V4 Pro, Kimi K3). Nobody has published what happens when the root is a Q4 27B on a Strix Halo box. This spike measures exactly that, before rlm-halo builds another line of loop code.
**Inputs:** prime-agent v0.8.1 (2026-08-26; `packages/coding-agent/docs/{usage,models,rlm,rlm-runtime,settings,session-format,compaction}.md`, `src/cli/args.ts`, `src/core/autonomous.ts`, `src/core/refinement/refinement.ts`, `src/core/kernel/bootstrap.ts`, `install.sh` — all fetched 2026-08-26 [R]); llama-server `--help` on both local builds and `tools/server/{server-context,server-task}.cpp` (master) [V]; `config.yaml` servers.root; `bench/tasks`, `bench/manifest.json`; S4 run `1cbafb8f` and DFlash2 run `c1740386` per-task costs from `traces/rlm.duckdb` (read-only, 2026-08-26 [V]); `docs/research/2026-08-22-gate0-soak.md` §5–§7; ARCHITECTURE.md §10 R13/R14/R15; `prompts/root.v3.md`, `prompts/strat-aggregation.v2.md`, `prompts/strat-codeqa.v1.md`.
**Evidence discipline:** every reading below is pre-registered with a threshold before any run. Figures from prime-agent's own accounting are [R] until cross-checked against the server's `/metrics` counters, which are the [V] source for tokens. Two independent verification passes (flags/paths against the v0.8.1 sources; protocol logic) were run on the draft of this plan on 2026-08-26 and their corrections are folded in; the corrections that changed a reading are marked **(v2)**.

---

## 0. What this spike answers, and what it cannot

Three questions, in priority order:

| # | Question | Why it matters |
|---|---|---|
| **Q1** | Does a Q4_K_M 27B root **drive an RLM harness at all** — native tool calls, IPython REPL, subagents — on the frozen v1 tasks the scaffold already solves 30/30? | This is F2 (the collapse finding) measured on *our* model with *someone else's* scaffold. It is the out-of-scaffold comparison the 2026-08-25 review said benchmark v2 lacks |
| **Q2** | Does `/refine` (Continual Harness) **produce valid, applied refinements** with this root, and does a refinement made on seen tasks change cost or correctness on an unseen task of the same category? | The self-improvement mechanism the owner wants, on local weights. Issue #1143 says weaker local models fail the JSON contract |
| **Q3** | What does **one long autonomous session** cost and break on this box — server endurance past 35 minutes, repetition loops (R15 / prime-agent #1326), subagent fan-out against two slots? | No `llama-server` in this project's history is provable past ~35 min. Prime Agent's 7-day Factorio run was 23.4M output tokens; at 12 t/s that is 22 days here |

**Not answered here:** delegation quality (prime-agent subagents call the same 27B, there is no leaf); anything about benchmark v2; whether to adopt prime-agent (D-C5 stands — this spike feeds it numbers).

**Comparability caveat, stated once:** prime-agent cannot pin seeds, ships its own ~5–8K-token system prompt, counts tokens through its provider layer, and its runs *rewind* the prefix (compaction, divergent runs) where the scaffold's are append-only. Wall-clock and pass/fail are on-box [V]; token totals are [V] only from `/metrics`; nothing here is seed-matched to S4. Compare against S4/DFlash2 **medians per task**, not cells. **(v2)** The scaffold's root also receives a per-category *strategy block* (`strat-aggregation.v2`, `strat-codeqa.v1`, ~2.5K tokens) that the spike's base prompt does not; §2 A′ handles that asymmetry explicitly.

---

## 1. Topology

```
Windows 11 (host)                              WSL2 Ubuntu 24.04 (mirrored networking)
┌──────────────────────────────────┐           ┌──────────────────────────────────────┐
│ llama-server.exe (Vulkan)        │  :8080    │ prime-agent v0.8.1  (standalone Node) │
│ Qwen3.8-27B Q4_K_M               │◄──────────│   └ IPython kernel (uv Python 3.11)   │
│ -c 131072 -np 2 -a qwen3.8-27b   │  loopback │   └ subagents → same :8080            │
│ --metrics  (-ctxcp: Phase 0)     │           │ user `spike`: no /mnt/*, no interop   │
│ NO leaf server                   │           │ ~/prime-spike (isolated agent dir)    │
└──────────────────────────────────┘           └──────────────────────────────────────┘
```

Verified on the box 2026-08-26: WSL2 Ubuntu 24.04.4, default user `rene` uid 1000; Node v22.22.2 is **rene's per-user install** (`~/.local/bin/node`) — `spike` has none, the installer fetches a standalone Node (network on Day 0 only); Python 3.12.3; 31 GB visible to WSL; `.wslconfig` already has `networkingMode=mirrored`, so `127.0.0.1:8080` inside WSL is the Windows server. **The default `wsl` distro is `docker-desktop`** — every host-side command in this plan says `wsl -d Ubuntu`, and `wsl --shutdown` also stops Docker Desktop. Ports 8080/8081 were closed at survey time; no scaffold process may run during the spike (port and memory contention, and `traces/rlm.duckdb` must stay closed).

**Why the server launch differs from `config.yaml` (recorded, not hidden):**

| Flag | Shipped root | Spike root | Reason |
|---|---|---|---|
| `-c` / `-np` | 32768 / 1 | **131072 / 2** (65,536 per slot) | prime-agent's system prompt + compaction reserve (16,384) + keep-recent (20,000) make 32K unworkable; subagents need a second slot. Root KV is 34 KiB/token → 4.25 GiB; with no leaf resident there is ~45 GiB free of the 64 GiB carve |
| `--jinja` | default-enabled | default-enabled (kept explicit) | **(v2)** both local builds print `--jinja … (default: enabled)`; tool-call rendering/parsing is already on. Explicit for the record only |
| `-ctxcp 0` | absent on root | **Phase 0 step 4b decides** | Gate 0 §6 measured root median turn 13.88 s → 2.48 s (209 checkpoints created / 0 restored) on an *append-only* workload, and its own caveat says "measure a real episode both ways before pinning it". prime-agent's workload rewinds (divergent runs, compaction). The local build's `--checkpoint-min-step` default is 8192, so no checkpoint could sit at a 5–8K system-prompt boundary anyway (llama.cpp #24055 open). Step 4b measures a divergent second request under both settings and applies the pre-registered rule |
| `--metrics` | absent | **present** | Prometheus counters `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, `llamacpp:n_decode_total`; gauges `llamacpp:requests_processing`, `llamacpp:requests_deferred`, `llamacpp:n_busy_slots_per_decode`. **(v2)** There is **no** cumulative request counter — requests are counted from the session JSONLs and the `-lv 4` log |
| `-a qwen3.8-27b` | absent | **present** | prime-agent lists model ids manually; the alias is what `/v1/models` reports and `/props.model_alias` shows |
| DFlash2 draft (`-md … --spec-type draft-dflash`) | present | present (Phase 0 decides) | Keep the shipped decode path unless Phase 0's tool-call round-trip fails on the PR build; fallback is `tools\llamacpp-vulkan\llama-server.exe` (release b10375) without `-md`, at ~3× slower decode |

None of this touches `config.yaml`; the server is launched by hand (§3 step 1) and the argv is recorded verbatim in the results doc.

---

## 2. Pre-registered readings (fixed before any run)

### Phase A — parity on v1 (Q1)

Eight tasks, three runs each, thinking **off** (matches the scaffold's `enable_thinking: false` and S4's cost basis), base prompt **P1** (§4). A task **passes** if ≥2/3 runs pass the repo checker (`rlm.measure.checkers.check`) on the answer. A category **fails** if ≤1 of its tasks pass (aggregation and code QA have 3 tasks each; needle and synthesis 1 each).

| Reading | Threshold | Interpretation |
|---|---|---|
| **A1 — drives it** | ≥6/8 tasks pass | The 27B operates prime-agent's REPL loop on v1. F2 does not apply to this model in this harness. Go to Phase B |
| **A2 — partial** | 3–5/8 pass | Harness-and-prompt-sensitive. Record *which* categories fail and the failure mode (no tool call emitted / malformed tool call / loop / context overflow / wrong answer). Run **A′** on every failing category. Phase B proceeds only on a category that passes in A or A′ |
| **A3 — collapse** | ≤2/8 pass in A **and** the failing categories stay failed in A′ | The same model that scores 30/30 in rlm-halo cannot drive this harness even when handed the scaffold's method. **(v2)** Only with A′ run may this be written as "scaffold shape, not model size, decides"; without A′ it is "harness + prompt". Phase B is skipped; Phase C runs anyway (Q3 is independent) |
| **A′ — strategy arm (v2)** | for each failing category: same tasks × 3 runs with prompt **P2** = P1 + the scaffold's strategy block for that category with the `llm_query`/`chunks`-specific sentences removed (exact P2 text and its sha256 recorded in the results folder *before* the first A′ run) | Separates "the harness cannot be driven" from "the model was not told the method". A category that fails in A and passes in A′ is a *prompt* result, and a cheap one |
| **A-cost** | wall ratio vs DFlash2 per-task median | Per task: prime-agent median wall ÷ scaffold median wall. Expectation from Discussion #1596 (27B on M4 Max): 1.3–1.5×. >2.5× is a finding (system-prompt / post-compaction re-prefill at 208 t/s); <1.0× is a finding (re-examine the scaffold's per-turn overhead). The Phase 0 4b decision on `-ctxcp` is recorded next to this table because it is the dominant term |
| **A-loop** | identical consecutive `ipython` calls | ≥3 identical consecutive tool calls in any run = R15 reproduced outside the scaffold; ≥20 = prime-agent #1326 reproduced. Count from the session JSONL |
| **A-refine (v2)** | model-initiated `refine.run` calls | The system prompt *invites* the model to call `await refine.run()` and the setting `autoRefine.enabled=false` does not stop that. P1 forbids it; any call is counted, its entries archived, and a run that produced a **global** entry has that entry deleted before the next run (logged as an event). Any local self-refinement in Phase A is recorded as `self-refined` |

### Phase B — self-improvement (Q2)

One **interactive TUI** session per category (`/refine` is interactive-only — print mode has no slash-command path **(v2)**), launched with the same cap flags as Phase A. Train tasks run as consecutive prompts in that one session with an operator `/refine` after each; then two held-out tasks of the same category, never shown to that session, run in **fresh print-mode sessions** (Phase A invocation) that load the refined **global** state.

| Reading | Threshold | Interpretation |
|---|---|---|
| **B1 — refine works** | the 4 operator `/refine` calls yield ≥3 with ≥1 *applied* edit, and `harness_state.json` validates (no "Refiner did not return a JSON object") | The JSON contract holds for this root. If ≤1/4: the local root cannot operate the mechanism as shipped — the `session_before_refine` hook (plan with a different model) is the recorded workaround, and rlm-halo's scaffold-applied artifacts are the design answer. **(v2)** Model-initiated refinements are logged separately and are **not** in B1's denominator |
| **B2 — transfer** | held-out task: pass where Phase A failed, **or** median wall ≤0.8× Phase A **and** still passing | A refinement made on seen tasks helped an unseen one. n=2 tasks × 3 runs — this is a *signal*, never a verdict; write it as such |
| **B3 — harm** | held-out task fails where Phase A passed | Refinement hurt. Equally important: it is the failure mode S6-lite's held-out gate exists to catch, observed in the wild |
| **B-content** | qualitative | Copy every refinement (kind ∈ `prompt` / `memory` / `skill` / `subagent` **(v2: kind is `prompt`, not `prompt_note`)**, title, content, before/after). Which kinds did the root use? Does any refinement reference the corpus (i.e. would it be a held-out leak under S6-lite's rule)? |

**Promotion is pre-registered as a file copy** of the train session's final local `harness_state.json` into the global store **(v2)** — never `/refine --global`, which would be a fifth refiner call with new content. Global state is archived and **cleared before step 1 of each category and again before each held-out session**, sha256 recorded at every point, so the second category's train session is not authored under the first category's lessons. B train runs are used for B1 and B-content only; they are **not** wall/token-comparable to Phase A (interactive session, shared context).

### Phase C — one long autonomous session (Q3)

One `--goal` session, depth 2, over the seven aggregation registers, budget 2 h wall / 1.5M tokens.

| Reading | Threshold | Interpretation |
|---|---|---|
| **C1a — endurance, observed (v2)** | server alive after the 2 h session, RSS growth <2×; request count **recorded** (assistant messages across root + child JSONLs, cross-checked with the `-lv 4` log), no threshold | First provable >35-min `llama-server` run in the project. A crash: record the last 50 log lines and the request count |
| **C1b — endurance, scripted (v2)** | the Gate 0 §5 design: a scripted loop of **2,000 short, independent chat completions** (distinct ~200-token user messages, no shared prefix beyond the system line) against the **same server process**, after Phase C; alive at the end, RSS <2×, zero non-200 responses | The #23181 datum this box has never produced, with the request count under our control (~1 h at Phase 0's per-turn cost) |
| **C2 — completes** | `results.json` present with ≥5/7 SEALED counts correct within budget. **Answer key (v2): agg-01 541, agg-02 544, agg-03 514, agg-04 520, agg-05 519, agg-06 539, agg-07 546**, produced by `grep -cE '^Status: SEALED$' bench/corpora/agg-0N.txt` on 2026-08-26 — *not* the task JSONs, whose agg-04..07 questions ask for WITHHELD records | The root finishes a multi-step goal unattended. The goal text makes it write each file's count as soon as it is verified, so compaction cannot summarise an early count away |
| **C3 — fan-out** | subagents spawned, max concurrent, each child's outcome; queueing from `llamacpp:requests_deferred` sampled every 60 s plus per-request timestamps in the child JSONLs **(v2)** | With `-np 2` a third concurrent request queues |
| **C4 — loop** | as A-loop | Long sessions are where R15 lives (70- and 111-turn loops in c1740386) |

### Stop rules

- Phase 0 tool-call round-trip not working after **4 hours** on both builds → stop, record; the finding is "the local serving stack cannot serve prime-agent's tool contract", which is itself a Capa 0 result.
- Any run in which the model's Python touches anything outside `/home/spike` → stop, record, tighten (§3 step 2 makes this impossible by permissions; the rule exists so it is checked).
- Total spike wall time > 3 working days → stop and write up whatever exists. This is a spike, not a slice.

---

## 3. Setup (Day 0, ~half a day)

### Step 1 — root server, by hand, on Windows

PowerShell (never Git Bash — `taskkill //F` from MSYS is a silent no-op; use `Stop-Process`):

```powershell
$srv = 'D:\PROJECTS\rlm-halo-framework\tools\llamacpp-vulkan-dflash2\llama-server.exe'
$argv = @(
  '-m','D:\AI\models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf',
  '-a','qwen3.8-27b',
  '--host','127.0.0.1','--port','8080',
  '-c','131072','-np','2',
  '-ctk','q8_0','-ctv','q8_0','-fa','on','-ub','512','-b','2048',
  '-lv','4','-lm','none','--no-context-shift',
  '--jinja','--metrics',
  # '-ctxcp','0',   # added or not per Phase 0 step 4b
  '-md','D:\AI\models\z-lab\Qwen3.8-27B-DFlash2-GGUF\Qwen3.8-27B-DFlash2-Q4_K_M.gguf',
  '--spec-type','draft-dflash','--spec-draft-n-max','4'
)
New-Item -ItemType Directory -Force D:\spike | Out-Null
$p = Start-Process -FilePath $srv -ArgumentList $argv -PassThru -WindowStyle Hidden `
     -RedirectStandardOutput D:\spike\root.out.log -RedirectStandardError D:\spike\root.err.log
1..60 | % { try { (Invoke-WebRequest http://127.0.0.1:8080/health -UseBasicParsing).StatusCode; break } catch { Start-Sleep 2 } }
# RSS sampler for C1 (leave running in its own window)
while ($true) { "$(Get-Date -Format o),$((Get-Process -Id $p.Id).WorkingSet64)" >> D:\spike\rss.csv; Start-Sleep 60 }
```

Record `$p.Id`, the two log paths, the full argv. Sanity:
- `curl http://127.0.0.1:8080/v1/models` → `data[0].id == "qwen3.8-27b"`
- `curl http://127.0.0.1:8080/props` → `default_generation_settings.n_ctx == 65536`, `total_slots == 2`, `model_alias == "qwen3.8-27b"` **(v2: there are no `n_ctx_slot`/`n_slots` keys)**
- `curl http://127.0.0.1:8080/metrics` → the `llamacpp:*` lines above

**Fallback build** (only if step 4.1's tool-call test fails on the PR build): `D:\PROJECTS\rlm-halo-framework\tools\llamacpp-vulkan\llama-server.exe` with the same argv minus `-md`, `--spec-type`, `--spec-draft-n-max`.

### Step 2 — an unprivileged WSL user that cannot see Windows drives or run Windows binaries

The IPython kernel runs model-written Python as the OS user. Prime-agent's docs: "not a security sandbox". Verified 2026-08-26: `/etc/wsl.conf` already exists (`[boot] systemd=true`, `[user] default=rene`); `/mnt/c` and `/mnt/d` are `drwxrwxrwx` (drvfs without metadata) — **any new user can read and write every Windows file until this step is done**; and `binfmt_misc/WSLInterop` is enabled, so a Windows PE anywhere readable would execute as the Windows user with full `D:\` access — umask on `/mnt/*` does not close that path **(v2)**.

```bash
# as rene (uid 1000), inside `wsl -d Ubuntu`
sudo useradd -m -s /bin/bash spike
# APPEND — do not overwrite the existing [boot]/[user] sections
sudo tee -a /etc/wsl.conf >/dev/null <<'EOF'

[automount]
options = "metadata,uid=1000,gid=1000,umask=077"

[interop]
enabled = false
appendWindowsPath = false
EOF
```

From PowerShell: `wsl --shutdown` (this also stops Docker Desktop), reopen with `wsl -d Ubuntu`, then verify **before anything else**:

```bash
sudo -u spike ls /mnt/c 2>&1 | head -1                  # must be: Permission denied
sudo -u spike ls /mnt/d 2>&1 | head -1                  # must be: Permission denied
sudo -u spike sh -c 'cmd.exe /c echo x' 2>&1 | head -1  # must FAIL (interop off)
sudo -u spike curl -s http://127.0.0.1:8080/health      # must be: {"status":"ok"}
```

Interop off also removes `/mnt/c/...` from `spike`'s PATH, so nothing Windows-side is reachable by name. Copy the corpora **as rene** (who can still read `/mnt/d`) into `/home/spike/tasks/<task_id>/corpus.txt` (§4), plus `agg-01..07.txt` into `/home/spike/tasks/allagg/`, then `sudo chown -R spike:spike /home/spike/tasks`. Record each file's sha256 against `bench/manifest.json`. Nothing else from the repo goes in.

### Step 3 — prime-agent, isolated and offline

As `spike` (`sudo -iu spike`), **in a terminal on Day 0** — `install.sh` prompts on `/dev/tty` for the npm install and the kernel runtime, and it will download a standalone Node because `spike` has none **(v2)**; record the answers:

```bash
cat >> ~/.bashrc <<'EOF'
export PRIME_AGENT_CODING_AGENT_DIR=$HOME/prime-spike        # isolated agent dir
export PRIME_AGENT_KERNEL_VENV=$HOME/prime-spike/kernel-venv  # (v2) the venv ignores the agent dir unless told
export PRIME_AGENT_TELEMETRY=0 DO_NOT_TRACK=1                 # nothing leaves the box
export PRIME_AGENT_INSTALL_UV=1                               # runtime uv bootstrap without a prompt
export PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=1              # (v2) prepare the kernel venv at install time
export PRIME_AGENT_VERSION=0.8.1                              # pin
EOF
source ~/.bashrc; mkdir -p $PRIME_AGENT_CODING_AGENT_DIR
curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh
```

`$PRIME_AGENT_CODING_AGENT_DIR/models.json`:

```json
{
  "providers": {
    "llamacpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStore": false,
        "supportsStrictMode": false
      },
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B Q4_K_M (llama-server, Vulkan)",
          "reasoning": true,
          "contextWindow": 65536,
          "maxTokens": 8192,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "compat": { "thinkingFormat": "qwen-chat-template" }
        }
      ]
    }
  }
}
```

`$PRIME_AGENT_CODING_AGENT_DIR/settings.json`:

```json
{
  "defaultProvider": "llamacpp",
  "defaultModel": "qwen3.8-27b",
  "defaultThinkingLevel": "off",
  "rlmMaxDepth": 1,
  "autoRefine": { "enabled": false },
  "telemetry": { "enabled": false },
  "idleEvictionMinutes": "off",
  "retry": { "maxRetries": 3, "provider": { "timeoutMs": 3600000 } }
}
```

`autoRefine` off so the only *scheduled* refinements are the operator's (the model can still call `refine.run` — see A-refine); `rlmMaxDepth: 1` for Phases A/B (v0.8.1 changed the default to 2), raised to 2 in Phase C; `idleEvictionMinutes: off` because #1072 (llama.cpp endpoint, idle-eviction sweep leaks worker+kernel pairs); `retry.provider.timeoutMs` set explicitly **(v2)** because the SDK default (10 min) can be hit by a post-compaction 50–60K-token prefill at 208 t/s and a retry would silently inflate request counts and cost.

Add `--offline` to every invocation. Never run `/traces on` or `/share`.

### Step 4 — Phase 0 gates (must all pass before Day 1)

1. **Tool call round-trip, raw** (bypasses prime-agent): POST `/v1/chat/completions` with one `tools` entry `{name:"ipython", parameters:{code:string}}`, `tool_choice:"auto"`, user message "Use the tool to print 2+2", `stream:true`, `stream_options:{include_usage:true}`. Pass = a `tool_calls[0].function.name == "ipython"` with parseable JSON `arguments`, **and** the final streamed chunk (empty `choices`) carries `usage` with `prompt_tokens_details.cached_tokens`. If `usage` is absent, set `supportsUsageInStreaming: false` and rely on `/metrics` only.
2. **Harness smoke:** `prime-agent -p --offline --thinking off "Using the ipython tool, compute 17*23 and reply with just the number."` → prints `391`; the session JSONL under `$PRIME_AGENT_CODING_AGENT_DIR/sessions/` shows one `ipython` tool call and nonzero `usage.input/output`.
3. **Kernel bootstrap:** the uv venv is at `$HOME/prime-spike/kernel-venv` with Python 3.11 and `ipykernel`; nothing was created under `~/.prime/`; `prime-agent doctor` is clean.
4. **Per-turn cost and the `-ctxcp` decision (v2):**
   - 4a. Time the smoke run three times; record median wall and `/metrics` deltas. This is the fixed per-turn cost of prime-agent's system prompt on this box (expected 25–40 s cold at 208 t/s prefill; near zero warm on an identical prompt).
   - 4b. Two conditions, server relaunched between (default `-ctxcp` vs `-ctxcp 0`): run the smoke prompt, then a **second prompt with the same system prompt and a different user message**; record `timings.cache_n` and wall of the second request under each. **Rule:** use `-ctxcp 0` if `cache_n` on the divergent request is equal under both settings (checkpoints buy nothing on this build/model); otherwise keep the default and record the checkpoint tax as part of A-cost. Either way the chosen argv and both measurements go in the results doc.
5. **Corpus file test:** `prime-agent -p --offline --thinking off --cwd /home/spike/tasks/agg-03 "How many lines does corpus.txt have? Use the tool."` → an integer, no crash on a 474 KB read.
6. **Gate mechanism (v2):** `prime-agent -p --offline --thinking off --autonomous --autonomous-gate 'test -s /home/spike/tmp/answer.txt' --autonomous-max-continuations 1 --cwd /home/spike/tmp "Write the text FINAL: 391 to ./answer.txt using the tool, then reply with the same line."` → exit code 0, the JSONL contains **no** host continuation message ("No human input is available in autonomous mode…"), and `answer.txt` exists. Without a gate, autonomous mode injects that continuation after the model's own stop and the run's second pass would be what gets scored.
7. **Artifact location:** with `--session-dir /home/spike/runs/A/sessions`, harness state and kernel snapshots appear under `/home/spike/runs/A/session-artifacts/<session-id>/` — i.e. `$(dirname <session-dir>)/session-artifacts/` **(v2)**, not under the agent dir. `usage.py` reads that path.

---

## 4. Tasks and prompts

### Phase A task set (8 tasks; all frozen v1, `bench/tasks/*.json`; answers and medians verified 2026-08-26 [V])

| task | category | corpus (leaf tokens) | expected | scaffold medians S4 → DFlash2 (wall s / tokens / turns) |
|---|---|---|---|---|
| agg-03 | aggregation, regex-solvable | 129,112 | `514` | 50.7 / 7,306 / 4 → 37.1 / 7,601 / 4 |
| agg-04 | aggregation, regex-defeating | 129,130 | `594` | 97.0 / 17,918 / 6 → 76.8 / 19,394 / 7 |
| agg-07 | aggregation, regex-defeating (Phase B held-out) | 129,127 | `576` | 84.8 / 14,405 / 6 → 95.5 / 21,802 / 7 |
| codeqa-01 | code QA (`in_flight`) | 103,861 | `rlm/dispatcher.py` | 69.3 / 8,856 / 5 → 66.7 / 11,405 / 6 |
| codeqa-03 | code QA | 103,861 | `rlm/dispatcher.py` | 72.2 / 8,788 / 5 → 108.0 / 24,567 / 10 |
| codeqa-05 | code QA (Phase B held-out) | 103,861 | `rlm/trace.py` | 87.5 / 13,816 / 7 → 98.4 / 17,149 / 7 |
| needle-02 | needle | 63,497 | `da7ed32f-12d7-2f9b-91b8-7db7572549a0` | 78.8 / 13,077 / 6 → 43.3 / 8,872 / 5 |
| synth-02 | synthesis | 35,974 | `Nethnethercleave Ledgerhouse` | 95.4 / 15,733 / 6 → 142.9 / 36,555 / 8 (83 leaf calls) |

Aggregation and code QA are the two categories where the scaffold's root wins by **writing a real program** and every baseline scores 0 — the categories that test Q1 hardest. Needle and synthesis are the sanity floor (B1 passes them 24/24 on Vulkan).

Each run gets its own working directory `/home/spike/work/A/<task>/run<i>/` containing a copy of `corpus.txt` (so `answer.txt` is per run and the model can write freely there).

### Prompt P1 (identical shape for every task; only the question changes)

```
The file ./corpus.txt in the current directory is a large document (about {CHARS} characters).
Question: {TASK_TEXT}
Work in the ipython tool: load the file into a variable and compute the answer with code; do not
paste the document into the conversation. Do not call refine.run and do not modify the continual
harness. When you are done, write exactly one line of the form
FINAL: <answer>
to ./answer.txt, and reply with that same line.
```

`{TASK_TEXT}` is the `text` field of the task JSON verbatim. "Compute with code" is the generic instruction the scaffold's `root.v3` also gives; it names no method. The scaffold *additionally* gives a per-category strategy block — that is prompt **P2** in arm A′ (§2), whose exact text is derived from `prompts/strat-<category>.v*.md` with the `llm_query`/`chunks` sentences removed and its sha256 recorded before any A′ run.

### Invocation (Phase A, A′, and Phase B held-out)

```bash
T=agg-03; i=1; W=/home/spike/work/A/$T/run$i; mkdir -p $W; cp /home/spike/tasks/$T/corpus.txt $W/
snap pre $W                                   # (§5) sha256 of global + local harness state
curl -s 127.0.0.1:8080/metrics > $W/metrics.pre
cd $W && /usr/bin/time -f '%e' -o $W/wall.txt \
prime-agent -p --offline --thinking off \
  --autonomous --autonomous-gate "test -s $W/answer.txt" \
  --autonomous-max-continuations 1 --autonomous-max-turns 25 \
  --autonomous-max-tokens 300000 --autonomous-timeout-ms 1300000 \
  --session-dir /home/spike/runs/A/sessions --cwd $W \
  "$(cat $W/prompt.txt)" > $W/stdout.txt 2> $W/stderr.txt; echo $? > $W/exit.txt
curl -s 127.0.0.1:8080/metrics > $W/metrics.post
snap post $W
```

Budgets mirror the scaffold's C5 caps: `max_wall_clock_s: 1300`; a 300K-token cap that no S4 episode approached except the two R15 loops (1.08–1.26M); 25 turns (the scaffold has **no** turn cap — S4's longest scored episode was 15 turns; 25 is a loop guard, not a parity number **(v2)**). The gate makes the run end at the model's own stop; `--autonomous-max-continuations 1` is the flag's minimum and only fires if the model never writes `answer.txt`. Runs are sequential; never two harness runs at once.

### Phase B protocol (per category; aggregation shown) **(v2 rewritten)**

0. Archive and clear the global store: `mv $PRIME_AGENT_CODING_AGENT_DIR/harness /home/spike/runs/B/archive/global-before-agg-$(date +%s)`; record that it is absent.
1. Working dir `/home/spike/work/B/agg/` with **only** `agg-03.txt`, `agg-04.txt`, `agg-05.txt`, `agg-06.txt` (held-out corpora never present). Launch the **TUI**: `prime-agent --offline --thinking off --autonomous-max-turns 25 --autonomous-max-tokens 300000 --autonomous-timeout-ms 1300000 --session-dir /home/spike/runs/B/agg/sessions --cwd /home/spike/work/B/agg`, then `/autonomous on`.
2. In that session, as consecutive prompts: P1 for agg-03 (file name substituted) → `/refine` → P1 for agg-04 → `/refine` → agg-05 → `/refine` → agg-06 → `/refine`. After each `/refine`, copy `/home/spike/runs/B/agg/session-artifacts/<id>/harness/harness_state.json` and the `[refinement]` transcript entry to `/home/spike/runs/B/agg/refine<k>/`. Snapshot the global store after every prompt (A-refine rule).
3. **Promote by file copy:** `mkdir -p $PRIME_AGENT_CODING_AGENT_DIR/harness && cp <final local harness_state.json> $PRIME_AGENT_CODING_AGENT_DIR/harness/harness_state.json`; sha256 recorded.
4. Held-out, **fresh print-mode sessions** with the Phase A invocation: **agg-07 × 3** (never seen) and **agg-03 × 3** (seen re-test). Compare with Phase A per B2/B3.
5. Repeat 0–4 for code QA: train codeqa-01, 03, 02, 04 (bundle copied as `code-bundle.txt`), held-out codeqa-05 × 3 + codeqa-01 × 3 re-test. Step 0 guarantees code QA's train session never sees aggregation's entries.

### Phase C protocol

Fresh session, `rlmMaxDepth: 2` (set in settings.json for this phase), `--cwd /home/spike/work/C/` containing `agg-01.txt … agg-07.txt`:

```bash
prime-agent -p --offline --thinking off \
  --goal "There are seven register files agg-01.txt … agg-07.txt in the current directory. For each file, count the records whose Status line reads exactly SEALED. Verify each count by a second, independently written method. As soon as one file's count is verified, merge it into ./results.json as {\"agg-0N\": <int>} (create the file on the first write; keep earlier entries). You may delegate files to subagents. Do not call refine.run. Stop when all seven are written." \
  --goal-token-budget 1500000 \
  --autonomous --autonomous-gate "python3 -c \"import json;d=json.load(open('/home/spike/work/C/results.json'));assert len(d)==7\"" \
  --autonomous-max-continuations 20 --autonomous-max-turns 400 \
  --autonomous-max-tokens 1500000 --autonomous-timeout-ms 7200000 \
  --session-dir /home/spike/runs/C/sessions --cwd /home/spike/work/C
```

Answer key (§2 C2): 541 / 544 / 514 / 520 / 519 / 539 / 546. During the run: `/metrics` every 60 s to CSV (`prompt_tokens_total`, `tokens_predicted_total`, `requests_processing`, `requests_deferred`), the RSS sampler from step 1, `tail -f` on `root.err.log`.

**C1b after Phase C**, same server process: `tools/endurance.py` — 2,000 sequential `/v1/chat/completions` calls, each a distinct ~200-token user message (no tools, `max_tokens: 64`), logging status, `timings`, wall; then RSS and `/health`.

---

## 5. Measurement plumbing (write on Day 0, ~150 lines total, lives in `docs/research/2026-08-26-prime-agent-spike/tools/`)

Runs on the **host** (the spike user has no repo access): copy `/home/spike/runs` and `/home/spike/work` out with `wsl -d Ubuntu --cd /home/spike -- tar cf - runs work | tar xf - -C D:\spike\out` (run as rene, who can read `spike`'s tree via `sudo`), then:

- **`snap`** (bash, in WSL) — `snap pre|post <dir>`: sha256 + copy of `$PRIME_AGENT_CODING_AGENT_DIR/harness/*` and of the run's local `session-artifacts/<id>/harness/` into `<dir>/harness.<pre|post>/`; a diff between pre and post is a model-initiated refinement (A-refine rule).
- **`score.py`** — for a run dir: `answer.txt` if present, else the **first** assistant message in the session JSONL containing `^FINAL:` **(v2)**; record whether any later `FINAL:` differs; `rlm.measure.checkers.check(checker, got, want)` from the repo venv; write `pass|fail`.
- **`usage.py`** — parse the session JSONL(s) under `$(dirname session-dir)/`: per assistant message `usage.{input,output,cacheRead}`, `stopReason`, tool-call count, identical-consecutive-call streak (A-loop / C4), `refine.run` calls, host continuation messages (`continuationsUsed`), `stopReason == "error"` count, child sessions and `child_usage_attributed`; `metrics.pre/post` deltas. One CSV row per run: `task, run, arm(P1|P2), pass, wall_s, turns, tool_calls, tokens_in_harness, tokens_out_harness, prompt_tokens_metrics, predicted_tokens_metrics, max_identical_streak, refine_calls, continuations, errors, stop_reason, subagents, exit_code`.
- **`compare.py`** — join with the per-task scaffold medians in §4 (S4 `1cbafb8f`, DFlash2 `c1740386`) and print the A-cost ratios.
- **`endurance.py`** — C1b driver (above).

Nothing in the spike reads or writes `traces/rlm.duckdb`.

---

## 6. Schedule

| Day | Work | Output |
|---|---|---|
| **0** (½ day) | §3 steps 1–4 (installer interactive; 4b needs two server launches); write §5 tools; copy corpora | Phase 0 gate log; per-turn cost; the `-ctxcp` decision |
| **1** | Phase A: 8 tasks × 3 runs (~24 runs; 1–5 min each plus prefill, 2–3 h). If A2/A3: A′ on the failing categories the same evening (≤9 tasks × 3) | `runs/A/`, CSV, A1–A3 (+A′), A-cost, A-loop, A-refine |
| **2** | Phase B: two categories (train 4 in TUI + `/refine` ×4; held-out 2 × 3 + re-test 1 × 3, print mode) | `runs/B/`, every refinement archived with sha256 trail, B1–B3, B-content |
| **3** | Phase C (2 h, monitored) → C1b (~1 h) → write-up | `runs/C/`, endurance CSVs, C1a/C1b/C2–C4; `docs/research/2026-08-2x-prime-agent-spike.md` |

---

## 7. What each outcome means for the build order (pre-registered)

| Outcome | Next step |
|---|---|
| A1 + B1 pass | The local root can run a self-improving RLM harness. rlm-halo's S6-lite v0 spec is written **on this evidence**: the four artifact kinds prime-agent uses (prompt / memory / skill / subagent) map onto the scaffold-applied, sha-pinned, held-out-gated artifacts of ARCHITECTURE §9 S6; what rlm-halo adds is the gate and the replay. A-cost sets the per-turn budget |
| A1 but B1 fails | The root drives the loop but cannot author refinements in prime-agent's JSON contract. rlm-halo's design (scaffold authors the artifact from traces; the model proposes in free text) is the answer — record that as the reason, not a preference |
| A2, recovered by A′ | A prompt result: the category needs the method stated. Cheap, and it says the scaffold's strategy blocks are load-bearing — a fact for benchmark v2's design and for what S6-lite's `prompt` artifacts should learn first |
| A2, not recovered by A′ | Category-specific collapse. Feed the failing categories to benchmark v2's design as the concrete F2 reading the 2026-08-25 review said was missing; S5's A3B-as-root / mit-oasys LoRA rows get a real trigger |
| A3 (with A′ run) | Publish: same 27B, 30/30 in one scaffold and ≤2/8 in another even with the method supplied. The scaffold-shape result is the deliverable; the LoRA row moves up the S5 order |
| C1a crash or C1b fails | The first #23181 datum on this box. Endurance becomes the next Capa 0 item before any long-horizon slice, as the 2026-08-22 design already argued |
| C2 passes | An unattended multi-file goal completes on a local 27B. That is the demonstration the owner asked for; its cost (tokens, hours) is the number that prices "long tasks with autonomy" locally |

---

## 8. Hazards carried in, with the mitigation each has

| Hazard | Mitigation in this plan |
|---|---|
| Model code runs unsandboxed as the OS user (prime-agent docs) | `spike` user; `/mnt/*` unreadable by umask; **interop off** so no Windows binary can be executed (§3.2, all four checks verified before use); no repo, no keys |
| Data leaving the box (telemetry on by default; `/traces`) | `PRIME_AGENT_TELEMETRY=0`, `DO_NOT_TRACK=1`, `telemetry.enabled=false`, `--offline` (also sets `PI_OFFLINE=1`); never `/traces on`, never `/share`. Network is needed once, on Day 0, for the installer |
| **Autonomous continuation nudge (v2)** — without a gate, the host injects "No human input is available… Do not end the session yourself" after the model's stop, and a second pass gets scored | `--autonomous-gate "test -s answer.txt"` on every scored run (Phase 0 gate 6); `score.py` takes the first `FINAL:`; `continuations` is a CSV column |
| **Model-initiated refinement (v2)** — the system prompt invites `await refine.run()`, including `global_=True`; `autoRefine` off does not prevent it | P1 forbids it; `snap` before/after every run; global entries created outside a planned step are deleted before the next run and logged; A-refine reading |
| R15 repetition attractor — the scaffold's guards (`max_identical_turns`, per-turn seeds) do not exist in prime-agent; #1326 reports 20–50× identical calls on a local Qwen | Turn/token/time caps on every run; identical-streak counter in `usage.py`; A-loop/C4 readings |
| Endurance (#23181) — crash after 1,500–2,000 requests, right after a checkpoint line | C1a observed + C1b scripted to 2,000 requests; RSS sampler; `-ctxcp` per step 4b |
| **Compaction (v2)** — tool results truncated to 2,000 chars and history replaced past `contextWindow − reserveTokens`; an early result can be summarised away | Phase C writes each count as soon as verified; Phase A tasks are single-answer and short |
| **Retry storms (v2)** — agent-level retry ×3 with backoff; SDK provider timeout default 10 min can be hit by a 50–60K re-prefill | `retry.provider.timeoutMs: 3600000`; `errors` column from `stopReason`; request counts from JSONL + log, never from a counter that does not exist |
| `--no-context-shift` → HTTP 400 past the slot size | `contextWindow: 65536` so compaction fires first; a 400 in any run is its own stop reason |
| Two slots, subagents queue | `llamacpp:requests_deferred` sampled; C3 records queueing |
| Control-token injection (literal `<|im_end|>` in corpus text becomes a real token on both endpoints) | The v1 corpora are synthetic and free of it (S4 ran them); P1 forbids pasting the document into the conversation |
| `maxTokens` clamp at 32,000 on the wire (#755) | Irrelevant at `maxTokens: 8192` |
| `traces/rlm.duckdb` single writer | The spike never opens it; scoring imports `rlm.measure.checkers` only |
| Comparability | No seed pinning; prime-agent's own system prompt; rewinding prefixes; DFlash2 root; strategy-block asymmetry handled by A′. Stated in §0 and repeated in the results doc; comparisons are per-task medians, [V] tokens from `/metrics` |

---

## 9. Not decided here

- Whether any of prime-agent's code is adopted (D-C5 stands; §7 feeds it numbers).
- Benchmark v2's design (unchanged).
- Whether `-ctxcp 0` should move into `config.yaml` for the scaffold's root — a §8 comparability event, owner's call, now informed by Phase 0 step 4b's two-condition measurement (which is the "real episode both ways" Gate 0 §6 asked for).
