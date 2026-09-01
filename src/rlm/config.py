"""Config schema + prompt registry (spec §5, Appendix A).

`Config` is the contract every other component reads: `extra="forbid"`
everywhere (a typo'd key silently running at defaults is an I1-grade hazard),
cross-field validators encode the §4/§5 arithmetic, and `config_snapshot` is
the canonical, stably-ordered JSON dump of the *validated* model.

`Config.model_validate` is overridden so pydantic's `ValidationError` never
escapes this module — callers only ever catch `rlm.errors.ConfigError`.

Prompt-pinning decision (see the module docstring on `PromptRegistry` below
and the report): `sha256` fields are nullable ("not yet pinned"), and the
file-existence / hash-match check happens only inside `PromptRegistry.load()`
— an explicit call, never implicit in `Config.model_validate()`. This is
required for `config.yaml` to load today, since Task 14 has not yet written
the prompt files it references.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from rlm.errors import ConfigError
from rlm.context.truncate import MIN_MARKER_CAP

# The package's own data directory: the default config and the frozen prompts.
_DATA = Path(__file__).resolve().parent / "_data"


def default_config_path() -> Path:
    """The config the package ships with, resolved against the package itself.

    A copied `rlm/` has no repo to read, so this is what `load_config()` falls back
    to. It is derived from the repo's `config.yaml` and differs from it in exactly
    TEN leaves: the three server `model` paths, their three `backend_dir`s, the root's
    `dflash` flag AND its `extra_flags` list (the DFlash set cannot leave by halves --
    three validators interlock), and the two sandbox paths. Counted, not estimated:
    `tests/test_default_config.py` asserts the differing set is exactly those ten and
    that everything else is structurally identical, so the two cannot drift.
    """
    return _DATA / "config.default.yaml"


def resolve_prompt_path(path: Path) -> Path:
    """Resolve a prompt path as written, then inside the package.

    Two roots, one path string, and the order matters. Step one is the path exactly
    as the config (or a stored `config_snapshot`) declares it, so every one of the
    614 recorded episodes keeps resolving the way it did when it ran, and
    `trace/replay.py` needs no change. Step two is the package's own `_data/prompts/`,
    which is what makes a copied `rlm/` work with no repo around it.

    The path is NOT resolved at parse time, deliberately: `config_snapshot` records
    `ref.path` verbatim, and that record is evidence. Resolving early would write an
    absolute, machine-specific path into every future snapshot and change the shape
    of a format that stored runs are already in.

    The safety net for step two is the sha256 pin the caller checks next: if the
    package copy has diverged from the file a run actually used, the hash mismatch
    catches it. Without a pin there is no net, which is the argument for pinning
    every prompt rather than for resolving differently.
    """
    if path.exists():
        return path
    inside = _DATA / "prompts" / path.name
    return inside if inside.exists() else path


class _Strict(BaseModel):
    """Shared base: `extra="forbid"` on every nested model, not just the root."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# servers.*
# --------------------------------------------------------------------------- #


class ServerConfig(_Strict):
    model: Path
    #: Root speculative-decoding DECLARATIONS. Exactly the same contract as each
    #: other: the flags that do the work live in `extra_flags`, and a
    #: cross-field validator refuses any state where the two disagree. They are
    #: separate booleans rather than one enum because `--spec-type` is itself a
    #: comma-separated list upstream, so "which methods are on" is not a
    #: single-valued question.
    mtp: bool = False
    dflash: bool = False
    port: int
    backend: str
    backend_dir: Path
    log_path: Path
    ctx: int
    parallel: int
    cache_type: str
    flash_attn: str
    ub: int
    b: int
    extra_flags: list[str] = []
    #: THIS server's per-call HTTP timeout, overriding
    #: `scaffold.retries.per_call_timeout_s` for the client that talks to it.
    #: `None` keeps the global value, which is what every pre-S4 config gets.
    #:
    #: It exists because one number cannot serve two topologies. The global 240
    #: s is sized for the RLM profile, whose slots are 2,560 tokens and whose
    #: calls answer in seconds; `servers.bench_leaf` gives B1/B3 a 262,144-token
    #: slot and pre-registers corpora that fill it, where a single first byte is
    #: minutes away (measured on this box: 227 s at 104K tokens). Raising the
    #: global instead would slacken every RLM-profile call's deadline by the
    #: same factor, and C5's wall clock is not a substitute -- it kills the
    #: EPISODE, so a hung call would burn the whole budget instead of failing
    #: its own attempt.
    per_call_timeout_s: int | None = None
    #: Environment overlay for the launch, MERGED over `os.environ` (never
    #: substituted for it -- a child with a two-entry environment has no PATH
    #: and loads no backend DLL). Here rather than in code for the same reason
    #: `extra_flags` is: a launch value invented in code is one
    #: `config_snapshot` cannot record (R11), and ROCBLAS_USE_HIPBLASLT is set
    #: for every leaf measurement in `milestones/s2/` (`milestones/s2/run_occupancy.py:455`) while
    #: the scaffold's own launch path dropped it.
    env: dict[str, str] = {}


class LeafServerConfig(ServerConfig):
    """The leaf, which is the only server whose slots are allocated by the
    scaffold -- hence the only one carrying a slot policy.

    `slot_policy` has exactly ONE supported value, and the point of declaring
    it anyway is that the choice becomes explicit and greppable instead of
    implied by dispatcher code. R13 (spec §10): a slot that has held one
    document injects that document's content into answers about the next one
    (shared slot 24/54 vs virgin slot 0/54, p = 4.4e-9). Every cheaper option
    was measured and leaks -- `cache_prompt: false` 15/18, `--parallel 1`
    4/18 (it makes reuse mandatory), `action=erase` 33/54, and the
    non-recurrent control leaf leaked MORE than the hybrid (gemma-4-12B-it:
    no `ssm.*` keys, but SWA-interleaved -- not a full-attention model). There is
    nothing else to put here.
    """

    slot_policy: Literal["never_reuse"] = "never_reuse"


class ServersConfig(_Strict):
    root: ServerConfig
    leaf: LeafServerConfig
    fallback_leaf: ServerConfig | None = None
    #: §8's B1/B3 relaunch profile (S4): the SAME leaf weights on the same port,
    #: relaunched with two slots of the model's full native window so the
    #: single-shot baselines are measured on what the model can actually read
    #: rather than on the 2,560-token slot R13's pool arithmetic gives the RLM
    #: arm. A plain `ServerConfig`: slot discipline belongs to the pool the
    #: scaffold allocates from, and this profile serves two sequential calls
    #: per block on fixed slots 0 and 1, so a `slot_policy` here would
    #: advertise a knob nothing reads. `None` on a pre-S4 config.
    bench_leaf: ServerConfig | None = None


# --------------------------------------------------------------------------- #
# scaffold.*
# --------------------------------------------------------------------------- #


class ChunkCfg(_Strict):
    """C2's window geometry (spec §5 C2, §7 #2).

    `stride_tokens` is the distance between one window's end and the next
    one's. `None` means "same as `size_tokens`" -- a partition, today's
    behaviour -- so a config that never mentions stride keeps it. Below
    `size_tokens` the windows overlap, which is what puts every token within
    `stride` tokens of the end of some window containing it: the measured
    retrieval cliff is at ~1,000 tokens of needle-to-question DISTANCE
    (38/39 correct inside, 0/39 outside), not at any chunk size.
    """

    size_tokens: int
    overhead_tokens: int
    snap_to_boundary: bool = True
    snap_tolerance: float = 0.10
    stride_tokens: int | None = None

    @property
    def stride(self) -> int:
        return self.size_tokens if self.stride_tokens is None else self.stride_tokens


class MaxPredict(_Strict):
    root: int
    leaf: int


class Budgets(_Strict):
    max_depth: int = 1
    #: 522 = 261 windows x 2 questions, one full pass over a 200K-token corpus
    #: at §7 #2's window 1,024 / stride 768 geometry. The old default of 32
    #: covered 24,832 tokens -- 12.4% of such a corpus -- and coverage broke
    #: silently rather than loudly (`milestones/s2/R13-mitigations.md` §8.3).
    max_subcalls: int = 522
    max_wall_clock_s: int = 900
    #: The DELEGATION arm's wall clock, when it must differ from the shared one.
    #: `None` (every arm but `rlm-restricted`, and every pre-2026-08-20 config)
    #: means "use max_wall_clock_s", so this is inert unless declared.
    #:
    #: It exists because `max_wall_clock_s` was derived for a call volume this
    #: arm exceeds BY CONSTRUCTION. §8's 1300 s comes from
    #: `milestones/s2/aggregation_options.py`: 604 sub-calls x 2.78 s = 1,199 s plus 8.4%.
    #: An arm that must delegate for every READ, not only for every question,
    #: sits at that ceiling on arrival -- measured, synth-01 completed 629 leaf
    #: calls and was killed still running.
    #:
    #: A DIFFERENT BUDGET IS A DIFFERENT MEASUREMENT, and it must be stated
    #: beside every margin this arm appears in: its episodes are not scored
    #: under the same kill threshold as rlm/b1/b2/b3.
    restricted_max_wall_clock_s: int | None = None
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


class Retries(_Strict):
    max_attempts: int = 3
    backoff_s: list[float] = [1, 4]
    per_call_timeout_s: int = 240


class SamplingParams(_Strict):
    temperature: float
    top_p: float
    seed: int


class SamplingCfg(_Strict):
    root: SamplingParams
    leaf: SamplingParams


class PromptRef(_Strict):
    path: Path
    sha256: str | None = None


class StrategyTemplates(_Strict):
    needle: PromptRef
    aggregation: PromptRef
    synthesis: PromptRef
    code_qa: PromptRef
    default: PromptRef


#: The four §8 baseline prompt slots. A `Literal` rather than a bare `str` so
#: an arm runner asking for a name that was never authored fails at the type
#: checker and at `render_baseline`, not silently at scoring time.
BaselineName = Literal["b1_single_shot", "b2_leaf_summary",
                       "b2_root_final", "b3_single_shot"]

BASELINE_NAMES: tuple[BaselineName, ...] = (
    "b1_single_shot", "b2_leaf_summary", "b2_root_final", "b3_single_shot")


class BaselinePromptsCfg(_Strict):
    """§8's baseline arms' prompts (S4), pinned like every other registry file.

    They are pre-registered: authored once, hashed at commit, and never
    iterated against benchmark content. §8's whole comparison is RLM against
    these three arms, so a prompt tuned until B1 looked weak would make the
    result an artefact of the tuning. The pin is what makes that auditable --
    `config_snapshot` records the bytes each arm actually ran.
    """

    b1_single_shot: PromptRef
    b2_leaf_summary: PromptRef
    b2_root_final: PromptRef
    b3_single_shot: PromptRef


class PromptsCfg(_Strict):
    root: PromptRef
    leaf_prefix: PromptRef
    #: The envelope's format instructions, APPENDED to whichever leaf prefix is
    #: pinned rather than baked into one (spec §5, the S2 leaf-envelope A/B).
    #: Its own file and its own sha256 so prefix and envelope vary
    #: independently: that is what makes `leaf-prefix.v1.md` -- the control
    #: behind every leaf measurement recorded so far -- usable as an arm of the
    #: A/B without being edited. `null` when the envelope is not in use.
    leaf_envelope: PromptRef | None = None
    #: The `rlm-restricted` arm's root prompt (the delegation arm). `null`
    #: everywhere that arm is not run, including every pre-2026-08-20 snapshot.
    root_restricted: PromptRef | None = None
    strategy_templates: StrategyTemplates
    #: `None` on a pre-S4 config (and in a pre-S4 `config_snapshot`, which must
    #: keep replaying). Every slot declared here must ALSO appear in
    #: `Config._prompt_refs`, `Config.prompt_registry` and `cli.episode_config`'s
    #: rebuild -- a slot the registry loads but the rebuild does not is
    #: `PromptDrift` on replay of every episode that recorded it.
    baselines: BaselinePromptsCfg | None = None


class LeafEnvelope(_Strict):
    """The S2 leaf-envelope A/B's two switches (spec §5, §10 R5).

    `enabled` turns on the whole mechanism: the format block is appended to the
    leaf prefix, and C4 parses and validates the reply scaffold-side.

    `grammar` is the SEPARATE, optional, server-side enforcement flag, and it is
    off by default and never trusted even when on: llama.cpp has documented
    silent fail-open on schema-parse failure (§13), so a server that accepted a
    grammar is not evidence that one was applied. Scaffold-side validation runs
    identically either way -- `grammar` can only change what the model emits,
    never what the scaffold believes about it.
    """

    enabled: bool = False
    grammar: bool = False


class SandboxCfg(_Strict):
    interpreter: Path
    bootstrap_dir: Path
    network_isolation: Literal["appcontainer", "audit_only"] = "appcontainer"
    appcontainer_per_episode: bool = True
    deny_ctypes: bool = False
    memory_limit_mb: int = 4096
    # Kernel-level ban on the sandbox spawning helper processes (Job Object
    # JOB_OBJECT_LIMIT_ACTIVE_PROCESS). Default 1 matches winjob.Job's own
    # default and is load-bearing for the isolation design: without it,
    # nothing stops sandboxed code from shelling out to a clean interpreter
    # with no audit hook installed.
    active_process_limit: int = 1


class CellExtraction(_Strict):
    languages: list[str] = ["repl", "python", "py"]
    select: Literal["first", "last"] = "first"


class RootScaffoldCfg(_Strict):
    enable_thinking: bool = False
    window_tokens: int
    #: v0.3.16. `fixed` (pre-amendment behaviour, the default so old snapshots
    #: validate to what they ran): the same seed on every turn. `per_turn`:
    #: seed*1000 + turn, distinct per turn, reproducible from the snapshot.
    seed_schedule: Literal["fixed", "per_turn"] = "fixed"

    #: v0.3.16. How the root's own reply is stored in its conversation history.
    #: `prefix_plus_raw` (the pre-amendment rule, the default so old snapshots
    #: replay exactly): assistant_prefix(rendered) + raw -- which, because the
    #: chat template prepends its OWN think block to every past assistant turn,
    #: rendered two empty think blocks per turn. `raw`: the model's completion,
    #: with any reasoning split into `reasoning_content`; the template then
    #: renders exactly what the model saw and generated.
    history_mode: Literal["prefix_plus_raw", "raw"] = "prefix_plus_raw"


class LeafScaffoldCfg(_Strict):
    """The leaf's counterpart to `scaffold.root` (today: the thinking flag
    only — the leaf has no conversation window; it is one turn per call).

    S1 (finding F3) measured leaf replies consisting of nothing but an empty
    `<think></think>` block, and for chunk-level extraction the reasoning
    trace is pure cost — decode tokens the root never reads. Off by default,
    same as the root's."""

    enable_thinking: bool = False


class ScaffoldCfg(_Strict):
    dispatcher: Literal["real", "mock"] = "real"
    chunk: ChunkCfg
    truncation_cap_chars: int = 2000
    budgets: Budgets
    retries: Retries = Retries()
    root_window_kill_fraction: float = 0.90
    sampling: SamplingCfg
    prompts: PromptsCfg
    leaf_envelope: LeafEnvelope = LeafEnvelope()
    sandbox: SandboxCfg
    cell_extraction: CellExtraction = CellExtraction()
    root: RootScaffoldCfg
    leaf: LeafScaffoldCfg = LeafScaffoldCfg()
    dispatch_concurrency: int


# --------------------------------------------------------------------------- #
# trace / benchmark / power_sampling
# --------------------------------------------------------------------------- #


class TraceCfg(_Strict):
    db_path: Path
    blob_root: Path
    export_every_episode: bool = True


class BenchmarkCfg(_Strict):
    version: str | None = None
    seeds: list[int] = [1, 2, 3]
    #: The frozen manifest's OWN sha256 -- the one number that moves if any
    #: task, corpus hash, checker or precondition result changes. `version`
    #: names the manifest; this identifies the exact freeze, so an episode's
    #: snapshot says which task set it was scored against. `None` pre-S4.
    manifest_sha256: str | None = None
    #: §8's escalation draws, spent only on a block whose sign test lands short
    #: of significance. Disjoint from `seeds` by construction: re-running a
    #: seed already spent would report the same draw twice and inflate n
    #: without adding evidence. Declared here so escalation is a pre-registered
    #: rule rather than a decision made after seeing the result.
    escalation_seeds: list[int] = [4, 5]


class PowerSamplingCfg(_Strict):
    enabled: bool = False


# --------------------------------------------------------------------------- #
# top-level Config
# --------------------------------------------------------------------------- #


class Config(_Strict):
    servers: ServersConfig
    scaffold: ScaffoldCfg
    trace: TraceCfg
    benchmark: BenchmarkCfg = BenchmarkCfg()
    power_sampling: PowerSamplingCfg = PowerSamplingCfg()

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "Config":
        """Wrap pydantic's ValidationError so callers only ever catch ConfigError."""
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc

    @model_validator(mode="after")
    def _cross_field_validators(self) -> "Config":
        s = self.scaffold
        leaf = self.servers.leaf
        root = self.servers.root

        # A leaf slot must hold one window plus its overhead.
        #
        # This WAS `leaf.ctx == leaf.parallel * (size + overhead)`, and that
        # equality was right while chunk size was the parallelism lever (§4).
        # It is not right any more, for two independent reasons, and it is
        # relaxed rather than deleted:
        #   * §7 #2 (v0.2.5) fixed the window at 1,024 tokens -- retrieval
        #     falls off a cliff at ~1,000 tokens of needle-to-question
        #     distance, not at any chunk size -- so the equality would force
        #     either a falsified 32K window or a fictional 39,936-token
        #     "overhead" to absorb the difference;
        #   * R13's mitigation makes slot COUNT the lever (`-np 128` ->
        #     2,560 tok/slot at the same total KV budget), and §4 says that
        #     count must be MEASURED before it is pinned -- so the leaf runs
        #     deliberately over-provisioned per slot until it is.
        # What must still hold is the thing the arithmetic was protecting: a
        # prompt of one window plus overhead has to fit in a slot, or C4's
        # pre-flight rejects every call the chunker produces.
        slot_capacity = leaf.ctx // leaf.parallel
        window_budget = s.chunk.size_tokens + s.chunk.overhead_tokens
        if slot_capacity < window_budget:
            raise ValueError(
                f"servers.leaf.ctx ({leaf.ctx}) / leaf.parallel ({leaf.parallel}) "
                f"= {slot_capacity} tokens per slot, which cannot hold "
                f"chunk.size_tokens + chunk.overhead_tokens = {window_budget}"
            )

        if s.chunk.stride > s.chunk.size_tokens:
            raise ValueError(
                f"scaffold.chunk.stride_tokens ({s.chunk.stride}) must not exceed "
                f"chunk.size_tokens ({s.chunk.size_tokens}): a stride longer than "
                "the window leaves tokens in no window at all (§7 #2)"
            )
        if s.chunk.stride <= 0:
            raise ValueError("scaffold.chunk.stride_tokens must be positive")

        if root.ctx != s.root.window_tokens:
            raise ValueError(
                f"servers.root.ctx ({root.ctx}) must equal scaffold.root.window_tokens "
                f"({s.root.window_tokens})"
            )

        # DECOUPLED from `leaf.parallel` (v0.2.6). The old equality read
        # "C4 semaphore == leaf --parallel" (§5 Config schema) and was right
        # while `--parallel` meant "how many calls this server serves at once".
        # Under R13's never-reuse policy it means something else entirely: the
        # SLOT POOL, i.e. how many WINDOWS one leaf process can serve before it
        # must be rotated -- sized by the measured per-slot memory bill
        # (62.8125 MiB of recurrent state per slot, `milestones/s2/R13-slotcount.md`), not
        # by throughput. Dispatch concurrency is still a throughput lever, and
        # S0 measured aggregate prefill FLAT across slots, so the two numbers
        # answer different questions. Keeping the equality at the measured pool
        # size would put 128 leaf calls in flight at once.
        # What must still hold is the thing the equality protected: concurrency
        # may not exceed the pool, because every in-flight call holds a slot.
        if s.dispatch_concurrency < 1:
            raise ValueError(
                f"scaffold.dispatch_concurrency ({s.dispatch_concurrency}) must be "
                "at least 1")
        if s.dispatch_concurrency > leaf.parallel:
            raise ValueError(
                f"scaffold.dispatch_concurrency ({s.dispatch_concurrency}) must not "
                f"exceed servers.leaf.parallel ({leaf.parallel}): every in-flight "
                "leaf call holds one never-reused slot, so more concurrent calls "
                "than slots can only end in an exhaustion the pool was never "
                "sized for (§5 C4, R13)"
            )

        # Reservation <= slot capacity. Only the leaf role admits chunk-sized
        # prompts (a leaf call's pre-flight reservation is dominated by one
        # chunk); the root's prompt is the accumulated, truncated episode
        # view, not a raw chunk, so chunk.size_tokens is not a meaningful
        # reservation proxy for it.
        leaf_slot_capacity = leaf.ctx // leaf.parallel
        if s.budgets.max_predict.leaf + s.chunk.size_tokens > leaf_slot_capacity:
            raise ValueError(
                "scaffold.budgets.max_predict.leaf + scaffold.chunk.size_tokens "
                f"must not exceed the leaf slot capacity ({leaf_slot_capacity}); "
                f"got {s.budgets.max_predict.leaf} + {s.chunk.size_tokens}"
            )

        if (root.mtp or root.dflash) and root.parallel != 1:
            raise ValueError(
                f"servers.root.mtp/dflash=true requires servers.root.parallel "
                f"== 1 (got {root.parallel}). MTP is single-slot on this build; "
                f"DFlash2 is measured upstream to collapse to at or below "
                f"baseline decode at -np > 1 (llama.cpp#27342), so neither is "
                f"safe to run against a multi-slot root"
            )

        # `mtp` and `dflash` are DECLARATIONS; the flags that actually turn
        # speculative decoding on live in extra_flags, because
        # `serverproc.launch_argv` refuses to invent flags in code ("a flag
        # invented in code is a flag config_snapshot cannot record"). That split
        # is only safe if the two cannot disagree: a declaration that emitted
        # nothing would be a silent lie in every snapshot, and spec flags with
        # the declaration false would make §7 #4's optimization invisible to the
        # scoring query.
        #
        # BOTH methods are guarded, deliberately. When the root moved from
        # `draft-mtp` to `draft-dflash` (milestones/s2/DFLASH2.md) a check that only knew
        # the token "draft-mtp" would have gone VACUOUS -- `mtp: false` plus
        # DFlash flags satisfies it while reintroducing exactly the silent lie
        # it exists to prevent. A new `--spec-type` value must arrive here too.
        spec_tokens = [w for f in root.extra_flags for w in f.split()]
        spec = " ".join(spec_tokens)
        for field, token in (("mtp", "draft-mtp"), ("dflash", "draft-dflash")):
            declared = getattr(root, field)
            emitted = token in spec
            if declared and not emitted:
                raise ValueError(
                    f"servers.root.{field}=true but no '--spec-type ... {token}' "
                    f"in servers.root.extra_flags -- nothing would actually "
                    f"enable it at launch, and config_snapshot would record a "
                    f"run that did not happen"
                )
            if emitted and not declared:
                raise ValueError(
                    f"servers.root.extra_flags asks for '--spec-type {token}' "
                    f"but servers.root.{field} is false -- set it true so §7 #4's "
                    f"optimization is visible to the trace and to scoring"
                )

        # DFlash is the one method that needs a SECOND file on disk: the drafter
        # is a separate GGUF, not a head inside the target quant the way MTP's
        # is. `--spec-type draft-dflash` without `-md` dies at launch, and a
        # stale path dies after the 15.7 GB target has already been read. Both
        # are config errors and belong here, next to the prompt-hash pinning
        # that exists for the same reason.
        if root.dflash:
            draft: str | None = None
            for i, w in enumerate(spec_tokens[:-1]):
                if w in ("-md", "--spec-draft-model", "--model-draft"):
                    draft = spec_tokens[i + 1]
            if draft is None:
                raise ValueError(
                    "servers.root.dflash=true but servers.root.extra_flags "
                    "carries no '-md/--spec-draft-model <path>' -- DFlash needs "
                    "a separate drafter GGUF and llama-server refuses to start "
                    "without one"
                )
            if not Path(draft).exists():
                raise ValueError(
                    f"servers.root.extra_flags pins a DFlash drafter GGUF that "
                    f"does not exist: {draft}"
                )

        if s.truncation_cap_chars < MIN_MARKER_CAP:
            raise ValueError(
                f"scaffold.truncation_cap_chars ({s.truncation_cap_chars}) must be "
                f">= rlm.context.truncate.MIN_MARKER_CAP ({MIN_MARKER_CAP}); below that cap "
                "the truncator cannot emit its marker and silently degrades to "
                "head-only output"
            )

        # The envelope needs its format block, or it is on in name only: the
        # leaf would be asked for nothing in particular and C4 would reject
        # every plain-text reply it got back, turning the switch into a way to
        # fail every leaf call three times.
        if s.leaf_envelope.enabled and s.prompts.leaf_envelope is None:
            raise ValueError(
                "scaffold.leaf_envelope.enabled is true but "
                "scaffold.prompts.leaf_envelope is null: the envelope's format "
                "instructions must be a pinned registry file, appended to the "
                "leaf prefix (spec §5)"
            )

        # Every prompt path exists and its sha256 matches the pinned value
        # (when pinned) — brief's own qualifier. A no-op today (config.yaml
        # ships every sha256 as null); once Task 14 pins real hashes, a
        # drifted or deleted prompt file must be refused at load time, not
        # silently accepted (spec §5: the pin is what makes config_snapshot a
        # meaningful record of what actually ran). PromptRegistry.load() keeps
        # its own, separate check for rendering.
        for name, ref in self._prompt_refs():
            if ref.sha256 is None:
                continue
            resolved = resolve_prompt_path(ref.path)
            if not resolved.exists():
                raise ValueError(
                    f"scaffold.prompts.{name}.path ({ref.path}) is pinned "
                    f"(sha256={ref.sha256}) but does not exist"
                )
            actual = _sha256_hex(resolved.read_bytes())
            if actual != ref.sha256:
                raise ValueError(
                    f"scaffold.prompts.{name}.path ({ref.path}): sha256 mismatch "
                    f"(pinned {ref.sha256}, file is {actual})"
                )

        return self

    def _prompt_refs(self) -> list[tuple[str, PromptRef]]:
        """Every prompt slot this config declares, as (registry name, ref).

        ONE OF FOUR ENUMERATIONS that must agree on the slot set: this one,
        `PromptsCfg`, `prompt_registry()`, and `cli.episode_config`'s replay
        rebuild. A slot added here but not to the rebuild is `PromptDrift` on
        every episode recorded after it lands (see `cli.py`'s `leaf_envelope`
        note for the bug class).
        """
        prompts = self.scaffold.prompts
        envelope = ([("leaf_envelope", prompts.leaf_envelope)]
                    if prompts.leaf_envelope is not None else [])
        restricted = ([("root_restricted", prompts.root_restricted)]
                      if prompts.root_restricted is not None else [])
        baselines = ([(f"baselines.{name}", getattr(prompts.baselines, name))
                      for name in BASELINE_NAMES]
                     if prompts.baselines is not None else [])
        return [
            ("root", prompts.root),
            ("leaf_prefix", prompts.leaf_prefix),
            *envelope,
            *restricted,
            ("strategy_templates.needle", prompts.strategy_templates.needle),
            ("strategy_templates.aggregation", prompts.strategy_templates.aggregation),
            ("strategy_templates.synthesis", prompts.strategy_templates.synthesis),
            ("strategy_templates.code_qa", prompts.strategy_templates.code_qa),
            ("strategy_templates.default", prompts.strategy_templates.default),
            *baselines,
        ]

    def pinned_prompt_hashes(self) -> dict[str, str]:
        """path -> pinned sha256, for every prompt entry that IS pinned.

        Entries with `sha256: null` ("not yet pinned") are skipped.
        """
        return {
            str(ref.path): ref.sha256
            for _, ref in self._prompt_refs()
            if ref.sha256 is not None
        }

    def prompt_registry(self) -> "PromptRegistry":
        """Build (but do not `.load()`) the PromptRegistry this config declares.

        Third of the four slot enumerations (see `_prompt_refs`): what lands in
        the registry is what `registry.hashes()` records into every episode's
        snapshot, and therefore what `cli.episode_config` must rebuild.
        """
        prompts = self.scaffold.prompts
        baseline_refs = ({name: getattr(prompts.baselines, name)
                          for name in BASELINE_NAMES}
                         if prompts.baselines is not None else {})
        strategy_refs = {
            "needle": prompts.strategy_templates.needle,
            "aggregation": prompts.strategy_templates.aggregation,
            "synthesis": prompts.strategy_templates.synthesis,
            "code_qa": prompts.strategy_templates.code_qa,
            "default": prompts.strategy_templates.default,
        }
        return PromptRegistry.from_files(
            root_path=resolve_prompt_path(prompts.root.path),
            root_sha256=prompts.root.sha256,
            leaf_prefix_path=resolve_prompt_path(prompts.leaf_prefix.path),
            leaf_prefix_sha256=prompts.leaf_prefix.sha256,
            root_restricted_path=(resolve_prompt_path(prompts.root_restricted.path)
                                  if prompts.root_restricted is not None else None),
            root_restricted_sha256=(prompts.root_restricted.sha256
                                    if prompts.root_restricted is not None else None),
            leaf_envelope_path=(resolve_prompt_path(prompts.leaf_envelope.path)
                                if prompts.leaf_envelope else None),
            leaf_envelope_sha256=(prompts.leaf_envelope.sha256
                                  if prompts.leaf_envelope else None),
            strategy_paths={cat: resolve_prompt_path(ref.path) for cat, ref in strategy_refs.items()},
            strategy_sha256={
                cat: ref.sha256 for cat, ref in strategy_refs.items()
                if ref.sha256 is not None
            },
            baseline_paths={name: resolve_prompt_path(ref.path) for name, ref in baseline_refs.items()},
            baseline_sha256={
                name: ref.sha256 for name, ref in baseline_refs.items()
                if ref.sha256 is not None
            },
        )


def load_config(path: Path) -> Config:
    """Read + parse `path` as YAML and validate it into a Config.

    Any failure (missing file, bad YAML, schema/cross-field violation) is
    raised as ConfigError — callers never need to catch anything else.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse config {path} as YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")
    return Config.model_validate(raw)


# --------------------------------------------------------------------------- #
# config_snapshot (D19: scrub lone surrogates BEFORE json.dumps)
# --------------------------------------------------------------------------- #


def safe_text(x: str) -> str:
    """Scrub lone surrogates. DuckDB/json.dumps reject them; scrubbing after
    json.dumps turns them into escapes downstream readers then reject, so
    this must run on the raw string, before any serialisation."""
    return x.encode("utf-8", "backslashreplace").decode("utf-8")


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, dict):
        return {_scrub(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def config_snapshot(cfg: Config, extra: dict[str, Any]) -> dict[str, Any]:
    """The canonical JSON dump of the validated config (stable field order),
    merged with `extra` and scrubbed of lone surrogates before any json.dumps.
    """
    snap = cfg.model_dump(mode="json")
    snap.update(extra)
    return _scrub(snap)


# --------------------------------------------------------------------------- #
# Prompt registry
# --------------------------------------------------------------------------- #

# \r?\n? (not just \n?) because prompt files may ship with CRLF line endings
# (this is a Windows-native project; Path.write_text/open("a") both translate
# "\n" to os.linesep in text mode) — an unconsumed trailing \r would otherwise
# leak into the rendered body.
_CHANGELOG_RE = re.compile(r"\A<!--\s*changelog\b.*?-->\r?\n?", re.DOTALL)


def _strip_changelog(text: str) -> str:
    return _CHANGELOG_RE.sub("", text, count=1)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class PromptRegistry:
    """Root/leaf system prompts + strategy templates, loaded from files under
    prompts/ and pinned by sha256 (spec §5 "Prompt registry").

    Per the resolved gap on hash stability: `.load()` strips a leading
    `<!-- changelog ... -->` block before computing the *body* hash, but also
    records the *file* hash (both, always) — `test_registry_strips_changelog_
    header_but_hashes_both` pins this.

    Pinning is tolerant of "not yet pinned": `*_sha256` fields default to
    None, and the hash-match check in `.load()` only runs when a pin is
    actually present. Existence/hash-match failures raise ConfigError only
    when `.load()` is explicitly called — never as a side effect of
    constructing a Config (see the module docstring for why).
    """

    root_path: Path
    leaf_prefix_path: Path
    strategy_paths: dict[str, Path]
    root_sha256: str | None = None
    leaf_prefix_sha256: str | None = None
    strategy_sha256: dict[str, str] = field(default_factory=dict)
    #: The envelope format block, appended to the leaf prefix when the envelope
    #: is on. Its own file so the prefix arm and the envelope arm of the S2 A/B
    #: vary independently and `leaf-prefix.v1.md` never has to be edited.
    leaf_envelope_path: Path | None = None
    leaf_envelope_sha256: str | None = None
    #: The `rlm-restricted` arm's root prompt. Optional, and absent on every
    #: pre-delegation-arm config and snapshot, which `cli.episode_config`
    #: rebuilds this registry from -- replay of an older episode must keep
    #: working without it.
    root_restricted_path: Path | None = None
    root_restricted_sha256: str | None = None
    #: §8's baseline-arm prompts (S4), keyed by `BaselineName`. Empty on a
    #: pre-S4 config -- and on a pre-S4 SNAPSHOT, which `cli.episode_config`
    #: rebuilds this registry from, so replay of an old episode must keep
    #: working with no baselines at all.
    baseline_paths: dict[str, Path] = field(default_factory=dict)
    baseline_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._root_body: str | None = None
        self._root_restricted_body: str | None = None
        self._leaf_prefix_body: str | None = None
        self._leaf_envelope_body: str | None = None
        self._strategy_bodies: dict[str, str] = {}
        self._baseline_bodies: dict[str, str] = {}
        self._hashes: dict[str, str] = {}
        self._loaded = False

    @classmethod
    def from_files(
        cls,
        *,
        root_path: Path,
        leaf_prefix_path: Path,
        strategy_paths: dict[str, Path],
        root_sha256: str | None = None,
        leaf_prefix_sha256: str | None = None,
        strategy_sha256: dict[str, str] | None = None,
        leaf_envelope_path: Path | None = None,
        leaf_envelope_sha256: str | None = None,
        root_restricted_path: Path | None = None,
        root_restricted_sha256: str | None = None,
        baseline_paths: dict[str, Path] | None = None,
        baseline_sha256: dict[str, str] | None = None,
    ) -> "PromptRegistry":
        return cls(
            root_path=root_path,
            leaf_prefix_path=leaf_prefix_path,
            strategy_paths=dict(strategy_paths),
            root_sha256=root_sha256,
            leaf_prefix_sha256=leaf_prefix_sha256,
            strategy_sha256=dict(strategy_sha256 or {}),
            leaf_envelope_path=leaf_envelope_path,
            leaf_envelope_sha256=leaf_envelope_sha256,
            root_restricted_path=root_restricted_path,
            root_restricted_sha256=root_restricted_sha256,
            baseline_paths=dict(baseline_paths or {}),
            baseline_sha256=dict(baseline_sha256 or {}),
        )

    def _load_one(self, name: str, path: Path, pinned: str | None) -> str:
        # Resolution lives HERE, at the one place every prompt is actually opened,
        # rather than at each call site. Three separate callers were found bypassing
        # a call-site version on 2026-09-01 -- `Config.prompt_registry`,
        # `trace/replay.py`, and a test building a registry by hand -- which is what
        # a convention enforced in N places always gets you. `resolve_prompt_path`
        # is idempotent, so a caller that already resolved loses nothing.
        path = resolve_prompt_path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ConfigError(f"prompt {name!r} at {path} could not be read: {exc}") from exc
        file_hash = _sha256_hex(data)
        if pinned is not None and file_hash != pinned:
            raise ConfigError(
                f"prompt {name!r} at {path}: sha256 mismatch "
                f"(pinned {pinned}, file is {file_hash})"
            )
        text = data.decode("utf-8")
        body = _strip_changelog(text)
        body_hash = _sha256_hex(body.encode("utf-8"))
        self._hashes[f"{name}.file"] = file_hash
        self._hashes[f"{name}.body"] = body_hash
        return body

    def load(self) -> "PromptRegistry":
        self._root_body = self._load_one("root", self.root_path, self.root_sha256)
        self._leaf_prefix_body = self._load_one(
            "leaf_prefix", self.leaf_prefix_path, self.leaf_prefix_sha256
        )
        if self.leaf_envelope_path is not None:
            self._leaf_envelope_body = self._load_one(
                "leaf_envelope", self.leaf_envelope_path, self.leaf_envelope_sha256
            )
        if self.root_restricted_path is not None:
            self._root_restricted_body = self._load_one(
                "root_restricted", self.root_restricted_path,
                self.root_restricted_sha256
            )
        for category, path in self.strategy_paths.items():
            pinned = self.strategy_sha256.get(category)
            self._strategy_bodies[category] = self._load_one(
                f"strategy.{category}", path, pinned
            )
        for name, path in self.baseline_paths.items():
            self._baseline_bodies[name] = self._load_one(
                f"baselines.{name}", path, self.baseline_sha256.get(name)
            )
        self._loaded = True
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def render_root(self, category: str, *, restricted: bool = False) -> str:
        """root body + '\\n\\n' + the strategy block for `category`.

        Raises ConfigError for an unknown category (I1: the model never
        chooses its own strategy — only a config-declared category is valid).

        `restricted` selects the `rlm-restricted` arm's root body, which is a
        SEPARATE pinned file and never an edit of the pinned one: root.v3 is the
        `rlm` arm's prompt and the S4 re-validation depends on it being
        untouched. Refused loudly when that arm asks and no restricted prompt is
        configured — silently falling back to root.v3 would run the arm on a
        prompt whose tip 2 says "a regex, a keyword scan, a count over `chunks`
        is free and exact", which in that arm raises `TypeError`. Measured
        consequence of exactly that: 83 of 149 steps in the smoke's codeqa-01
        episode were `repl_exec / rejected / no_cell_extracted` — the root
        stopped emitting code and talked until the wall clock killed it.
        """
        self._ensure_loaded()
        if category not in self._strategy_bodies:
            raise ConfigError(
                f"{category!r} is not a declared strategy category; the model "
                "never chooses its own strategy (spec §5, I1)"
            )
        body = self._root_body
        if restricted:
            if self._root_restricted_body is None:
                raise ConfigError(
                    "the restricted arm needs scaffold.prompts.root_restricted; "
                    "falling back to the pinned root prompt would tell the root "
                    "to scan chunks this arm makes unreadable"
                )
            body = self._root_restricted_body
        return f"{body}\n\n{self._strategy_bodies[category]}"

    def leaf_prefix(self) -> str:
        self._ensure_loaded()
        assert self._leaf_prefix_body is not None
        return self._leaf_prefix_body

    def leaf_envelope(self) -> str:
        self._ensure_loaded()
        if self._leaf_envelope_body is None:
            raise ConfigError(
                "no leaf envelope block is declared "
                "(scaffold.prompts.leaf_envelope is null)")
        return self._leaf_envelope_body

    def render_leaf(self, *, envelope: bool) -> str:
        """§4's byte-identical leaf head: the prefix, plus the envelope block
        when the envelope is on.

        ONE concatenation, from two pinned files, computed once per dispatcher
        and held as a constant for its lifetime -- so the head stays
        byte-identical across every call in the run, which is the property §4's
        prefix contract and R3's drift detector both rest on. Nothing volatile
        can enter it: both operands are file bytes with their changelog headers
        stripped.
        """
        if not envelope:
            return self.leaf_prefix()
        return f"{self.leaf_prefix()}\n\n{self.leaf_envelope()}"

    def render_baseline(self, name: BaselineName) -> str:
        """The pinned prompt for one §8 baseline arm (spec §8, S4).

        Returned as-is: unlike `render_root`, nothing is appended. The baselines
        are single, self-contained instruction blocks by design -- an arm whose
        prompt were assembled from parts would be measuring the assembly as much
        as the topology.

        Raises ConfigError when the slot is not configured, rather than
        KeyError-ing inside an arm runner an hour into a bench block: a config
        with no `scaffold.prompts.baselines` cannot run §8's baselines at all,
        and that is worth being told once, by name.
        """
        self._ensure_loaded()
        if name not in self._baseline_bodies:
            raise ConfigError(
                f"no baseline prompt {name!r} is declared; "
                f"scaffold.prompts.baselines must pin all of {list(BASELINE_NAMES)} "
                "before §8's baseline arms can run (spec §8)"
            )
        return self._baseline_bodies[name]

    def hashes(self) -> dict[str, str]:
        self._ensure_loaded()
        return dict(self._hashes)
