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
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from rlm.errors import ConfigError
from rlm.truncate import MIN_MARKER_CAP


class _Strict(BaseModel):
    """Shared base: `extra="forbid"` on every nested model, not just the root."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# servers.*
# --------------------------------------------------------------------------- #


class ServerConfig(_Strict):
    model: Path
    mtp: bool = False
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


class ServersConfig(_Strict):
    root: ServerConfig
    leaf: ServerConfig
    fallback_leaf: ServerConfig | None = None


# --------------------------------------------------------------------------- #
# scaffold.*
# --------------------------------------------------------------------------- #


class ChunkCfg(_Strict):
    size_tokens: int
    overhead_tokens: int
    snap_to_boundary: bool = True
    snap_tolerance: float = 0.10


class MaxPredict(_Strict):
    root: int
    leaf: int


class Budgets(_Strict):
    max_depth: int = 1
    max_subcalls: int = 32
    max_wall_clock_s: int = 900
    max_total_tokens: int = 1_500_000
    max_predict: MaxPredict


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


class PromptsCfg(_Strict):
    root: PromptRef
    leaf_prefix: PromptRef
    strategy_templates: StrategyTemplates


class LeafEnvelope(_Strict):
    enabled: bool = False


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

        expected_leaf_ctx = leaf.parallel * (s.chunk.size_tokens + s.chunk.overhead_tokens)
        if leaf.ctx != expected_leaf_ctx:
            raise ValueError(
                f"servers.leaf.ctx ({leaf.ctx}) must equal leaf.parallel * "
                f"(chunk.size_tokens + chunk.overhead_tokens) = {expected_leaf_ctx}"
            )

        if root.ctx != s.root.window_tokens:
            raise ValueError(
                f"servers.root.ctx ({root.ctx}) must equal scaffold.root.window_tokens "
                f"({s.root.window_tokens})"
            )

        if s.dispatch_concurrency != leaf.parallel:
            raise ValueError(
                f"scaffold.dispatch_concurrency ({s.dispatch_concurrency}) must equal "
                f"servers.leaf.parallel ({leaf.parallel})"
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

        if root.mtp and root.parallel != 1:
            raise ValueError(
                f"servers.root.mtp=true requires servers.root.parallel == 1 "
                f"(got {root.parallel})"
            )

        if s.truncation_cap_chars < MIN_MARKER_CAP:
            raise ValueError(
                f"scaffold.truncation_cap_chars ({s.truncation_cap_chars}) must be "
                f">= rlm.truncate.MIN_MARKER_CAP ({MIN_MARKER_CAP}); below that cap "
                "the truncator cannot emit its marker and silently degrades to "
                "head-only output"
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
            if not ref.path.exists():
                raise ValueError(
                    f"scaffold.prompts.{name}.path ({ref.path}) is pinned "
                    f"(sha256={ref.sha256}) but does not exist"
                )
            actual = _sha256_hex(ref.path.read_bytes())
            if actual != ref.sha256:
                raise ValueError(
                    f"scaffold.prompts.{name}.path ({ref.path}): sha256 mismatch "
                    f"(pinned {ref.sha256}, file is {actual})"
                )

        return self

    def _prompt_refs(self) -> list[tuple[str, PromptRef]]:
        prompts = self.scaffold.prompts
        return [
            ("root", prompts.root),
            ("leaf_prefix", prompts.leaf_prefix),
            ("strategy_templates.needle", prompts.strategy_templates.needle),
            ("strategy_templates.aggregation", prompts.strategy_templates.aggregation),
            ("strategy_templates.synthesis", prompts.strategy_templates.synthesis),
            ("strategy_templates.code_qa", prompts.strategy_templates.code_qa),
            ("strategy_templates.default", prompts.strategy_templates.default),
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
        """Build (but do not `.load()`) the PromptRegistry this config declares."""
        prompts = self.scaffold.prompts
        strategy_refs = {
            "needle": prompts.strategy_templates.needle,
            "aggregation": prompts.strategy_templates.aggregation,
            "synthesis": prompts.strategy_templates.synthesis,
            "code_qa": prompts.strategy_templates.code_qa,
            "default": prompts.strategy_templates.default,
        }
        return PromptRegistry.from_files(
            root_path=prompts.root.path,
            root_sha256=prompts.root.sha256,
            leaf_prefix_path=prompts.leaf_prefix.path,
            leaf_prefix_sha256=prompts.leaf_prefix.sha256,
            strategy_paths={cat: ref.path for cat, ref in strategy_refs.items()},
            strategy_sha256={
                cat: ref.sha256 for cat, ref in strategy_refs.items()
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

    def __post_init__(self) -> None:
        self._root_body: str | None = None
        self._leaf_prefix_body: str | None = None
        self._strategy_bodies: dict[str, str] = {}
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
    ) -> "PromptRegistry":
        return cls(
            root_path=root_path,
            leaf_prefix_path=leaf_prefix_path,
            strategy_paths=dict(strategy_paths),
            root_sha256=root_sha256,
            leaf_prefix_sha256=leaf_prefix_sha256,
            strategy_sha256=dict(strategy_sha256 or {}),
        )

    def _load_one(self, name: str, path: Path, pinned: str | None) -> str:
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
        for category, path in self.strategy_paths.items():
            pinned = self.strategy_sha256.get(category)
            self._strategy_bodies[category] = self._load_one(
                f"strategy.{category}", path, pinned
            )
        self._loaded = True
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def render_root(self, category: str) -> str:
        """root body + '\\n\\n' + the strategy block for `category`.

        Raises ConfigError for an unknown category (I1: the model never
        chooses its own strategy — only a config-declared category is valid).
        """
        self._ensure_loaded()
        if category not in self._strategy_bodies:
            raise ConfigError(
                f"{category!r} is not a declared strategy category; the model "
                "never chooses its own strategy (spec §5, I1)"
            )
        return f"{self._root_body}\n\n{self._strategy_bodies[category]}"

    def leaf_prefix(self) -> str:
        self._ensure_loaded()
        assert self._leaf_prefix_body is not None
        return self._leaf_prefix_body

    def hashes(self) -> dict[str, str]:
        self._ensure_loaded()
        return dict(self._hashes)
