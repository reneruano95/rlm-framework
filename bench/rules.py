"""§8's pre-registered scoring rules as DATA, read from the benchmark manifest.

Until 2026-09-02 every one of these lived as a module constant in
`src/rlm/measure/verdict.py` (MARGIN_GATE = 3, BASELINES = ("b1","b2","b3"), the
{+1,+2,+3} band, the >=3 zero-floor). v2 needs different values and a per-arm
abstention, and v1's frozen record needs the old ones forever. So the values
move here, the manifest may carry a `rules` block, and a manifest without one
(v1) gets exactly the constants it was scored under.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_KNOWN = {"rlm_arm", "baselines", "margin", "escalation_band", "tripwire_floor",
          "abstentions", "scored_stream", "n_tasks"}


@dataclass(frozen=True)
class BenchmarkRules:
    rlm_arm: str = "rlm"
    baselines: tuple[str, ...] = ("b1", "b2", "b3")
    margin: int = 3
    escalation_band: tuple[int, ...] = (1, 2, 3)
    tripwire_floor: int = 3
    abstentions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    scored_stream: str | None = None
    n_tasks: int | None = None

    @property
    def arms(self) -> tuple[str, ...]:
        return (self.rlm_arm, *self.baselines)

    def abstains(self, arm: str, category: str) -> bool:
        return category in self.abstentions.get(arm, ())

    @classmethod
    def for_manifest(cls, manifest) -> "BenchmarkRules":
        raw = getattr(manifest, "rules", None) or {}
        unknown = set(raw) - _KNOWN
        if unknown:
            raise ValueError(f"unknown rule key(s) {sorted(unknown)}; known: {sorted(_KNOWN)}")
        kw = dict(raw)
        if "baselines" in kw:
            kw["baselines"] = tuple(kw["baselines"])
        if "escalation_band" in kw:
            kw["escalation_band"] = tuple(int(x) for x in kw["escalation_band"])
        if "abstentions" in kw:
            kw["abstentions"] = {a: tuple(c) for a, c in kw["abstentions"].items()}
        return cls(**kw)


__all__ = ["BenchmarkRules"]
