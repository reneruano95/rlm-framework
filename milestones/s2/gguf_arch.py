"""Read a GGUF file's metadata header and report its attention layout.

Pure stdlib, no GPU, no model load: the question "does this model keep every
token in every layer, or does it compress history into a fixed-size state"
is answered by the metadata keys alone. `ssm.*` keys (state size, inner size,
conv kernel) mean recurrent/linear-attention layers; a full-attention model has
none. Sliding-window attention shows up as `attention.sliding_window` or a
per-layer pattern key.

    uv run --python 3.12 --no-project milestones/s2/gguf_arch.py <file.gguf> [more.gguf...]
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

# GGUF value type ids
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
FIXED = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
         F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, fmt: str):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self) -> str:
        n = self.raw("<Q")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, t: int):
        if t in FIXED:
            return self.raw(FIXED[t])
        if t == STR:
            return self.string()
        if t == ARR:
            et = self.raw("<I")
            n = self.raw("<Q")
            # Token vocabularies are huge and irrelevant here: skip the payload
            # rather than materialising it, but keep the shape for reporting.
            if n > 64:
                if et in FIXED:
                    self.f.seek(struct.calcsize(FIXED[et]) * n, 1)
                else:
                    for _ in range(n):
                        self.value(et)
                return f"<array of {n} type{et}>"
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown gguf value type {t}")


def read_kv(path: Path) -> dict:
    with path.open("rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path.name}: not a GGUF file")
        ver = r.raw("<I")
        n_tensors = r.raw("<Q")
        n_kv = r.raw("<Q")
        kv = {"__version__": ver, "__tensors__": n_tensors}
        for _ in range(n_kv):
            k = r.string()
            kv[k] = r.value(r.raw("<I"))
        return kv


INTERESTING = ("architecture", "block_count", "context_length", "ssm.",
               "attention.", "expert", "rope.", "embedding_length",
               "attn_", "recurrent", "sliding", "layer")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in map(Path, sys.argv[1:]):
        kv = read_kv(p)
        arch = kv.get("general.architecture", "?")
        print(f"\n{'='*72}\n{p.name}\n  arch = {arch}   "
              f"gguf v{kv['__version__']}   {kv['__tensors__']} tensors")

        ssm = {k: v for k, v in kv.items() if ".ssm." in k or "recurrent" in k}
        swa = {k: v for k, v in kv.items()
               if "sliding" in k or "window" in k}
        print(f"\n  RECURRENT / STATE-SPACE KEYS : "
              f"{'NONE — no compressed history state' if not ssm else ''}")
        for k, v in sorted(ssm.items()):
            print(f"      {k} = {v}")
        print(f"  SLIDING-WINDOW KEYS          : "
              f"{'NONE — no bounded attention window' if not swa else ''}")
        for k, v in sorted(swa.items()):
            print(f"      {k} = {v}")

        print("\n  other architecture keys:")
        for k, v in sorted(kv.items()):
            if k.startswith("__") or k in ssm or k in swa:
                continue
            if any(s in k for s in INTERESTING) and "tokenizer" not in k:
                print(f"      {k} = {v}")

        verdict = ("HYBRID / RECURRENT — compresses history into a fixed state"
                   if ssm else
                   "SWA-INTERLEAVED — carries a bounded attention window"
                   if swa else
                   "UNIFORM GLOBAL ATTENTION — every layer sees every token")
        print(f"\n  >>> {verdict}")


if __name__ == "__main__":
    main()
