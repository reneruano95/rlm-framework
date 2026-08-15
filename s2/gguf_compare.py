"""S5 day-one checklist, the parts that need no GPU: compare two GGUFs on
MTP head (item 3), chat template (item 4) and tokenizer identity (item 5).

    uv run --python 3.12 --no-project s2/gguf_compare.py <incumbent> <candidate>

Reading this from metadata rather than from a running server is the point: a
template or tokenizer difference found here is found before anything is loaded,
and the sha256 of each answer is what `config_snapshot` would have to record.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from gguf_arch import read_kv

# item 3 (MTP), item 4 (template), item 5 (tokenizer), plus size/rope identity
KEYS = [
    ("architecture", "general.architecture"),
    ("block_count", "{a}.block_count"),
    ("context_length", "{a}.context_length"),
    ("embedding_length", "{a}.embedding_length"),
    ("MTP layers (item 3)", "{a}.nextn_predict_layers"),
    ("expert_count", "{a}.expert_count"),
    ("ssm.inner_size", "{a}.ssm.inner_size"),
    ("ssm.state_size", "{a}.ssm.state_size"),
    ("attn heads", "{a}.attention.head_count"),
    ("attn kv heads", "{a}.attention.head_count_kv"),
    ("rope.freq_base", "{a}.rope.freq_base"),
    ("tokenizer model (item 5)", "tokenizer.ggml.model"),
    ("tokenizer pre", "tokenizer.ggml.pre"),
    ("bos id", "tokenizer.ggml.bos_token_id"),
    ("eos id", "tokenizer.ggml.eos_token_id"),
    ("padding id", "tokenizer.ggml.padding_token_id"),
]


def digest(kv: dict, key: str) -> str:
    v = kv.get(key)
    if v is None:
        return "-"
    return hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    paths = [Path(p) for p in sys.argv[1:]]
    kvs = [read_kv(p) for p in paths]
    archs = [kv.get("general.architecture", "?") for kv in kvs]

    w = 34
    print(f"{'':<26} {paths[0].name[:w]:<{w}} {paths[1].name[:w]:<{w}}")
    print("-" * (26 + 2 * w + 2))
    for label, tmpl in KEYS:
        vals = [str(kv.get(tmpl.format(a=a), "-"))[:w]
                for kv, a in zip(kvs, archs)]
        flag = "" if vals[0] == vals[1] else "   <-- DIFFERS"
        print(f"{label:<26} {vals[0]:<{w}} {vals[1]:<{w}}{flag}")

    # Template and vocabulary are too large to print; compare by hash and size.
    print()
    for label, key in (("chat template (item 4)", "tokenizer.chat_template"),
                       ("vocab (item 5)", "tokenizer.ggml.tokens"),
                       ("merges", "tokenizer.ggml.merges")):
        a, b = (digest(kv, key) for kv in kvs)
        sizes = []
        for kv in kvs:
            v = kv.get(key)
            sizes.append(f"{len(v)} chars" if isinstance(v, str)
                         else (str(v) if v is not None else "-"))
        flag = "   <-- DIFFERS" if a != b else "   identical"
        print(f"{label:<26} {sizes[0]:<{w}} {sizes[1]:<{w}}{flag}")
        print(f"{'  sha256[:16]':<26} {a:<{w}} {b:<{w}}")

    print("\nfile sha256 (record this: a re-uploaded community quant is a "
          "DIFFERENT model, S5 checklist item 7):")
    for p in paths:
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(1 << 22):
                h.update(chunk)
        print(f"  {h.hexdigest()}  {p.name}")


if __name__ == "__main__":
    main()
