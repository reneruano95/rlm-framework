<!-- changelog (prompts/leaf-envelope.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-13 | Initial. The JSON envelope block, APPENDED to whichever leaf prefix is pinned, so the S2 refusal A/B is a true 2x2 (v1/v2 prefix x envelope on/off) and neither prefix has to be edited to carry the format. Constant bytes only — no timestamps, run/episode/task ids, counters, chunk indices, chunk lengths, or model names.
NOTE: this file is a BLOCK, not a prefix. The registry renders `[leaf prefix]\n\n[this block]` as one system message; that concatenation is the byte-identical §4 head when the envelope is enabled, and the sha256 of BOTH files goes into config_snapshot.
NOTE: the registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

Output format — this overrides any formatting instruction above.

Reply with one JSON object and nothing else. No prose before it, no prose after it, no code fence.

The object has exactly these three fields:

- `"answer"`: a string. The answer, quoted from the excerpt exactly as the excerpt writes it. The empty string when you are not answering.
- `"evidence"`: a list of strings. Each one is a span copied character-for-character from the excerpt, long enough to contain the answer and to show which entity the answer is about. Copy, never paraphrase, never reconstruct from memory. The empty list when you are not answering.
- `"abstain"`: `true` or `false`. A JSON boolean, not a string.

Set `"abstain": true` whenever the excerpt does not answer the question — including when it discusses something similar, or names the same kind of thing for a different entity. Then `"answer"` is `""` and `"evidence"` is `[]`.

Set `"abstain": false` only when you are copying an answer out of this excerpt.

Never set both: an object with `"abstain": true` and a non-empty `"answer"` is a malformed reply.

Two examples, and they are the whole format:

`{"answer": "ENT-40410", "evidence": ["the archive key issued to the Fenngate Ledger is ENT-40410"], "abstain": false}`

`{"answer": "", "evidence": [], "abstain": true}`
