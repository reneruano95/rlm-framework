<!-- changelog (prompts/leaf-envelope.v2.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-13 | Initial. The JSON envelope block, APPENDED to whichever leaf prefix is pinned, so the S2 refusal A/B is a true 2x2 (v1/v2 prefix x envelope on/off) and neither prefix has to be edited to carry the format. Constant bytes only.
v2 | 2026-08-13 | v1's override clause pointed the WRONG WAY and the block was inert. §4 fixes the layout as [system prefix][chunk][question] with the question LAST, and the sweep's questions end "Reply with the key itself and nothing else" — so the conflicting format instruction is AFTER the block, not "above" it, and the model followed the later one: a smoke run of 12 calls returned 12 bare plain-text answers and zero JSON, i.e. the envelope arm was measuring nothing at all. v2 names the question explicitly as the thing it overrides and says where the question's requested layout goes instead (inside `answer`). CORRECTED BEFORE ANY A/B DATA WAS COLLECTED — no arm has been scored against v1, which stays on disk unmodified and unpinned. This is the only change; the fields, their types and the abstain rule are byte-identical to v1.
NOTE: this file is a BLOCK, not a prefix. The registry renders `[leaf prefix]\n\n[this block]` as one system message; that concatenation is the byte-identical §4 head when the envelope is enabled, and the sha256 of BOTH files goes into config_snapshot.
NOTE: the registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

Output format. This rule outranks every other formatting instruction you are given, including the question's own — the question comes last and may tell you to reply with a bare value and nothing else. It is still describing what belongs in the `answer` field, never what the reply looks like.

Reply with one JSON object and nothing else. No prose before it, no prose after it, no code fence.

The object has exactly these three fields:

- `"answer"`: a string. The answer, quoted from the excerpt exactly as the excerpt writes it, laid out however the question asked for it. The empty string when you are not answering.
- `"evidence"`: a list of strings. Each one is a span copied character-for-character from the excerpt, long enough to contain the answer and to show which entity the answer is about. Copy, never paraphrase, never reconstruct from memory. The empty list when you are not answering.
- `"abstain"`: `true` or `false`. A JSON boolean, not a string.

Set `"abstain": true` whenever the excerpt does not answer the question — including when it discusses something similar, or names the same kind of thing for a different entity. Then `"answer"` is `""` and `"evidence"` is `[]`.

Set `"abstain": false` only when you are copying an answer out of this excerpt.

Never set both: an object with `"abstain": true` and a non-empty `"answer"` is a malformed reply.

Three examples, and they are the whole format:

`{"answer": "ENT-40410", "evidence": ["the archive key issued to the Fenngate Ledger is ENT-40410"], "abstain": false}`

`{"answer": "", "evidence": [], "abstain": true}`

A question ending "reply with the value itself and nothing else" is answered like this, not with a bare value:

`{"answer": "Marek Vantholt", "evidence": ["GIVEN . . . Marek", "FAMILY . . . Vantholt"], "abstain": false}`
