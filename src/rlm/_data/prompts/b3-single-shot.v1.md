<!-- changelog (prompts/b3-single-shot.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-16 | Initial. Authored for S4; §8 B3 -- BM25-selected excerpts.
NOTE: pre-registered. Authored once, pinned at commit, never iterated against benchmark content -- §8's comparison is RLM against these arms, so a prompt tuned until a baseline looked weak would make the result an artefact of the tuning.
NOTE: the registry loader strips this leading HTML comment before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->
You are given excerpts from a document, in their original order, and a
question. The excerpts were selected by keyword search and may be
incomplete. Answer the question from the excerpts. Answer with only the
answer value -- no explanation, no hedging. If the answer is a name or
identifier, reproduce it exactly as it appears.
