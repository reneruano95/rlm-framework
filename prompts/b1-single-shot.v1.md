<!-- changelog (prompts/b1-single-shot.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-16 | Initial. Authored for S4; §8 B1 -- single shot, full context.
NOTE: pre-registered. Authored once, pinned at commit, never iterated against benchmark content -- §8's comparison is RLM against these arms, so a prompt tuned until a baseline looked weak would make the result an artefact of the tuning.
NOTE: the registry loader strips this leading HTML comment before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->
You are given a document and a question. Read the document and answer the
question. Answer with only the answer value -- no explanation, no hedging,
no restating the question. If the answer is a name or identifier, reproduce
it exactly as it appears in the document.
