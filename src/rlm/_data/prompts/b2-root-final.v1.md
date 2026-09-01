<!-- changelog (prompts/b2-root-final.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-16 | Initial. Authored for S4; §8 B2 reduce step.
NOTE: pre-registered. Authored once, pinned at commit, never iterated against benchmark content -- §8's comparison is RLM against these arms, so a prompt tuned until a baseline looked weak would make the result an artefact of the tuning.
NOTE: the registry loader strips this leading HTML comment before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->
You are given summaries of consecutive fragments of one document, in order,
followed by a question. Answer the question using only the summaries.
Answer with only the answer value -- no explanation, no hedging. If the
answer is a name or identifier, reproduce it exactly as it appears.
