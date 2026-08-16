<!-- changelog (prompts/b2-leaf-summary.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-16 | Initial. Authored for S4; §8 B2 map step.
NOTE: pre-registered. Authored once, pinned at commit, never iterated against benchmark content -- §8's comparison is RLM against these arms, so a prompt tuned until a baseline looked weak would make the result an artefact of the tuning.
NOTE: the registry loader strips this leading HTML comment before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->
Summarize the following document fragment. Preserve every concrete fact:
names, identifiers, numbers, statuses, dates, and relationships between
named entities. Omit style and filler. Output only the summary.
