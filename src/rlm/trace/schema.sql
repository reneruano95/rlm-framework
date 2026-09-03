-- C6 trajectory trace store (ARCHITECTURE.md v0.2.2 §6).
-- ENUM over VARCHAR+CHECK: DuckDB supports neither ADD nor DROP CONSTRAINT
-- (evolution would require a full table rebuild), while ENUM evolution is
-- CREATE TYPE v2 + ALTER TABLE ... ALTER COLUMN ... SET DATA TYPE, a
-- one-liner. ENUM also exports to parquet as plain VARCHAR, so the bundle
-- stays readable by a foreign reader with no catalog types.
--
-- `cache_hit` intentionally does NOT exist here: removed in spec v0.1,
-- replaced by steps.tokens_cached (never stored as a boolean).
--
-- NOT NULL columns that carry a DEFAULT ('', '{}'::JSON) do so only so a
-- minimal direct INSERT (e.g. crash-recovery bookkeeping) cannot violate the
-- constraint; TraceLogger.open_episode always supplies real values.
CREATE TYPE IF NOT EXISTS episode_outcome AS ENUM
    ('success','fail','budget_kill','context_exhausted','error');
CREATE TYPE IF NOT EXISTS step_actor  AS ENUM ('root','leaf');
CREATE TYPE IF NOT EXISTS step_action AS ENUM ('repl_exec','llm_call','final');
CREATE TYPE IF NOT EXISTS step_status AS ENUM ('ok','error','timeout','cancelled','rejected');

CREATE TABLE IF NOT EXISTS episodes (
    episode_id UUID PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_hash TEXT NOT NULL,
    tokenized_task_len INTEGER,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    outcome episode_outcome,
    outcome_reason TEXT,
    final_answer_ref TEXT,
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    scaffold_instance_id TEXT NOT NULL DEFAULT '',
    sandbox_pid INTEGER,
    superseded_by UUID,
    avg_power_w REAL,
    energy_j REAL,
    pkg_temp_c_start REAL,
    pkg_temp_c_end REAL,
    config_snapshot JSON NOT NULL DEFAULT '{}'::JSON,
    scaffold_git_sha TEXT NOT NULL DEFAULT '',
    benchmark_version TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    episode_id UUID NOT NULL REFERENCES episodes(episode_id),
    step_idx INTEGER NOT NULL,
    parent_step_idx INTEGER,
    call_id UUID,
    retry_idx INTEGER NOT NULL DEFAULT 0,
    depth INTEGER NOT NULL DEFAULT 0,
    actor step_actor NOT NULL,
    action_type step_action NOT NULL,
    status step_status NOT NULL,
    error_detail TEXT,
    action_payload TEXT,
    root_view_hash TEXT,
    root_request_ref TEXT,
    observation_view TEXT,
    observation_full_ref TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    tokens_cached INTEGER,
    slot_id INTEGER,
    t_dispatch TIMESTAMP,
    t_first_byte TIMESTAMP,
    t_end TIMESTAMP,
    latency_queue_ms INTEGER,
    latency_prefill_ms INTEGER,
    latency_decode_ms INTEGER,
    -- R13 (spec v0.2.6 §10): the foreign-string detector's verdict for this
    -- leaf answer. TRI-STATE, and the NULL is load-bearing: NULL = not
    -- checked (no corpus index, or the step never produced an answer), FALSE
    -- = checked and no foreign identifier found, TRUE = an identifier absent
    -- from the chunk sent and present in another chunk. FALSE is evidence,
    -- not a certificate: 138 clean calls give a 95% upper bound of 2.2%, and
    -- a 200K episode is ~848 leaf calls, so ~19 contaminated answers per
    -- episode are permitted by the evidence. Never read a column of FALSEs as
    -- "leak-free".
    leak_detected BOOLEAN,
    leak_detail TEXT,
    -- R13's rotation stamp (spec v0.2.6 §5 C4): the 1-based index of the
    -- leaf-server rotation THIS step triggered by exhausting the slot pool,
    -- NULL on every step that triggered none. A rotation is also a lifecycle
    -- event, but the S3 gate runs with that log deleted, so the trace has to
    -- carry the fact on its own -- and only the step ties the rotation to the
    -- window whose slot request could not be served.
    server_rotation INTEGER,
    PRIMARY KEY (episode_id, step_idx)
);

-- Migration for stores written before R13. The CREATE TABLE above is
-- IF NOT EXISTS, so it is a no-op against an existing steps table and the
-- new columns would never arrive -- the first INSERT after the upgrade would
-- fail against a real operator database. These ALTERs are idempotent.
ALTER TABLE steps ADD COLUMN IF NOT EXISTS leak_detected BOOLEAN;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS leak_detail TEXT;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS server_rotation INTEGER;

-- v2 (2026-09-02): the interactive category's `env` verb is a fourth action.
-- ENUM evolution per the header: new type, retype the column, idempotent --
-- DuckDB accepts ALTER COLUMN ... SET DATA TYPE to a column's own current
-- type as a no-op, so re-running this against an already-migrated store
-- (schema.sql is re-applied in full on every TraceLogger.start(), same as
-- the R13 ALTERs above) does not error. Verified by
-- checks/test_trace.py::test_opening_an_already_migrated_store_a_second_time_does_not_error
-- (a row is inserted and read back after the SECOND open, on a store that
-- was pre-existing before env_call), and the migrated-from-v1 path itself
-- by test_opening_a_v1_store_migrates_the_action_enum.
CREATE TYPE IF NOT EXISTS step_action_v2 AS ENUM ('repl_exec','llm_call','final','env_call');
ALTER TABLE steps ALTER COLUMN action_type SET DATA TYPE step_action_v2;
