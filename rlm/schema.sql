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
    started_at TIMESTAMP,
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
    PRIMARY KEY (episode_id, step_idx)
);
