-- Agent Release Assistant — MySQL schema
-- Convention: internal BIGINT `id` PK + external `uuid` (VARCHAR(36)); child
-- tables reference parents by BIGINT id. The app exposes uuids over the API and
-- resolves uuid → id via app/agent/identity.py before writing child rows.

CREATE DATABASE IF NOT EXISTS agent_release CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE agent_release;

-- ============ project ============
CREATE TABLE project (
    id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid        VARCHAR(36)     NOT NULL,
    name        VARCHAR(128)    NOT NULL,
    description TEXT,
    status      TINYINT         NOT NULL DEFAULT 1,          -- 1=active 2=archived
    created_at  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at  DATETIME(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    KEY         idx_status (status),
    KEY         idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ releases ============
CREATE TABLE releases (
    id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid        VARCHAR(36)     NOT NULL,
    project_id  BIGINT UNSIGNED NOT NULL,
    name        VARCHAR(128)    NOT NULL,
    description TEXT,
    status      VARCHAR(32)     NOT NULL DEFAULT 'draft',    -- draft→in_progress→review→done→archived
    created_at  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at  DATETIME(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    KEY         idx_project_id (project_id),
    KEY         idx_status (status),
    KEY         idx_project_status (project_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ sessions (added user_id) ============
CREATE TABLE sessions (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,                -- 对外 session_id（应用生成的 UUID）
    release_id      BIGINT UNSIGNED NOT NULL,
    user_id         VARCHAR(64),                            -- 发起会话的用户
    title           VARCHAR(256),
    status          VARCHAR(32)     NOT NULL DEFAULT 'active',  -- active→closed
    total_tokens    INT UNSIGNED    NOT NULL DEFAULT 0,
    total_cost_usd  DECIMAL(10, 6)  NOT NULL DEFAULT 0,
    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                    ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at      DATETIME(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    KEY         idx_release_id (release_id),
    KEY         idx_release_status (release_id, status),
    KEY         idx_user_id (user_id),
    KEY         idx_release_user (release_id, user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ message (role now includes 'tool'; content nullable) ============
CREATE TABLE message (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,
    session_id      BIGINT UNSIGNED NOT NULL,                -- → sessions.id
    role            VARCHAR(16)     NOT NULL,                -- user / assistant / system / tool
    content         MEDIUMTEXT,                              -- 可空：assistant 仅含 tool_calls 时
    content_type    VARCHAR(32)     NOT NULL DEFAULT 'text', -- text / tool_call / tool_result / error
    tool_call_id    VARCHAR(64),
    llm_trace_id    VARCHAR(36),
    token_count     INT UNSIGNED,
    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    KEY         idx_session_id (session_id),
    KEY         idx_session_created (session_id, created_at),
    KEY         idx_llm_trace_id (llm_trace_id),
    KEY         idx_tool_call_id (tool_call_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ artifact ============
CREATE TABLE artifact (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,
    release_id      BIGINT UNSIGNED NOT NULL,
    session_id      BIGINT UNSIGNED,
    type            VARCHAR(32)     NOT NULL,                -- design_doc/api_spec/code/test_case/pr_description
    title           VARCHAR(256)    NOT NULL,
    content         LONGTEXT        NOT NULL,
    source_trace    JSON,
    version         INT UNSIGNED    NOT NULL DEFAULT 1,
    parent_id       BIGINT UNSIGNED,
    status          VARCHAR(32)     NOT NULL DEFAULT 'draft',  -- draft→confirmed→deprecated
    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                    ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at      DATETIME(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    KEY         idx_release_id (release_id),
    KEY         idx_release_type (release_id, type),
    KEY         idx_session_id (session_id),
    KEY         idx_parent_id (parent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ llm_trace ============
CREATE TABLE llm_trace (
    id                      BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    trace_id                VARCHAR(36)     NOT NULL,
    session_id              BIGINT UNSIGNED,
    release_id              BIGINT UNSIGNED,
    project_id              BIGINT UNSIGNED,
    model                   VARCHAR(64)     NOT NULL,
    requested_model         VARCHAR(64)     NOT NULL,
    fallback_model          VARCHAR(64),
    fallback_reason         VARCHAR(256),
    prompt_template         VARCHAR(128),
    prompt_template_version VARCHAR(16),
    system_prompt_hash      VARCHAR(64),
    prompt_tokens           INT UNSIGNED    NOT NULL DEFAULT 0,
    completion_tokens       INT UNSIGNED    NOT NULL DEFAULT 0,
    total_tokens            INT UNSIGNED    NOT NULL DEFAULT 0,
    cost_usd                DECIMAL(10, 6)  NOT NULL DEFAULT 0,
    latency_ms              INT UNSIGNED    NOT NULL DEFAULT 0,
    ttfb_ms                 INT UNSIGNED,
    retry_count             TINYINT UNSIGNED NOT NULL DEFAULT 0,
    retry_reasons           JSON,
    finish_reason           VARCHAR(32),
    status                  VARCHAR(16)     NOT NULL DEFAULT 'success',  -- success/failed/fallback
    error_type              VARCHAR(64),
    error_message           VARCHAR(512),
    is_streaming            TINYINT(1)      NOT NULL DEFAULT 0,
    temperature             DECIMAL(3, 2),
    max_tokens              INT UNSIGNED,
    created_at              DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_trace_id (trace_id),
    KEY         idx_session_id (session_id),
    KEY         idx_release_id (release_id),
    KEY         idx_project_id (project_id),
    KEY         idx_model (model),
    KEY         idx_status (status),
    KEY         idx_created_at (created_at),
    KEY         idx_project_created (project_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ session_summary ============
CREATE TABLE session_summary (
    id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid        VARCHAR(36)     NOT NULL,
    session_id  BIGINT UNSIGNED NOT NULL,
    release_id  BIGINT UNSIGNED NOT NULL,
    summary     TEXT            NOT NULL,
    key_decisions JSON,
    artifacts_generated JSON,
    token_count INT UNSIGNED,
    created_at  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    UNIQUE KEY  uk_session_id (session_id),
    KEY         idx_release_id (release_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ release_context ============
CREATE TABLE release_context (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    release_id      BIGINT UNSIGNED NOT NULL,
    progress_summary TEXT           NOT NULL,
    key_decisions   JSON,
    pending_tasks   JSON,
    tech_constraints JSON,
    last_updated_by VARCHAR(36),
    version         INT UNSIGNED    NOT NULL DEFAULT 1,
    updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_release_id (release_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ knowledge_base ============
CREATE TABLE knowledge_base (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,
    project_id      BIGINT UNSIGNED NOT NULL,
    name            VARCHAR(128)    NOT NULL,
    description     TEXT,
    embedding_model VARCHAR(64)     NOT NULL DEFAULT 'BAAI/bge-m3',
    status          TINYINT         NOT NULL DEFAULT 1,          -- 1=active 2=archived
    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                    ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at      DATETIME(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    UNIQUE KEY  uk_project_id (project_id),                      -- one KB per project
    KEY         idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ document ============
CREATE TABLE document (
    id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid              VARCHAR(36)     NOT NULL,
    knowledge_base_id BIGINT UNSIGNED NOT NULL,
    project_id        BIGINT UNSIGNED NOT NULL,
    release_id        BIGINT UNSIGNED,                           -- nullable: project-scoped docs
    scope             VARCHAR(16)     NOT NULL DEFAULT 'project', -- project / release
    filename          VARCHAR(512)    NOT NULL,
    mime_type         VARCHAR(128),
    doc_type          VARCHAR(16)     NOT NULL DEFAULT 'text',    -- text / image / audio
    minio_bucket      VARCHAR(64),
    minio_object_key  VARCHAR(512),
    file_size         BIGINT UNSIGNED NOT NULL DEFAULT 0,
    file_hash         CHAR(64),                                  -- sha256 hex, for dedup
    parse_status      VARCHAR(16)     NOT NULL DEFAULT 'pending', -- pending/parsing/parsed/failed
    parse_error       VARCHAR(1024),
    chunk_count       INT UNSIGNED    NOT NULL DEFAULT 0,
    char_count        INT UNSIGNED    NOT NULL DEFAULT 0,
    created_at        DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at        DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                                      ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at        DATETIME(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    KEY         idx_kb_id (knowledge_base_id),
    KEY         idx_project_release (project_id, release_id),
    KEY         idx_file_hash (file_hash),
    KEY         idx_parse_status (parse_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ chunk ============
CREATE TABLE chunk (
    id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid              VARCHAR(36)     NOT NULL,
    document_id       BIGINT UNSIGNED NOT NULL,
    knowledge_base_id BIGINT UNSIGNED NOT NULL,
    project_id        BIGINT UNSIGNED NOT NULL,
    release_id        BIGINT UNSIGNED,
    scope             VARCHAR(16)     NOT NULL DEFAULT 'project',
    seq               INT UNSIGNED    NOT NULL DEFAULT 0,         -- order within the document
    content           MEDIUMTEXT      NOT NULL,
    token_count       INT UNSIGNED    NOT NULL DEFAULT 0,
    qdrant_point_id   VARCHAR(36)     NOT NULL,                   -- 1:1 with a Qdrant point
    metadata          JSON,
    created_at        DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_uuid (uuid),
    KEY         idx_document_seq (document_id, seq),
    KEY         idx_qdrant_point (qdrant_point_id),
    KEY         idx_project_release (project_id, release_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============ tool_call_trace ============
CREATE TABLE tool_call_trace (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    trace_id        VARCHAR(36)     NOT NULL,                -- 与 llm_trace 同一 trace_id
    session_id      BIGINT UNSIGNED,
    release_id      BIGINT UNSIGNED,
    tool_name       VARCHAR(64)     NOT NULL,
    tool_call_id    VARCHAR(64)     NOT NULL,
    input_params    JSON            NOT NULL,
    output_content  TEXT,
    status          VARCHAR(16)     NOT NULL,                -- success/failed/timeout/rejected
    error_type      VARCHAR(64),
    error_message   VARCHAR(512),
    retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0,
    latency_ms      INT UNSIGNED    NOT NULL DEFAULT 0,
    iteration       TINYINT UNSIGNED NOT NULL DEFAULT 1,
    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY  uk_trace_tool (trace_id, tool_call_id),
    KEY         idx_session_id (session_id),
    KEY         idx_tool_name (tool_name),
    KEY         idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
