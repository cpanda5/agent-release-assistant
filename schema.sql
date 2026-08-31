CREATE DATABASE IF NOT EXISTS agent_release CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE agent_release;

CREATE TABLE project (
    id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid                VARCHAR(36)     NOT NULL,
    name                VARCHAR(128)    NOT NULL,
    description         TEXT,
    owner_id            BIGINT UNSIGNED,

    -- Project Context
    context_summary     TEXT,
    tech_constraints    JSON,

    status              TINYINT         NOT NULL DEFAULT 1,

    created_at          DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at          DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                        ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at          DATETIME(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_uuid (uuid),
    KEY idx_owner_id (owner_id),
    KEY idx_status (status),
    KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE releases (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,

    project_id      BIGINT UNSIGNED NOT NULL,

    name            VARCHAR(128)    NOT NULL,
    description     TEXT,

    status          VARCHAR(32)     NOT NULL DEFAULT 'draft',
    -- draft → in_progress → review → done → archived

    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                    ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at      DATETIME(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_uuid (uuid),

    KEY idx_project_id (project_id),
    KEY idx_status (status),
    KEY idx_project_status (project_id, status)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE release_context (
    id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    release_id          BIGINT UNSIGNED NOT NULL,

    progress_summary    TEXT,
    pending_tasks       JSON,
    tech_constraints    JSON,
    version             INT UNSIGNED NOT NULL DEFAULT 1,

    updated_by          VARCHAR(64),
    updated_at          DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                        ON UPDATE CURRENT_TIMESTAMP(3),

    PRIMARY KEY (id),

    -- 一个 Release 只有一份当前 Context
    UNIQUE KEY uk_release_id (release_id),

    KEY idx_updated_at (updated_at)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE sessions (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,
    release_id      BIGINT UNSIGNED NOT NULL,
    user_id         VARCHAR(64),
    title           VARCHAR(256),
    status          VARCHAR(32)     NOT NULL DEFAULT 'active',
    -- active → closed

    total_tokens    INT UNSIGNED    NOT NULL DEFAULT 0,
    total_cost_usd  DECIMAL(10, 6)  NOT NULL DEFAULT 0,

    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                    ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at      DATETIME(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_uuid (uuid),

    KEY idx_release_id (release_id),
    KEY idx_release_status (release_id, status),
    KEY idx_user_id (user_id),
    KEY idx_release_user (release_id, user_id)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE message (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,
    session_id      BIGINT UNSIGNED NOT NULL,
    role            VARCHAR(16)     NOT NULL,
    -- user / assistant / system / tool
    content         MEDIUMTEXT,
    content_type    VARCHAR(32)     NOT NULL DEFAULT 'text',
    -- text / tool_call / tool_result / error

    tool_call_id    VARCHAR(64),
    llm_trace_id    VARCHAR(36),

    token_count     INT UNSIGNED,

    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_uuid (uuid),

    KEY idx_session_created (session_id, created_at),
    KEY idx_llm_trace_id (llm_trace_id),
    KEY idx_tool_call_id (tool_call_id)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE session_summary (
    id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid                VARCHAR(36)     NOT NULL,
    session_id          BIGINT UNSIGNED NOT NULL,
    release_id          BIGINT UNSIGNED NOT NULL,
    summary             TEXT            NOT NULL,
    key_decisions       JSON,
    artifacts_generated JSON,
    token_count         INT UNSIGNED,
    version             INT UNSIGNED    NOT NULL DEFAULT 1,

    created_at          DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_uuid (uuid),

    UNIQUE KEY uk_session_version (session_id, version),

    KEY idx_session_id (session_id),
    KEY idx_release_id (release_id)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE artifact (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid            VARCHAR(36)     NOT NULL,
    release_id      BIGINT UNSIGNED NOT NULL,
    session_id      BIGINT UNSIGNED,
    type            VARCHAR(32)     NOT NULL,
    -- design_doc / api_spec / code / test_case / pr_description
    title           VARCHAR(256)    NOT NULL,
    content         LONGTEXT        NOT NULL,
    source_trace    JSON,
    version         INT UNSIGNED    NOT NULL DEFAULT 1,
    parent_id       BIGINT UNSIGNED,
    status          VARCHAR(32)     NOT NULL DEFAULT 'draft',
    -- draft → confirmed → deprecated

    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                    ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at      DATETIME(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_uuid (uuid),

    KEY idx_release_id (release_id),
    KEY idx_release_type (release_id, type),
    KEY idx_session_id (session_id),
    KEY idx_parent_id (parent_id)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE llm_trace (
    id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    trace_id             VARCHAR(36)     NOT NULL,

    session_id           BIGINT UNSIGNED,
    release_id           BIGINT UNSIGNED,
    project_id           BIGINT UNSIGNED,

    -- 模型信息
    model                VARCHAR(64)     NOT NULL,
    requested_model      VARCHAR(64)     NOT NULL,
    fallback_model       VARCHAR(64),
    fallback_reason      VARCHAR(256),

    -- Prompt 信息
    prompt_template      VARCHAR(128),
    prompt_template_version VARCHAR(16),
    system_prompt_hash   VARCHAR(64),

    -- Token & 费用
    prompt_tokens        INT UNSIGNED    NOT NULL DEFAULT 0,
    completion_tokens    INT UNSIGNED    NOT NULL DEFAULT 0,
    total_tokens         INT UNSIGNED    NOT NULL DEFAULT 0,
    cost_usd             DECIMAL(10, 6)  NOT NULL DEFAULT 0,

    -- 性能
    latency_ms            INT UNSIGNED    NOT NULL DEFAULT 0,
    ttfb_ms               INT UNSIGNED,

    -- 重试
    retry_count           TINYINT UNSIGNED NOT NULL DEFAULT 0,
    retry_reasons         JSON,

    -- 结果
    finish_reason         VARCHAR(32),
    status                VARCHAR(16)     NOT NULL DEFAULT 'success',
    -- success / failed / fallback

    error_type            VARCHAR(64),
    error_message         VARCHAR(512),

    -- 请求上下文
    is_streaming          TINYINT(1)      NOT NULL DEFAULT 0,
    temperature           DECIMAL(3, 2),
    max_tokens            INT UNSIGNED,

    created_at             DATETIME(3)    NOT NULL DEFAULT CURRENT_TIMESTAMP(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_trace_id (trace_id),

    KEY idx_session_id (session_id),
    KEY idx_release_id (release_id),
    KEY idx_project_id (project_id),
    KEY idx_model (model),
    KEY idx_status (status),
    KEY idx_created_at (created_at),
    KEY idx_project_created (project_id, created_at)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE tool_call_trace (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    trace_id        VARCHAR(36)     NOT NULL,
    session_id      BIGINT UNSIGNED NOT NULL,
    release_id      BIGINT UNSIGNED,
    tool_name       VARCHAR(64)     NOT NULL,
    tool_call_id    VARCHAR(64)     NOT NULL,
    input_params    JSON            NOT NULL,
    output_content  TEXT,
    status          VARCHAR(16)     NOT NULL,
    -- success / failed / timeout / rejected
    error_type      VARCHAR(64),
    error_message   VARCHAR(512),
    retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0,
    latency_ms      INT UNSIGNED    NOT NULL DEFAULT 0,
    iteration       TINYINT UNSIGNED NOT NULL DEFAULT 1,

    created_at      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),

    PRIMARY KEY (id),

    UNIQUE KEY uk_trace_tool (trace_id, tool_call_id),

    KEY idx_session_id (session_id),
    KEY idx_tool_name (tool_name),
    KEY idx_status (status)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE user (
    id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    uuid                VARCHAR(36)     NOT NULL,
    username            VARCHAR(64)     NOT NULL,
    display_name        VARCHAR(64),
    email               VARCHAR(128),
    password_hash       VARCHAR(256)    NOT NULL,

    status              TINYINT         NOT NULL DEFAULT 1,
    -- 1: active, 0: disabled

    created_at          DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at          DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                        ON UPDATE CURRENT_TIMESTAMP(3),
    deleted_at          DATETIME(3),

    PRIMARY KEY (id),
    UNIQUE KEY uk_uuid (uuid),
    UNIQUE KEY uk_username (username),
    UNIQUE KEY uk_email (email),
    KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
