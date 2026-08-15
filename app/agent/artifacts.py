"""
Artifact persistence — stores Agent deliverables (design docs, API specs, code,
task lists) in the `artifact` table.

An artifact is the *product* of a conversation (as opposed to `message`, which is
the conversation itself). Each row records:
  * provenance   — `source_trace` JSON: which session / trace produced it
  * versioning   — `parent_id` + `version` chain revisions
  * lifecycle    — `status` (draft → confirmed → deprecated)

The session UUID is resolved to its BIGINT `sessions.id` via ``identity`` so the
foreign key stays consistent with the rest of the schema.
"""
import json
import logging
import uuid as _uuid

from app.agent.identity import resolve_session_pk
from app.db import get_db_pool

logger = logging.getLogger(__name__)


async def create_artifact(
    *,
    release_id: int,
    session_uuid: str | None,
    artifact_type: str,
    title: str,
    content: str,
    source_trace: dict | None = None,
    parent_uuid: str | None = None,
) -> dict | None:
    """
    Insert one artifact row. Returns ``{"artifact_id": <uuid>, "version": n}`` on
    success, or ``None`` if there is no DB pool (persistence skipped).

    If ``parent_uuid`` is given, the new row links to it and bumps ``version``
    (parent.version + 1), forming a revision chain.
    """
    pool = get_db_pool()
    if pool is None:
        return None

    session_pk = await resolve_session_pk(session_uuid)
    new_uuid = str(_uuid.uuid4())
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                version = 1
                parent_pk = None
                if parent_uuid:
                    await cur.execute(
                        "SELECT id, version FROM artifact WHERE uuid = %s", (parent_uuid,)
                    )
                    row = await cur.fetchone()
                    if row:
                        parent_pk, parent_version = row[0], row[1]
                        version = parent_version + 1

                await cur.execute(
                    """
                    INSERT INTO artifact
                        (uuid, release_id, session_id, type, title, content,
                         source_trace, version, parent_id, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft')
                    """,
                    (
                        new_uuid, release_id, session_pk, artifact_type, title, content,
                        json.dumps(source_trace or {}, ensure_ascii=False),
                        version, parent_pk,
                    ),
                )
                await conn.commit()
        return {"artifact_id": new_uuid, "version": version}
    except Exception:
        logger.exception("Failed to create artifact (type=%s, title=%s)", artifact_type, title)
        return None
