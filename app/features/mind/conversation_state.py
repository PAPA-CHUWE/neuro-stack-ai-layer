"""Mind feature — Conversation state tracking (DB-backed)."""

import logging
import uuid

from app.shared.database import get_pg_pool

logger = logging.getLogger(__name__)


async def has_introduced(conversation_id: str) -> bool:
    """Check if the assistant has already introduced itself in this conversation."""
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT has_introduced FROM mind_conversations WHERE conversation_id = $1",
                conversation_id,
            )
        if not row:
            return False
        result = row["has_introduced"]
        logger.debug("has_introduced(%s) => %s", conversation_id, result)
        return result
    except Exception as e:
        logger.warning("has_introduced check failed for conversation=%s: %s", conversation_id, e)
        return False


async def mark_introduced(conversation_id: str, tenant_id: str = "", user_id: str = ""):
    """Record that the assistant has introduced itself in this conversation."""
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO mind_conversations (conversation_id, tenant_id, user_id, has_introduced, updated_at)
                VALUES ($1, $2, $3, TRUE, NOW())
                ON CONFLICT (conversation_id) DO UPDATE SET has_introduced = TRUE, updated_at = NOW()
            """, conversation_id, tenant_id, user_id)
        logger.debug("mark_introduced(%s)", conversation_id)
    except Exception as e:
        logger.warning("mark_introduced failed for conversation=%s: %s", conversation_id, e)


async def append_message(conversation_id: str, tenant_id: str, role: str, content: str, user_id: str = ""):
    """Persist one turn (user or assistant) of a conversation. Best-effort — never raises."""
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO mind_messages (id, conversation_id, tenant_id, user_id, role, content, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, NOW())""",
                str(uuid.uuid4()), conversation_id, tenant_id, user_id, role, content,
            )
    except Exception as e:
        logger.warning("Failed to persist message for conversation=%s: %s", conversation_id, e)
