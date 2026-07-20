"""Mind feature — Response trace storage."""

import logging
import uuid

from app.shared.database import get_pg_pool
from app.features.mind.schemas import SourceItem

logger = logging.getLogger(__name__)


async def store_trace(
    tenant_id: str, mode: str, question: str, model: str,
    confidence: str, sources: list[SourceItem], latency_ms: int,
    conversation_id: str | None = None,
) -> str | None:
    """Store an AI response trace. Returns trace_id."""
    try:
        pool = await get_pg_pool()
        trace_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ai_response_traces
                   (id, tenant_id, conversation_id, user_intent, provider, model,
                    grounding_status, citation_count, latency_ms, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW())""",
                trace_id, tenant_id, conversation_id, mode, "mistral", model,
                confidence, len(sources), latency_ms,
            )
        return trace_id
    except Exception as e:
        logger.warning("Failed to store trace: %s", e)
        return None
