"""AI Response Traces — FastAPI router."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.shared.database import get_pg_pool
from app.features.traces.schemas import TraceCreateBody

logger = logging.getLogger(__name__)

router = APIRouter()


def _now():
    return datetime.now(timezone.utc)


@router.post("")
async def create_trace(body: TraceCreateBody):
    pool = await get_pg_pool()
    trace_id = str(uuid.uuid4())
    now = _now()

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ai_response_traces
                   (id, tenant_id, conversation_id, message_id, user_intent,
                    provider, model, prompt_version, knowledge_collection_codes,
                    retrieved_document_ids, retrieved_chunk_ids, tool_calls,
                    tool_results_summary, latency_ms, input_token_count,
                    output_token_count, response_status, grounding_status,
                    confidence_score, citation_count, error_message, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22)""",
                trace_id, body.tenant_id, body.conversation_id, body.message_id,
                body.user_intent, body.provider, body.model, body.prompt_version,
                body.knowledge_collection_codes, body.retrieved_document_ids,
                body.retrieved_chunk_ids, body.tool_calls, body.tool_results_summary,
                body.latency_ms, body.input_token_count, body.output_token_count,
                body.response_status, body.grounding_status, body.confidence_score,
                body.citation_count, body.error_message, now,
            )
    except Exception as e:
        logger.error("Failed to store trace: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store trace")

    return {"id": trace_id}


@router.get("")
async def list_traces(
    tenant_id: str = Query(...),
    grounding_status: str | None = None,
    model: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    pool = await get_pg_pool()
    conditions = ["tenant_id = $1"]
    params: list = [tenant_id]
    idx = 2

    if grounding_status:
        conditions.append(f"grounding_status = ${idx}")
        params.append(grounding_status)
        idx += 1
    if model:
        conditions.append(f"model = ${idx}")
        params.append(model)
        idx += 1

    where = " AND ".join(conditions)
    query = f"""
        SELECT * FROM ai_response_traces
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        count_row = await conn.fetchrow(
            f"SELECT COUNT(*) as total FROM ai_response_traces WHERE {where}",
            *params[:-2],
        )

    return {
        "items": [dict(r) for r in rows],
        "total": count_row["total"] if count_row else 0,
    }


@router.get("/analytics/grounding")
async def grounding_analytics(
    tenant_id: str = Query(...),
    days: int = Query(30, ge=1, le=365),
):
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        grounding_dist = await conn.fetch(
            """SELECT grounding_status, COUNT(*) as count
               FROM ai_response_traces
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
               GROUP BY grounding_status""",
            tenant_id, days,
        )
        model_stats = await conn.fetch(
            """SELECT model, COUNT(*) as total,
                      AVG(confidence_score) as avg_confidence,
                      AVG(latency_ms) as avg_latency,
                      SUM(input_token_count) as total_input_tokens,
                      SUM(output_token_count) as total_output_tokens
               FROM ai_response_traces
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
               GROUP BY model""",
            tenant_id, days,
        )
        daily_grounding = await conn.fetch(
            """SELECT DATE(created_at) as day, grounding_status, COUNT(*) as count
               FROM ai_response_traces
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
               GROUP BY day, grounding_status
               ORDER BY day""",
            tenant_id, days,
        )

    total = sum(r["count"] for r in grounding_dist)
    grounded = sum(r["count"] for r in grounding_dist if r["grounding_status"] == "grounded")

    return {
        "period_days": days,
        "total_responses": total,
        "grounding_distribution": {r["grounding_status"]: r["count"] for r in grounding_dist},
        "grounding_rate": round(grounded / total * 100, 1) if total > 0 else None,
        "model_stats": [
            {
                "model": r["model"],
                "total": r["total"],
                "avg_confidence": round(r["avg_confidence"], 3) if r["avg_confidence"] else None,
                "avg_latency_ms": round(r["avg_latency"]) if r["avg_latency"] else None,
                "total_input_tokens": r["total_input_tokens"],
                "total_output_tokens": r["total_output_tokens"],
            }
            for r in model_stats
        ],
        "daily_grounding": [
            {"day": str(r["day"]), "status": r["grounding_status"], "count": r["count"]}
            for r in daily_grounding
        ],
    }


@router.get("/{trace_id}")
async def get_trace(trace_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM ai_response_traces WHERE id = $1", trace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Trace not found")
    return dict(row)
