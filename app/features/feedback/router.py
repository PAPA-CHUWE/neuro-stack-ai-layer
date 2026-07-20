"""Structured AI feedback — FastAPI router."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.shared.database import get_pg_pool
from app.features.feedback.schemas import (
    RATING_VALUES, REASON_CODES, TRIAGE_CATEGORIES,
    FeedbackSubmitBody, FeedbackReviewBody, FeedbackResponse,
)
from app.features.feedback.service import classify_triage, derive_severity

logger = logging.getLogger(__name__)

router = APIRouter()


def _now():
    return datetime.now(timezone.utc)


@router.post("")
async def submit_feedback(body: FeedbackSubmitBody):
    if body.rating not in RATING_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid rating: {body.rating}")

    invalid_reasons = set(body.reason_codes) - REASON_CODES
    if invalid_reasons:
        raise HTTPException(status_code=400, detail=f"Invalid reason codes: {invalid_reasons}")

    pool = await get_pg_pool()
    feedback_id = str(uuid.uuid4())
    now = _now()

    confidence = None
    if body.response_trace_id:
        async with pool.acquire() as conn:
            trace_row = await conn.fetchrow(
                "SELECT grounding_status, confidence_score FROM ai_response_traces WHERE id = $1",
                body.response_trace_id,
            )
            if trace_row:
                confidence = trace_row["grounding_status"]

    triage_category = None
    triage_confidence = 0.0
    triage_remediation = None
    priority = "low"

    if body.rating in ("not_helpful", "partially_helpful") and body.reason_codes:
        triage_category, triage_confidence, triage_remediation = classify_triage(
            body.reason_codes, confidence
        )
        priority = derive_severity(triage_category, body.reason_codes)

    status = "new"
    if priority == "critical":
        status = "triaged"

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ai_feedback
                   (id, tenant_id, user_id, conversation_id, message_id, response_trace_id,
                    rating, numeric_score, reason_codes, comment, suggested_correction,
                    status, priority, triage_category, triage_confidence,
                    triage_suggested_remediation, created_at, updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)""",
                feedback_id, body.tenant_id, body.user_id, body.conversation_id,
                body.message_id, body.response_trace_id, body.rating, body.numeric_score,
                body.reason_codes, body.comment, body.suggested_correction,
                status, priority, triage_category, triage_confidence,
                triage_remediation, now, now,
            )
    except Exception as e:
        logger.error("Failed to store feedback: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store feedback")

    return FeedbackResponse(
        id=feedback_id, status=status,
        triage_category=triage_category, priority=priority,
    )


@router.get("")
async def list_feedback(
    tenant_id: str = Query(...),
    status: str | None = None,
    rating: str | None = None,
    triage_category: str | None = None,
    priority: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    pool = await get_pg_pool()
    conditions = ["tenant_id = $1"]
    params: list = [tenant_id]
    idx = 2

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    if rating:
        conditions.append(f"rating = ${idx}")
        params.append(rating)
        idx += 1
    if triage_category:
        conditions.append(f"triage_category = ${idx}")
        params.append(triage_category)
        idx += 1
    if priority:
        conditions.append(f"priority = ${idx}")
        params.append(priority)
        idx += 1

    where = " AND ".join(conditions)
    query = f"""
        SELECT * FROM ai_feedback
        WHERE {where}
        ORDER BY
            CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        count_row = await conn.fetchrow(
            f"SELECT COUNT(*) as total FROM ai_feedback WHERE {where}",
            *params[:-2],
        )

    return {
        "items": [dict(r) for r in rows],
        "total": count_row["total"] if count_row else 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/analytics/summary")
async def feedback_analytics(
    tenant_id: str = Query(...),
    days: int = Query(30, ge=1, le=365),
):
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        rating_stats = await conn.fetch(
            """SELECT rating, COUNT(*) as count
               FROM ai_feedback
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
               GROUP BY rating""",
            tenant_id, days,
        )
        triage_stats = await conn.fetch(
            """SELECT triage_category, COUNT(*) as count, AVG(triage_confidence) as avg_confidence
               FROM ai_feedback
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
                 AND triage_category IS NOT NULL
               GROUP BY triage_category
               ORDER BY count DESC""",
            tenant_id, days,
        )
        priority_stats = await conn.fetch(
            """SELECT priority, COUNT(*) as count
               FROM ai_feedback
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
               GROUP BY priority""",
            tenant_id, days,
        )
        status_stats = await conn.fetch(
            """SELECT status, COUNT(*) as count
               FROM ai_feedback
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
               GROUP BY status""",
            tenant_id, days,
        )
        reason_stats = await conn.fetch(
            """SELECT unnest(reason_codes) as reason, COUNT(*) as count
               FROM ai_feedback
               WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '1 day' * $2
               GROUP BY reason
               ORDER BY count DESC
               LIMIT 10""",
            tenant_id, days,
        )
        unresolved_critical = await conn.fetchval(
            """SELECT COUNT(*) FROM ai_feedback
               WHERE tenant_id = $1 AND status IN ('new', 'triaged')
                 AND priority IN ('critical', 'high')""",
            tenant_id,
        )
        avg_review_time = await conn.fetchval(
            """SELECT AVG(EXTRACT(EPOCH FROM (reviewed_at - created_at)) / 3600)
               FROM ai_feedback
               WHERE tenant_id = $1 AND reviewed_at IS NOT NULL
                 AND created_at >= NOW() - INTERVAL '1 day' * $2""",
            tenant_id, days,
        )

    total_helpful = sum(r["count"] for r in rating_stats if r["rating"] == "helpful")
    total_not_helpful = sum(r["count"] for r in rating_stats if r["rating"] == "not_helpful")
    total_partial = sum(r["count"] for r in rating_stats if r["rating"] == "partially_helpful")
    total = total_helpful + total_not_helpful + total_partial

    return {
        "period_days": days,
        "total_feedback": total,
        "rating_distribution": {
            "helpful": total_helpful,
            "not_helpful": total_not_helpful,
            "partially_helpful": total_partial,
        },
        "satisfaction_rate": round(total_helpful / total * 100, 1) if total > 0 else None,
        "triage_categories": [
            {"category": r["triage_category"], "count": r["count"],
             "avg_confidence": round(r["avg_confidence"], 3) if r["avg_confidence"] else None}
            for r in triage_stats
        ],
        "priority_distribution": {r["priority"]: r["count"] for r in priority_stats},
        "status_distribution": {r["status"]: r["count"] for r in status_stats},
        "top_reasons": [{"reason": r["reason"], "count": r["count"]} for r in reason_stats],
        "unresolved_critical": unresolved_critical or 0,
        "avg_review_time_hours": round(avg_review_time, 1) if avg_review_time else None,
    }


@router.get("/{feedback_id}")
async def get_feedback(feedback_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM ai_feedback WHERE id = $1", feedback_id)
    if not row:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return dict(row)


@router.patch("/{feedback_id}/status")
async def update_feedback_status(feedback_id: str, body: FeedbackReviewBody):
    pool = await get_pg_pool()
    now = _now()

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id, status FROM ai_feedback WHERE id = $1", feedback_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Feedback not found")

        await conn.execute(
            """UPDATE ai_feedback
               SET status = $2, assigned_reviewer_id = $3, review_decision = $4,
                   review_notes = $5, linked_document_id = $6,
                   linked_prompt_version_id = $7, reviewed_at = $8, updated_at = $9
               WHERE id = $1""",
            feedback_id, "under_review", body.reviewer_id, body.decision,
            body.notes, body.linked_document_id, body.linked_prompt_version_id, now, now,
        )

    return {"ok": True, "status": "under_review"}


@router.post("/{feedback_id}/resolve")
async def resolve_feedback(feedback_id: str, body: FeedbackReviewBody):
    pool = await get_pg_pool()
    now = _now()

    valid_decisions = {"approved", "rejected", "duplicate", "resolved"}
    if body.decision not in valid_decisions:
        raise HTTPException(status_code=400, detail=f"Invalid decision: {body.decision}")

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM ai_feedback WHERE id = $1", feedback_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Feedback not found")

        new_status = "resolved" if body.decision in ("approved", "resolved") else body.decision
        await conn.execute(
            """UPDATE ai_feedback
               SET status = $2, review_decision = $3, review_notes = $4,
                   linked_document_id = $5, linked_prompt_version_id = $6,
                   reviewed_at = $7, updated_at = $8
               WHERE id = $1""",
            feedback_id, new_status, body.decision, body.notes,
            body.linked_document_id, body.linked_prompt_version_id, now, now,
        )

    return {"ok": True, "status": new_status}
