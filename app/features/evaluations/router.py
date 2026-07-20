"""AI Evaluation — FastAPI router."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.shared.database import get_pg_pool
from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.features.evaluations.schemas import EvaluationCaseCreate, EvaluationRunBody

logger = logging.getLogger(__name__)

router = APIRouter()


def _now():
    return datetime.now(timezone.utc)


@router.post("/cases")
async def create_evaluation_case(body: EvaluationCaseCreate):
    pool = await get_pg_pool()
    case_id = str(uuid.uuid4())
    now = _now()

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ai_evaluation_cases
                   (id, tenant_id, category, user_input, expected_intent,
                    expected_tool, expected_knowledge_codes, reference_answer,
                    prohibited_claims, style_requirements, status,
                    source_feedback_id, created_by, created_at, updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'draft',$11,$12,$13,$14)""",
                case_id, body.tenant_id, body.category, body.user_input,
                body.expected_intent, body.expected_tool,
                body.expected_knowledge_codes, body.reference_answer,
                body.prohibited_claims, body.style_requirements,
                body.source_feedback_id, body.created_by, now, now,
            )
    except Exception as e:
        logger.error("Failed to create evaluation case: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create evaluation case")

    return {"id": case_id, "status": "draft"}


@router.get("/cases")
async def list_evaluation_cases(
    tenant_id: str | None = None,
    category: str | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    pool = await get_pg_pool()
    conditions = []
    params: list = []
    idx = 1

    if tenant_id:
        conditions.append(f"tenant_id = ${idx}")
        params.append(tenant_id)
        idx += 1
    if category:
        conditions.append(f"category = ${idx}")
        params.append(category)
        idx += 1
    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT * FROM ai_evaluation_cases
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        count_row = await conn.fetchrow(
            f"SELECT COUNT(*) as total FROM ai_evaluation_cases WHERE {where}",
            *params[:-2],
        )

    return {
        "items": [dict(r) for r in rows],
        "total": count_row["total"] if count_row else 0,
    }


@router.post("/cases/{case_id}/approve")
async def approve_evaluation_case(case_id: str, approved_by: str):
    pool = await get_pg_pool()
    now = _now()

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id, status FROM ai_evaluation_cases WHERE id = $1", case_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Evaluation case not found")
        if existing["status"] not in ("draft", "retired"):
            raise HTTPException(status_code=400, detail=f"Cannot approve case in status: {existing['status']}")

        await conn.execute(
            """UPDATE ai_evaluation_cases
               SET status = 'approved', approved_by = $2, updated_at = $3
               WHERE id = $1""",
            case_id, approved_by, now,
        )

    return {"ok": True, "status": "approved"}


@router.delete("/cases/{case_id}")
async def delete_evaluation_case(case_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM ai_evaluation_cases WHERE id = $1", case_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Evaluation case not found")
    return {"ok": True}


@router.post("/run")
async def run_evaluation(body: EvaluationRunBody):
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        case = await conn.fetchrow(
            "SELECT * FROM ai_evaluation_cases WHERE id = $1", body.case_id,
        )
    if not case:
        raise HTTPException(status_code=404, detail="Evaluation case not found")

    system_prompt = (
        "You are NeuroStack Mind, an enterprise AI learning assistant. "
        "Answer the following question directly and concisely based on the provided context. "
        "Do not use prohibited claims. Do not add unnecessary conversational filler."
    )
    user_content = case["user_input"]
    if body.context:
        user_content = f"CONTEXT:\n\n{body.context}\n\n---\n\nQuestion: {case['user_input']}"

    start = datetime.now(timezone.utc)
    try:
        completion = await llm_provider.complete(
            CompletionRequest(
                system_prompt=system_prompt,
                user_prompt=user_content,
                temperature=0.2,
                max_tokens=1024,
            )
        )
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
    except Exception as e:
        logger.error("Evaluation run failed: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    actual_output = completion.content

    prohibited_claims = case["prohibited_claims"] or []
    prohibited_claim_triggered = any(
        claim.lower() in actual_output.lower() for claim in prohibited_claims
    )

    style_requirements = case["style_requirements"] or []
    style_violations = []
    bad_starters = ["sure!", "certainly!", "absolutely!", "of course!", "here's a clear breakdown"]
    bad_enders = ["would you like me to", "let me know", "hope this helps", "feel free to ask"]
    output_lower = actual_output.lower().strip()
    for starter in bad_starters:
        if output_lower.startswith(starter):
            style_violations.append(f"Starts with '{starter}'")
    for ender in bad_enders:
        if output_lower.endswith(ender) or ender in output_lower[-100:]:
            style_violations.append(f"Contains '{ender}'")

    style_compliance = 1.0 if not style_violations else max(0, 1.0 - 0.3 * len(style_violations))

    groundedness_score = None
    if case["reference_answer"]:
        ref_words = set(case["reference_answer"].lower().split())
        out_words = set(actual_output.lower().split())
        if ref_words:
            overlap = len(ref_words & out_words) / len(ref_words)
            groundedness_score = round(min(overlap * 1.5, 1.0), 3)

    passed = not prohibited_claim_triggered and style_compliance >= 0.7

    run_id = str(uuid.uuid4())
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ai_evaluation_runs
                   (id, tenant_id, case_id, prompt_version, model, actual_output,
                    correctness_score, groundedness_score, style_compliance,
                    hallucination_detected, prohibited_claim_triggered,
                    latency_ms, passed, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
                run_id, case["tenant_id"], body.case_id, body.prompt_version,
                body.model, actual_output, None, groundedness_score,
                style_compliance, prohibited_claim_triggered,
                prohibited_claim_triggered, latency_ms, passed,
                datetime.now(timezone.utc),
            )
    except Exception as e:
        logger.error("Failed to store evaluation run: %s", e)

    return {
        "run_id": run_id,
        "case_id": body.case_id,
        "actual_output": actual_output,
        "passed": passed,
        "prohibited_claim_triggered": prohibited_claim_triggered,
        "style_compliance": round(style_compliance, 3),
        "style_violations": style_violations,
        "groundedness_score": groundedness_score,
        "latency_ms": latency_ms,
        "model": completion.model,
        "tokens": {
            "input": completion.prompt_tokens,
            "output": completion.completion_tokens,
        },
    }


@router.post("/run-suite")
async def run_evaluation_suite(
    tenant_id: str | None = None,
    prompt_version: str | None = None,
    model: str = "mistral-large-latest",
):
    pool = await get_pg_pool()

    conditions = ["status = 'approved'"]
    params: list = []
    idx = 1
    if tenant_id:
        conditions.append(f"tenant_id = ${idx}")
        params.append(tenant_id)
        idx += 1

    where = " AND ".join(conditions)
    async with pool.acquire() as conn:
        cases = await conn.fetch(
            f"SELECT * FROM ai_evaluation_cases WHERE {where}", *params,
        )

    if not cases:
        return {"total": 0, "passed": 0, "failed": 0, "results": []}

    results = []
    passed_count = 0
    for case in cases:
        try:
            result = await run_evaluation(
                EvaluationRunBody(
                    case_id=case["id"],
                    prompt_version=prompt_version,
                    model=model,
                )
            )
            results.append(result)
            if result["passed"]:
                passed_count += 1
        except Exception as e:
            logger.error("Evaluation run failed for case %s: %s", case["id"], e)
            results.append({"case_id": case["id"], "passed": False, "error": str(e)})

    total = len(cases)
    return {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(passed_count / total * 100, 1) if total > 0 else None,
        "results": results,
    }


@router.get("/runs")
async def list_evaluation_runs(
    case_id: str | None = None,
    tenant_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    pool = await get_pg_pool()
    conditions = []
    params: list = []
    idx = 1

    if case_id:
        conditions.append(f"case_id = ${idx}")
        params.append(case_id)
        idx += 1
    if tenant_id:
        conditions.append(f"tenant_id = ${idx}")
        params.append(tenant_id)
        idx += 1

    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT * FROM ai_evaluation_runs
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return {"items": [dict(r) for r in rows]}
