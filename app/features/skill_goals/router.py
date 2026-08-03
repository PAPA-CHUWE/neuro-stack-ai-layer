"""Skill Goals — FastAPI router."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.shared.database import get_pg_pool
from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.utils.json_parse import extract_json
from app.models.skill_goal import (
    CreateSkillGoalRequest, GoalStatus, SkillGoalResponse,
    StepStatus, doc_to_response,
)
from app.features.skill_goals.schemas import (
    PlanFromGoalBody, PlanFromGoalResponse, DiagnoseRiskBody, DiagnoseRiskResponse,
    AssessPerformanceBody, AssessPerformanceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


PLAN_FROM_GOAL_SYSTEM_PROMPT = """You are a learning-path planning agent for an LMS platform.

Given a learner's stated career goal in their own words, their current skills, and a catalog of
available courses, produce a structured learning plan.

Rules:
1. Infer a concise target role name from the stated goal (e.g. "AI Engineer", "Data Analyst").
2. Identify the skill gaps between the learner's current skills and what the target role
   plausibly requires. Estimate currentLevel and requiredLevel (Beginner/Intermediate/Advanced)
   for each gap skill.
3. Design a course sequence from the available courses that closes those gaps, ordered
   foundational to advanced. Only use courseIds that appear in the provided catalog — never
   invent one.
4. Each course must include a brief reason tied to a specific gap skill.
5. Return at most max_courses courses.
6. Return ONLY valid JSON. No markdown, no conversational text.

Output format:
{
  "targetRole": "Role Name",
  "gaps": [
    {"skill": "Skill Name", "currentLevel": "Beginner", "requiredLevel": "Intermediate"}
  ],
  "pathTitle": "Learning Path for Role Name",
  "pathDescription": "A personalized path to close your skill gaps",
  "courses": [
    {"courseId": "id", "courseTitle": "Title", "sortOrder": 1, "reason": "Addresses gap in X", "gapSkill": "Skill Name"}
  ]
}"""


@router.post("/plan-from-goal", response_model=PlanFromGoalResponse)
async def plan_from_goal(body: PlanFromGoalBody):
    skills_text = "\n".join(
        f"- {s.get('skill', 'Unknown')}: {s.get('proficiencyLevel', 'Beginner')} (source: {s.get('source', 'unknown')})"
        for s in body.current_skills
    ) or "- (no current skill signal available)"
    courses_text = "\n".join(
        f"- {c.get('courseId', 'unknown')}: {c.get('title', 'Untitled')} — covers: {', '.join(c.get('skills', []))}"
        for c in body.available_courses
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=PLAN_FROM_GOAL_SYSTEM_PROMPT,
                user_prompt=(
                    f"Stated goal: {body.goal_text}\n\nCurrent skills:\n{skills_text}\n\n"
                    f"Available courses:\n{courses_text}\n\n"
                    f"Return at most {body.max_courses} courses."
                ),
                temperature=0.2, max_tokens=1536, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse plan-from-goal response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return PlanFromGoalResponse(
        target_role=data.get("targetRole", "General learner"),
        gaps=data.get("gaps", []),
        path_title=data.get("pathTitle", "Learning Path"),
        path_description=data.get("pathDescription", ""),
        courses=data.get("courses", []),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


DIAGNOSE_RISK_SYSTEM_PROMPT = """You are a risk-intervention agent for an LMS platform.

The CAUSE has already been classified deterministically from real progress data —
never second-guess or restate it as uncertain. Your job is only to explain it
clearly to the learner and propose one concrete remediation step.

Rules:
1. Write a short (2-3 sentence) diagnosis addressed to the learner, explaining
   why they've been flagged, grounded in the specific signals provided.
2. Write a short (2-3 sentence) remediation recommendation: one concrete next
   action (e.g. revisit a specific course, adjust pace, a short refresher).
3. Be direct and encouraging, not alarming — this is a nudge, not a warning.
4. Return ONLY valid JSON. No markdown, no conversational text.

Output format:
{
  "diagnosis": "...",
  "remediation": "..."
}"""


@router.post("/diagnose-risk", response_model=DiagnoseRiskResponse)
async def diagnose_risk(body: DiagnoseRiskBody):
    user_prompt = (
        f"Target role: {body.role_name}\n"
        f"Classified cause: {body.cause}\n"
        f"Signals: {json.dumps(body.signals)}\n"
        f"Plan courses: {', '.join(body.step_titles) or '(none)'}\n\n"
        "Write the diagnosis and remediation."
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=DIAGNOSE_RISK_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3, max_tokens=512, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse diagnose-risk response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return DiagnoseRiskResponse(
        diagnosis=data.get("diagnosis", ""),
        remediation=data.get("remediation", ""),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


ASSESS_PERFORMANCE_SYSTEM_PROMPT = """You are the Assessment Agent for an LMS platform.

A learner has failed an assessment tied to a certification-track milestone after
exhausting their attempts. That determination — the fail, the exhausted attempts,
and the certification-track impact — has already been made deterministically from
real assessment data. Never second-guess or restate it as uncertain. Your only job
is to write a short, factual narrative for the human reviewer who will decide
whether to adjust the learner's roadmap.

Rules:
1. Write 2-3 sentences summarizing the situation: the course/milestone, the score
   versus the passing score, and why it needs review (certification-track impact,
   so it wasn't auto-adjusted).
2. Suggest one concrete option the reviewer could consider (e.g. extra practice
   before a retake, an alternate assessment, extending the deadline) without
   deciding for them.
3. Be neutral and factual — this narrative goes to a reviewer, not the learner.
4. Return ONLY valid JSON. No markdown, no conversational text.

Output format:
{"narrative": "..."}"""


@router.post("/assess-performance", response_model=AssessPerformanceResponse)
async def assess_performance(body: AssessPerformanceBody):
    user_prompt = (
        f"Course / milestone: {body.course_title}\n"
        f"Target role: {body.role_name}\n"
        f"Score: {body.score} (passing: {body.passing_score})\n\n"
        "Write the reviewer narrative."
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=ASSESS_PERFORMANCE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3, max_tokens=384, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse assess-performance response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return AssessPerformanceResponse(
        narrative=data.get("narrative", ""),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


def _now():
    return datetime.now(timezone.utc)


def _row_to_response(row, steps, events) -> SkillGoalResponse:
    return SkillGoalResponse(
        id=row["id"],
        userId=row["user_id"],
        roleName=row["role_name"],
        status=row["status"],
        gapSnapshot=json.loads(row["gap_snapshot"]) if isinstance(row["gap_snapshot"], str) else row["gap_snapshot"],
        sourceCvSubmissionId=row["source_cv_submission_id"],
        achievedAt=row["achieved_at"],
        createdAt=row["created_at"],
        updatedAt=row["updated_at"],
        steps=steps,
        events=events,
    )


async def _get_goal(pool, goal_id: str, user_id: str):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM skill_goals WHERE id = $1 AND user_id = $2", goal_id, user_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Skill goal not found")

        step_rows = await conn.fetch(
            "SELECT * FROM skill_goal_steps WHERE goal_id = $1 ORDER BY sequence_order", goal_id,
        )
        event_rows = await conn.fetch(
            "SELECT * FROM skill_goal_events WHERE goal_id = $1 ORDER BY created_at", goal_id,
        )

    steps = [
        {
            "id": s["id"], "courseId": s["course_id"], "sequenceOrder": s["sequence_order"],
            "status": s["status"],
            "skillsAddressed": json.loads(s["skills_addressed"]) if isinstance(s["skills_addressed"], str) else s["skills_addressed"],
            "completedAt": s["completed_at"],
        }
        for s in step_rows
    ]
    events = [
        {
            "id": e["id"], "type": e["type"],
            "payload": json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"],
            "createdAt": e["created_at"],
        }
        for e in event_rows
    ]

    return _row_to_response(row, steps, events)


@router.post("", response_model=SkillGoalResponse)
async def create_skill_goal(body: CreateSkillGoalRequest):
    pool = await get_pg_pool()
    goal_id = str(uuid.uuid4())
    now = _now()

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """INSERT INTO skill_goals (id, user_id, role_name, status, gap_snapshot, source_cv_submission_id, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                goal_id, body.userId, body.roleName, GoalStatus.active.value,
                json.dumps(body.gapSnapshot), body.sourceCvSubmissionId, now, now,
            )

            for s in body.steps:
                await conn.execute(
                    """INSERT INTO skill_goal_steps (id, goal_id, course_id, sequence_order, status, skills_addressed, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    str(uuid.uuid4()), goal_id, s.courseId, s.sequenceOrder,
                    StepStatus.not_started.value, json.dumps([sa.model_dump() for sa in s.skillsAddressed]), now,
                )

            await conn.execute(
                """INSERT INTO skill_goal_events (id, goal_id, type, payload, created_at)
                   VALUES ($1, $2, $3, $4, $5)""",
                str(uuid.uuid4()), goal_id, "goal_created",
                json.dumps({"roleName": body.roleName, "stepsCount": len(body.steps)}), now,
            )

    return await _get_goal(pool, goal_id, body.userId)


@router.get("", response_model=list[SkillGoalResponse])
async def list_skill_goals(userId: str = Query(...)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM skill_goals WHERE user_id = $1 ORDER BY created_at DESC", userId,
        )
        results = []
        for row in rows:
            step_rows = await conn.fetch(
                "SELECT * FROM skill_goal_steps WHERE goal_id = $1 ORDER BY sequence_order", row["id"],
            )
            event_rows = await conn.fetch(
                "SELECT * FROM skill_goal_events WHERE goal_id = $1 ORDER BY created_at", row["id"],
            )
            steps = [
                {
                    "id": s["id"], "courseId": s["course_id"], "sequenceOrder": s["sequence_order"],
                    "status": s["status"],
                    "skillsAddressed": json.loads(s["skills_addressed"]) if isinstance(s["skills_addressed"], str) else s["skills_addressed"],
                    "completedAt": s["completed_at"],
                }
                for s in step_rows
            ]
            events = [
                {
                    "id": e["id"], "type": e["type"],
                    "payload": json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"],
                    "createdAt": e["created_at"],
                }
                for e in event_rows
            ]
            results.append(_row_to_response(row, steps, events))
    return results


@router.get("/active", response_model=SkillGoalResponse | None)
async def get_active_skill_goal(userId: str = Query(...)):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM skill_goals WHERE user_id = $1 AND status = $2",
            userId, GoalStatus.active.value,
        )
        if not row:
            return None
        step_rows = await conn.fetch(
            "SELECT * FROM skill_goal_steps WHERE goal_id = $1 ORDER BY sequence_order", row["id"],
        )
        event_rows = await conn.fetch(
            "SELECT * FROM skill_goal_events WHERE goal_id = $1 ORDER BY created_at", row["id"],
        )
    steps = [
        {
            "id": s["id"], "courseId": s["course_id"], "sequenceOrder": s["sequence_order"],
            "status": s["status"],
            "skillsAddressed": json.loads(s["skills_addressed"]) if isinstance(s["skills_addressed"], str) else s["skills_addressed"],
            "completedAt": s["completed_at"],
        }
        for s in step_rows
    ]
    events = [
        {
            "id": e["id"], "type": e["type"],
            "payload": json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"],
            "createdAt": e["created_at"],
        }
        for e in event_rows
    ]
    return _row_to_response(row, steps, events)


@router.get("/{goal_id}", response_model=SkillGoalResponse)
async def get_skill_goal(goal_id: str, userId: str = Query(...)):
    pool = await get_pg_pool()
    return await _get_goal(pool, goal_id, userId)


@router.post("/{goal_id}/complete-step/{course_id}", response_model=SkillGoalResponse)
async def complete_step(goal_id: str, course_id: str, userId: str = Query(...)):
    pool = await get_pg_pool()
    now = _now()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM skill_goals WHERE id = $1 AND user_id = $2", goal_id, userId,
        )
        if not row or row["status"] != GoalStatus.active.value:
            raise HTTPException(status_code=404, detail="Active skill goal not found")

        step = await conn.fetchrow(
            "SELECT * FROM skill_goal_steps WHERE goal_id = $1 AND course_id = $2 AND status != $3",
            goal_id, course_id, StepStatus.completed.value,
        )
        if not step:
            raise HTTPException(status_code=400, detail="Step not found or already completed")

        async with conn.transaction():
            await conn.execute(
                "UPDATE skill_goal_steps SET status = $1, completed_at = $2 WHERE id = $3",
                StepStatus.completed.value, now, step["id"],
            )
            await conn.execute(
                """INSERT INTO skill_goal_events (id, goal_id, type, payload, created_at)
                   VALUES ($1, $2, $3, $4, $5)""",
                str(uuid.uuid4()), goal_id, "step_completed",
                json.dumps({"courseId": course_id, "stepId": step["id"]}), now,
            )

            all_steps = await conn.fetch(
                "SELECT status FROM skill_goal_steps WHERE goal_id = $1", goal_id,
            )
            if all(s["status"] == StepStatus.completed.value for s in all_steps):
                await conn.execute(
                    "UPDATE skill_goals SET status = $1, achieved_at = $2, updated_at = $3 WHERE id = $4",
                    GoalStatus.achieved.value, now, now, goal_id,
                )
                await conn.execute(
                    """INSERT INTO skill_goal_events (id, goal_id, type, payload, created_at)
                       VALUES ($1, $2, $3, $4, $5)""",
                    str(uuid.uuid4()), goal_id, "goal_achieved",
                    json.dumps({"completedAt": now.isoformat()}), now,
                )
            else:
                await conn.execute(
                    "UPDATE skill_goals SET updated_at = $1 WHERE id = $2", now, goal_id,
                )

    return await _get_goal(pool, goal_id, userId)


@router.patch("/{goal_id}/abandon", response_model=SkillGoalResponse)
async def abandon_skill_goal(goal_id: str, userId: str = Query(...)):
    pool = await get_pg_pool()
    now = _now()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM skill_goals WHERE id = $1 AND user_id = $2", goal_id, userId,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Skill goal not found")

        async with conn.transaction():
            await conn.execute(
                "UPDATE skill_goals SET status = $1, updated_at = $2 WHERE id = $3",
                GoalStatus.abandoned.value, now, goal_id,
            )
            await conn.execute(
                """INSERT INTO skill_goal_events (id, goal_id, type, payload, created_at)
                   VALUES ($1, $2, $3, $4, $5)""",
                str(uuid.uuid4()), goal_id, "goal_abandoned", json.dumps({}), now,
            )

    return await _get_goal(pool, goal_id, userId)
