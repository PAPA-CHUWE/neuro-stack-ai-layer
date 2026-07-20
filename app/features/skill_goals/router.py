"""Skill Goals — FastAPI router."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.shared.database import get_pg_pool
from app.models.skill_goal import (
    CreateSkillGoalRequest, GoalStatus, SkillGoalResponse,
    StepStatus, doc_to_response,
)

router = APIRouter()


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
