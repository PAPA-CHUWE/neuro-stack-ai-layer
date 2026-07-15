from __future__ import annotations

from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query

from app.db import get_db
from app.models.skill_goal import (
    CreateSkillGoalRequest,
    EventDoc,
    GoalStatus,
    SkillGoalResponse,
    StepDoc,
    StepStatus,
    doc_to_response,
)

router = APIRouter()


@router.post("", response_model=SkillGoalResponse)
async def create_skill_goal(body: CreateSkillGoalRequest):
    db = await get_db()
    now = datetime.utcnow()

    steps = [
        {
            "id": str(ObjectId()),
            "courseId": s.courseId,
            "sequenceOrder": s.sequenceOrder,
            "status": StepStatus.not_started.value,
            "skillsAddressed": [sa.model_dump() for sa in s.skillsAddressed],
            "completedAt": None,
        }
        for s in body.steps
    ]

    event = {
        "id": str(ObjectId()),
        "type": "goal_created",
        "payload": {
            "roleName": body.roleName,
            "stepsCount": len(body.steps),
        },
        "createdAt": now,
    }

    doc = {
        "userId": body.userId,
        "roleName": body.roleName,
        "status": GoalStatus.active.value,
        "gapSnapshot": body.gapSnapshot,
        "sourceCvSubmissionId": body.sourceCvSubmissionId,
        "achievedAt": None,
        "createdAt": now,
        "updatedAt": now,
        "steps": steps,
        "events": [event],
    }

    result = await db.skill_goals.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc_to_response(doc)


@router.get("", response_model=list[SkillGoalResponse])
async def list_skill_goals(userId: str = Query(...)):
    db = await get_db()
    cursor = db.skill_goals.find({"userId": userId}).sort("createdAt", -1)
    docs = await cursor.to_list(length=100)
    return [doc_to_response(d) for d in docs]


@router.get("/active", response_model=SkillGoalResponse | None)
async def get_active_skill_goal(userId: str = Query(...)):
    db = await get_db()
    doc = await db.skill_goals.find_one(
        {"userId": userId, "status": GoalStatus.active.value}
    )
    if not doc:
        return None
    return doc_to_response(doc)


@router.get("/{goal_id}", response_model=SkillGoalResponse)
async def get_skill_goal(goal_id: str, userId: str = Query(...)):
    db = await get_db()
    try:
        doc = await db.skill_goals.find_one({"_id": ObjectId(goal_id), "userId": userId})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid goal ID")
    if not doc:
        raise HTTPException(status_code=404, detail="Skill goal not found")
    return doc_to_response(doc)


@router.post("/{goal_id}/complete-step/{course_id}", response_model=SkillGoalResponse)
async def complete_step(goal_id: str, course_id: str, userId: str = Query(...)):
    db = await get_db()

    try:
        doc = await db.skill_goals.find_one({"_id": ObjectId(goal_id), "userId": userId})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid goal ID")

    if not doc or doc["status"] != GoalStatus.active.value:
        raise HTTPException(status_code=404, detail="Active skill goal not found")

    now = datetime.utcnow()
    updated = False

    for step in doc.get("steps", []):
        if step["courseId"] == course_id and step["status"] != StepStatus.completed.value:
            step["status"] = StepStatus.completed.value
            step["completedAt"] = now
            updated = True

            doc.setdefault("events", []).append(
                {
                    "id": str(ObjectId()),
                    "type": "step_completed",
                    "payload": {
                        "courseId": course_id,
                        "stepId": step["id"],
                        "skillsAddressed": step.get("skillsAddressed", []),
                    },
                    "createdAt": now,
                }
            )
            break

    if not updated:
        raise HTTPException(status_code=400, detail="Step not found or already completed")

    all_completed = all(
        s["status"] == StepStatus.completed.value for s in doc.get("steps", [])
    )

    if all_completed:
        doc["status"] = GoalStatus.achieved.value
        doc["achievedAt"] = now
        doc.setdefault("events", []).append(
            {
                "id": str(ObjectId()),
                "type": "goal_achieved",
                "payload": {"completedAt": now.isoformat()},
                "createdAt": now,
            }
        )

    doc["updatedAt"] = now

    await db.skill_goals.update_one({"_id": ObjectId(goal_id)}, {"$set": doc})
    return doc_to_response(doc)


@router.patch("/{goal_id}/abandon", response_model=SkillGoalResponse)
async def abandon_skill_goal(goal_id: str, userId: str = Query(...)):
    db = await get_db()

    try:
        doc = await db.skill_goals.find_one({"_id": ObjectId(goal_id), "userId": userId})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid goal ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Skill goal not found")

    now = datetime.utcnow()
    doc["status"] = GoalStatus.abandoned.value
    doc["updatedAt"] = now
    doc.setdefault("events", []).append(
        {
            "id": str(ObjectId()),
            "type": "goal_abandoned",
            "payload": {},
            "createdAt": now,
        }
    )

    await db.skill_goals.update_one({"_id": ObjectId(goal_id)}, {"$set": doc})
    return doc_to_response(doc)
