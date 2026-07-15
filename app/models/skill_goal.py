from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from bson import ObjectId


class GoalStatus(str, Enum):
    active = "active"
    achieved = "achieved"
    abandoned = "abandoned"
    stale = "stale"


class StepStatus(str, Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"


class SkillAddressed(BaseModel):
    skillId: str
    levelContribution: float = 0


class StepCreate(BaseModel):
    courseId: str
    sequenceOrder: int
    skillsAddressed: list[SkillAddressed] = []


class StepDoc(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()))
    courseId: str
    sequenceOrder: int
    status: StepStatus = StepStatus.not_started
    skillsAddressed: list[SkillAddressed] = []
    completedAt: datetime | None = None


class EventDoc(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()))
    type: str
    payload: dict[str, Any] = {}
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class CreateSkillGoalRequest(BaseModel):
    userId: str
    roleName: str
    gapSnapshot: list[dict[str, Any]] = []
    sourceCvSubmissionId: str | None = None
    steps: list[StepCreate] = []


class SkillGoalResponse(BaseModel):
    id: str
    userId: str
    roleName: str
    status: GoalStatus
    gapSnapshot: list[dict[str, Any]] = []
    sourceCvSubmissionId: str | None = None
    achievedAt: datetime | None = None
    createdAt: datetime
    updatedAt: datetime
    steps: list[StepDoc] = []
    events: list[EventDoc] = []


def doc_to_response(doc: dict) -> SkillGoalResponse:
    return SkillGoalResponse(
        id=str(doc["_id"]),
        userId=doc["userId"],
        roleName=doc["roleName"],
        status=doc["status"],
        gapSnapshot=doc.get("gapSnapshot", []),
        sourceCvSubmissionId=doc.get("sourceCvSubmissionId"),
        achievedAt=doc.get("achievedAt"),
        createdAt=doc["createdAt"],
        updatedAt=doc["updatedAt"],
        steps=[
            StepDoc(
                id=s["id"],
                courseId=s["courseId"],
                sequenceOrder=s["sequenceOrder"],
                status=s["status"],
                skillsAddressed=s.get("skillsAddressed", []),
                completedAt=s.get("completedAt"),
            )
            for s in doc.get("steps", [])
        ],
        events=[
            EventDoc(
                id=e["id"],
                type=e["type"],
                payload=e.get("payload", {}),
                createdAt=e["createdAt"],
            )
            for e in doc.get("events", [])
        ],
    )
