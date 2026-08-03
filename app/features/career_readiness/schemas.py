"""Career Readiness Agent — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class AssessReadinessBody(BaseModel):
    role_name: str
    readiness_level: str  # READY | NEARLY_READY | NOT_YET_READY
    signals: dict
    course_titles: list[str]


class AssessReadinessResponse(BaseModel):
    assessment: str
    next_stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
