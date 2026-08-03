"""Skill Goals — planning schemas (distinct from the CRUD models in app.models.skill_goal)."""

from __future__ import annotations

from pydantic import BaseModel


class PlanFromGoalBody(BaseModel):
    goal_text: str
    current_skills: list[dict]
    available_courses: list[dict]
    max_courses: int = 5


class PlanFromGoalResponse(BaseModel):
    target_role: str
    gaps: list[dict]
    path_title: str
    path_description: str
    courses: list[dict]
    model: str
    prompt_tokens: int
    completion_tokens: int


class DiagnoseRiskBody(BaseModel):
    role_name: str
    cause: str  # STALLED_START | STALLED_PROGRESS | COMPREHENSION_GAP
    signals: dict
    step_titles: list[str]


class DiagnoseRiskResponse(BaseModel):
    diagnosis: str
    remediation: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class AssessPerformanceBody(BaseModel):
    course_title: str
    role_name: str
    score: int
    passing_score: int


class AssessPerformanceResponse(BaseModel):
    narrative: str
    model: str
    prompt_tokens: int
    completion_tokens: int
