"""AI Evaluation — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class EvaluationCaseCreate(BaseModel):
    tenant_id: str | None = None
    category: str
    user_input: str
    expected_intent: str | None = None
    expected_tool: str | None = None
    expected_knowledge_codes: list[str] = []
    reference_answer: str | None = None
    prohibited_claims: list[str] = []
    style_requirements: list[str] = []
    source_feedback_id: str | None = None
    created_by: str


class EvaluationRunBody(BaseModel):
    case_id: str
    prompt_version: str | None = None
    model: str = "mistral-large-latest"
    context: str | None = None
