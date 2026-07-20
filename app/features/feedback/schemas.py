"""Structured AI feedback — schemas and constants."""

from __future__ import annotations

from pydantic import BaseModel, Field


RATING_VALUES = {"helpful", "not_helpful", "partially_helpful"}

REASON_CODES = {
    "INCORRECT", "OUTDATED", "TOO_GENERIC", "INCOMPLETE", "IRRELEVANT",
    "HALLUCINATED", "WRONG_ORGANIZATION_CONTEXT", "WRONG_TENANT_CONTEXT",
    "WRONG_POLICY", "FAILED_TO_USE_KNOWLEDGE_BASE", "FAILED_TO_USE_LIVE_DATA",
    "POOR_CITATION", "TOO_VERBOSE", "TOO_SHORT", "REPETITIVE",
    "UNPROFESSIONAL_TONE", "FORMAT_ERROR", "STREAMING_DUPLICATION",
    "TOOL_FAILURE", "OTHER",
    "ACCURATE", "CLEAR", "RELEVANT", "WELL_GROUNDED", "GOOD_CITATIONS",
    "USEFUL_RECOMMENDATION", "GOOD_PERSONALIZATION", "APPROPRIATE_TONE",
}

TRIAGE_CATEGORIES = {
    "KNOWLEDGE_GAP", "RETRIEVAL_FAILURE", "PROMPT_FAILURE", "TOOL_FAILURE",
    "DATA_ACCESS_FAILURE", "OUTDATED_DOCUMENT", "HALLUCINATION",
    "PERSONALIZATION_FAILURE", "TENANT_CONTEXT_FAILURE", "STYLE_FAILURE",
    "STREAMING_OR_UI_FAILURE", "MODEL_LIMITATION", "USER_MISUNDERSTANDING",
}

SEVERITY_MAP = {
    "HALLUCINATION": "high",
    "WRONG_TENANT_CONTEXT": "critical",
    "WRONG_ORGANIZATION_CONTEXT": "critical",
    "FAILED_TO_USE_KNOWLEDGE_BASE": "medium",
    "RETRIEVAL_FAILURE": "medium",
    "KNOWLEDGE_GAP": "medium",
    "TOOL_FAILURE": "medium",
    "DATA_ACCESS_FAILURE": "medium",
    "OUTDATED_DOCUMENT": "medium",
    "PROMPT_FAILURE": "low",
    "STYLE_FAILURE": "low",
    "STREAMING_OR_UI_FAILURE": "low",
    "MODEL_LIMITATION": "low",
    "USER_MISUNDERSTANDING": "low",
}


class FeedbackSubmitBody(BaseModel):
    tenant_id: str
    user_id: str
    conversation_id: str | None = None
    message_id: str | None = None
    response_trace_id: str | None = None
    rating: str
    numeric_score: int | None = Field(None, ge=1, le=5)
    reason_codes: list[str] = []
    comment: str | None = None
    suggested_correction: str | None = None


class FeedbackReviewBody(BaseModel):
    reviewer_id: str
    decision: str
    notes: str | None = None
    linked_document_id: str | None = None
    linked_prompt_version_id: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    status: str
    triage_category: str | None = None
    priority: str = "low"
