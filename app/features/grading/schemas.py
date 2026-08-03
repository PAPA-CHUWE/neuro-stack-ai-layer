"""Grading — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class AnswerGradingBody(BaseModel):
    question_text: str
    question_type: str  # "ESSAY" | "FILL_BLANK"
    correct_answer: str | None = None
    max_points: int
    learner_response: str
    course_context: str | None = None


class AnswerGradingResponse(BaseModel):
    is_correct: bool
    points_awarded: float
    explanation: str
    model: str
    prompt_tokens: int
    completion_tokens: int
