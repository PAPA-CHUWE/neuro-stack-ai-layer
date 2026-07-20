"""CV Extractions — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class SaveExtractionBody(BaseModel):
    user_id: str
    file_name: str
    mime_type: str
    cv_text: str
    skills: list[dict]
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    validation: dict | None = None


class ExtractionResponse(BaseModel):
    id: str
    user_id: str
    file_name: str
    mime_type: str
    cv_text: str
    skills: list[dict]
    model: str | None
    prompt_tokens: int
    completion_tokens: int
    validation: dict | None
    created_at: str
