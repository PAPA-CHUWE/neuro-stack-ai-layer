"""Insights Agent — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class SummarizeCohortBody(BaseModel):
    metrics: dict


class SummarizeCohortResponse(BaseModel):
    headline: str
    narrative: str
    model: str
    prompt_tokens: int
    completion_tokens: int
