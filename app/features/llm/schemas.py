"""LLM — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CompleteBody(BaseModel):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False


class CompleteResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
