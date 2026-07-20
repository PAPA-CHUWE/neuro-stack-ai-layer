"""Streaming — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class StreamBody(BaseModel):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False
