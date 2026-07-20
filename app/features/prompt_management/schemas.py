"""Prompt Template Versioning — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class PromptVersionCreate(BaseModel):
    tenant_id: str | None = None
    code: str
    name: str
    description: str | None = None
    content: str
    change_note: str | None = None
    created_by: str


class PromptVersionUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    content: str | None = None
    change_note: str | None = None
