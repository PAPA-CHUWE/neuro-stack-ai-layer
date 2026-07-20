"""Mind feature — Pydantic schemas."""

from enum import Enum

from pydantic import BaseModel


class PageContext(BaseModel):
    route: str | None = None
    title: str | None = None
    selectedEntity: dict | None = None
    userRole: str | None = None
    tenantId: str | None = None


class MindQueryBody(BaseModel):
    tenant_id: str
    question: str
    collection: str = "knowledge"
    top_k: int = 5
    user_context: str | None = None
    category: str | None = None
    user_name: str | None = None
    conversation_id: str | None = None
    pageContext: PageContext | None = None


class MindStreamBody(BaseModel):
    tenant_id: str
    question: str
    collection: str = "knowledge"
    top_k: int = 5
    user_context: str | None = None
    category: str | None = None
    user_name: str | None = None
    conversation_id: str | None = None
    pageContext: PageContext | None = None


class SourceItem(BaseModel):
    id: str
    title: str
    excerpt: str
    relevance_score: float


class Confidence(str, Enum):
    grounded = "grounded"
    partial = "partial"
    ungrounded = "ungrounded"


class MindMode(str, Enum):
    conversation = "conversation"
    knowledge = "knowledge"
    platform = "platform"
    reasoning = "reasoning"
    action = "action"


class MindResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: Confidence
    needs_review: bool
    model: str
    mode: MindMode = MindMode.knowledge
    trace_id: str | None = None
    actions: list[dict] | None = None
