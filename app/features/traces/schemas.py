"""AI Response Traces — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class TraceCreateBody(BaseModel):
    tenant_id: str
    conversation_id: str | None = None
    message_id: str | None = None
    user_intent: str | None = None
    provider: str
    model: str
    prompt_version: str | None = None
    knowledge_collection_codes: list[str] = []
    retrieved_document_ids: list[str] = []
    retrieved_chunk_ids: list[str] = []
    tool_calls: dict | list | None = None
    tool_results_summary: str | None = None
    latency_ms: int | None = None
    input_token_count: int | None = None
    output_token_count: int | None = None
    response_status: str = "success"
    grounding_status: str = "ungrounded"
    confidence_score: float | None = None
    citation_count: int = 0
    error_message: str | None = None
