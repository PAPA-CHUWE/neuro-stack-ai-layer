"""Knowledge (RAG) — schemas."""

from __future__ import annotations

from pydantic import BaseModel


DEFAULT_COLLECTION = "knowledge"


class IngestDocumentBody(BaseModel):
    tenant_id: str
    title: str
    content: str
    collection: str = DEFAULT_COLLECTION
    metadata: dict | None = None


class IngestDocumentResponse(BaseModel):
    id: str
    title: str
    chunk_count: int
    collection: str


class SearchBody(BaseModel):
    tenant_id: str
    query: str
    collection: str = DEFAULT_COLLECTION
    top_k: int = 5


class SearchResultItem(BaseModel):
    id: str
    document_id: str
    content: str
    score: float
    metadata: dict | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query: str
    model: str
    prompt_tokens: int


class RagQueryBody(BaseModel):
    tenant_id: str
    question: str
    collection: str = DEFAULT_COLLECTION
    top_k: int = 5


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[SearchResultItem]
    model: str
    prompt_tokens: int
    completion_tokens: int


class DocumentInfo(BaseModel):
    id: str
    title: str
    collection: str
    chunk_count: int
    created_at: str
