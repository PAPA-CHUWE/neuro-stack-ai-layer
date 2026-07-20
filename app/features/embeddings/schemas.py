"""Embeddings — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class EmbedBody(BaseModel):
    input: str
    model: str | None = None


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    prompt_tokens: int


class SearchBody(BaseModel):
    tenant_id: str
    config: dict
    query_vector: list[float]
    top_k: int = 10
    filter: str | None = None


class UpsertBody(BaseModel):
    tenant_id: str
    config: dict
    documents: list[dict]


class DeleteBody(BaseModel):
    tenant_id: str
    collection_name: str
    ids: list[str]


class DeleteCollectionBody(BaseModel):
    tenant_id: str
    collection_name: str
