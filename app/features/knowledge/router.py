"""Knowledge (RAG) — FastAPI router."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.shared.database import get_pg_pool
from app.features.knowledge.schemas import (
    IngestDocumentBody, IngestDocumentResponse,
    SearchBody, SearchResponse, SearchResultItem,
    RagQueryBody, RagQueryResponse, DocumentInfo,
)
from app.features.knowledge.service import ingest_document, search_documents, rag_query

logger = logging.getLogger(__name__)

router = APIRouter()


def _now():
    return datetime.now(timezone.utc)


@router.post("/ingest", response_model=IngestDocumentResponse)
async def ingest(body: IngestDocumentBody):
    doc_id, chunk_count = await ingest_document(
        body.tenant_id, body.title, body.content, body.collection, body.metadata,
    )
    return IngestDocumentResponse(
        id=doc_id, title=body.title, chunk_count=chunk_count, collection=body.collection,
    )


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchBody):
    try:
        items, model, prompt_tokens = await search_documents(
            body.tenant_id, body.query, body.collection, body.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e}")

    return SearchResponse(
        results=items, query=body.query, model=model, prompt_tokens=prompt_tokens,
    )


@router.post("/query", response_model=RagQueryResponse)
async def query(body: RagQueryBody):
    try:
        answer, sources, model, pt, ct = await rag_query(
            body.tenant_id, body.question, body.collection, body.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RAG query failed: {e}")

    return RagQueryResponse(
        answer=answer, sources=sources, model=model,
        prompt_tokens=pt, completion_tokens=ct,
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(tenant_id: str, collection: str | None = None):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        if collection:
            rows = await conn.fetch(
                "SELECT id, title, collection, chunk_count, created_at FROM rag_documents WHERE tenant_id = $1 AND collection = $2 ORDER BY created_at DESC",
                tenant_id, collection,
            )
        else:
            rows = await conn.fetch(
                "SELECT id, title, collection, chunk_count, created_at FROM rag_documents WHERE tenant_id = $1 ORDER BY created_at DESC",
                tenant_id,
            )
    return [
        DocumentInfo(
            id=r["id"], title=r["title"], collection=r["collection"],
            chunk_count=r["chunk_count"],
            created_at=r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
        )
        for r in rows
    ]


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, tenant_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT collection FROM rag_documents WHERE id = $1 AND tenant_id = $2",
            document_id, tenant_id,
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        chunk_rows = await conn.fetch(
            "SELECT id FROM rag_chunks WHERE document_id = $1", document_id,
        )
        chunk_ids = [r["id"] for r in chunk_rows]

        await conn.execute("DELETE FROM rag_documents WHERE id = $1", document_id)

    if chunk_ids:
        from app.shared.providers.vector_store import vector_store
        await vector_store.delete_documents(tenant_id, doc["collection"], chunk_ids)

    return {"ok": True}
