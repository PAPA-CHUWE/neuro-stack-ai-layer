"""Knowledge (RAG) — chunking, ingestion, search, query."""

from __future__ import annotations

import logging

from app.shared.providers.base import EmbeddingRequest, CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.providers.vector_store import vector_store, CollectionConfig, Document
from app.shared.database import get_pg_pool
from app.features.knowledge.schemas import DEFAULT_COLLECTION, SearchResultItem

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

RAG_SYSTEM_PROMPT = """You are a knowledge assistant. Answer the user's question using ONLY the provided reference documents.
If the documents don't contain enough information to answer, say so honestly.
Always cite which document(s) your answer is based on."""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]


async def ingest_document(tenant_id: str, title: str, content: str,
                          collection: str = DEFAULT_COLLECTION, metadata: dict | None = None):
    import json, uuid
    from datetime import datetime, timezone

    pool = await get_pg_pool()
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    chunks = chunk_text(content)

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO rag_documents (id, tenant_id, collection, title, content, metadata, chunk_count, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            doc_id, tenant_id, collection, title, content, json.dumps(metadata or {}), len(chunks), now,
        )

    vc = CollectionConfig(name=collection, dimension=1024)
    docs: list[Document] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        try:
            emb = await llm_provider.embed(EmbeddingRequest(input=chunk))
            docs.append(Document(id=chunk_id, vector=emb.embedding, fields={
                "document_id": doc_id, "tenant_id": tenant_id,
                "collection": collection, "chunk_index": i, "content_preview": chunk[:200],
            }))
        except Exception as e:
            logger.error("Failed to embed chunk %d: %s", i, e)
            continue

    if docs:
        await vector_store.upsert(tenant_id, vc, docs)

    async with pool.acquire() as conn:
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            await conn.execute(
                """INSERT INTO rag_chunks (id, document_id, tenant_id, collection, chunk_index, content, embedding_id, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                chunk_id, doc_id, tenant_id, collection, i, chunk, chunk_id, now,
            )

    return doc_id, len(chunks)


async def search_documents(tenant_id: str, query: str, collection: str = DEFAULT_COLLECTION, top_k: int = 5):
    emb = await llm_provider.embed(EmbeddingRequest(input=query))
    vc = CollectionConfig(name=collection, dimension=1024)
    results = await vector_store.search(tenant_id, vc, emb.embedding, top_k=top_k)

    pool = await get_pg_pool()
    items: list[SearchResultItem] = []
    for r in results:
        fields = r.fields or {}
        chunk_content = ""
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT content FROM rag_chunks WHERE id = $1", r.id)
            if row:
                chunk_content = row["content"]
        items.append(SearchResultItem(
            id=r.id, document_id=fields.get("document_id", ""),
            content=chunk_content or fields.get("content_preview", ""),
            score=r.score,
            metadata={"collection": collection, "chunk_index": fields.get("chunk_index", 0)},
        ))

    return items, emb.model, emb.prompt_tokens


async def rag_query(tenant_id: str, question: str, collection: str = DEFAULT_COLLECTION, top_k: int = 5):
    emb = await llm_provider.embed(EmbeddingRequest(input=question))
    vc = CollectionConfig(name=collection, dimension=1024)
    results = await vector_store.search(tenant_id, vc, emb.embedding, top_k=top_k)

    pool = await get_pg_pool()
    context_parts: list[str] = []
    sources: list[SearchResultItem] = []

    for r in results:
        fields = r.fields or {}
        chunk_content = ""
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT content FROM rag_chunks WHERE id = $1", r.id)
            if row:
                chunk_content = row["content"]
        content = chunk_content or fields.get("content_preview", "")
        if content:
            context_parts.append(content)
        sources.append(SearchResultItem(
            id=r.id, document_id=fields.get("document_id", ""),
            content=content, score=r.score,
            metadata={"collection": collection},
        ))

    if not context_parts:
        return "No relevant documents found for your question.", sources, emb.model, emb.prompt_tokens, 0

    context = "\n\n---\n\n".join(context_parts)
    completion = await llm_provider.complete(
        CompletionRequest(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=f"Reference documents:\n\n{context}\n\n---\n\nQuestion: {question}",
            temperature=0.2, max_tokens=1024,
        )
    )

    return completion.content, sources, completion.model, completion.prompt_tokens, completion.completion_tokens
