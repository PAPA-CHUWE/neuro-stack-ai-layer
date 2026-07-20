"""Mind feature — Shared retrieval, intent classification, and category routing."""

import json
import logging

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.database import get_pg_pool
from app.features.mind.schemas import SourceItem, MindMode
from app.features.mind.categories import KNOWLEDGE_CATEGORIES
from app.features.mind.prompts import INTENT_CLASSIFIER_PROMPT, ROUTER_SYSTEM_PROMPT
from app.features.knowledge.service import search_documents

logger = logging.getLogger(__name__)


async def classify_intent(question: str, tenant_id: str = "") -> dict:
    """Classify user intent into mode + optional categories/tool."""
    try:
        completion = await llm_provider.complete(
            CompletionRequest(
                system_prompt=INTENT_CLASSIFIER_PROMPT,
                user_prompt=f"User message: {question}",
                temperature=0.0,
                max_tokens=128,
                json_mode=True,
            )
        )
        data = json.loads(completion.content)
        mode = data.get("mode", "knowledge")
        if mode not in ("conversation", "knowledge", "platform", "reasoning", "action"):
            mode = "knowledge"
        categories = [c for c in data.get("categories", []) if c in KNOWLEDGE_CATEGORIES]
        tool = data.get("tool")
        return {"mode": mode, "categories": categories[:2], "tool": tool}
    except Exception as e:
        logger.warning("Intent classification failed for tenant=%s question=%r: %s", tenant_id, question, e)
        return {"mode": "conversation", "categories": [], "tool": None, "needs_clarification": True}


async def route_category(question: str) -> tuple[list[str], str]:
    """Classify the question into 1-2 knowledge categories."""
    try:
        completion = await llm_provider.complete(
            CompletionRequest(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=f"Question: {question}",
                temperature=0.0,
                max_tokens=128,
                json_mode=True,
            )
        )
        data = json.loads(completion.content)
        categories = [c for c in data.get("categories", []) if c in KNOWLEDGE_CATEGORIES]
        confidence = data.get("confidence", "low")
        return categories[:2], confidence
    except Exception as e:
        logger.warning("Category routing failed: %s", e)
        return [], "low"


async def retrieve_context(
    tenant_id: str, question: str, categories: list[str], collection: str, top_k: int
) -> tuple[list[str], list[SourceItem], float, list]:
    """Embed question, search vector store, fetch chunk content.

    Delegates to knowledge/service.search_documents for the core pipeline,
    then adapts results to mind-specific SourceItem format with document titles.

    Returns (context_parts, sources, top_score, search_results).
    """
    from app.features.knowledge.schemas import SearchResultItem

    collections_to_search = categories if categories else [collection]
    all_items: list[SearchResultItem] = []

    for coll in collections_to_search:
        try:
            items, _, _ = await search_documents(tenant_id, question, coll, top_k)
            all_items.extend(items)
        except Exception as e:
            logger.warning("Search failed for collection %s: %s", coll, e)

    all_items.sort(key=lambda r: r.score, reverse=True)
    all_items = all_items[:top_k]

    pool = await get_pg_pool()
    context_parts: list[str] = []
    sources: list[SourceItem] = []
    top_score = 0.0

    for i, item in enumerate(all_items):
        content = item.content
        doc_title = ""
        if item.document_id:
            async with pool.acquire() as conn:
                doc_row = await conn.fetchrow(
                    "SELECT title FROM rag_documents WHERE id = $1", item.document_id
                )
                if doc_row:
                    doc_title = doc_row["title"]

        if content:
            context_parts.append(f"[source:{i + 1}] {content}")
            sources.append(SourceItem(
                id=item.id,
                title=doc_title or f"Document {item.document_id or 'unknown'}",
                excerpt=content[:300],
                relevance_score=round(item.score, 4),
            ))
            if item.score > top_score:
                top_score = item.score

    return context_parts, sources, top_score, all_items
