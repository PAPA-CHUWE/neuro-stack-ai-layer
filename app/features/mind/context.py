"""Mind feature — Page context resolution via semantic similarity."""

import asyncio
import json
import logging
import math
import time

from app.features.mind.schemas import PageContext
from app.features.mind.tools import get_entity_context, get_page_context
from app.shared.providers.base import EmbeddingRequest
from app.shared.providers.mistral import llm_provider

logger = logging.getLogger(__name__)

PAGE_RELATIVE_EXEMPLARS = [
    "what page am I on",
    "which page is this",
    "what's on this page",
    "tell me about this page",
    "which page did I open",
    "where am I right now in the app",
    "what can I do here",
]

_exemplar_cache: list[list[float]] | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_exemplar_embeddings() -> list[list[float]]:
    global _exemplar_cache
    if _exemplar_cache is not None:
        return _exemplar_cache

    t0 = time.monotonic()
    results = await asyncio.gather(*[
        llm_provider.embed(EmbeddingRequest(input=e)) for e in PAGE_RELATIVE_EXEMPLARS
    ])
    _exemplar_cache = [r.embedding for r in results]
    logger.debug(
        "Cached %d exemplar embeddings in %.0fms",
        len(PAGE_RELATIVE_EXEMPLARS),
        (time.monotonic() - t0) * 1000,
    )
    return _exemplar_cache


async def is_page_relative(question: str, threshold: float = 0.70) -> bool:
    """Semantic check: does this question ask about the current page/UI state?"""
    t0 = time.monotonic()
    q_emb = await llm_provider.embed(EmbeddingRequest(input=question))
    exemplars = await _get_exemplar_embeddings()
    scores = [_cosine_similarity(q_emb.embedding, e) for e in exemplars]
    best = max(scores)
    elapsed = (time.monotonic() - t0) * 1000
    matched = best > threshold
    logger.debug(
        "is_page_relative(%r) => best=%.4f threshold=%.4f result=%s (%.0fms)",
        question, best, threshold, matched, elapsed,
    )
    if 0.60 <= best <= threshold:
        logger.info(
            "is_page_relative near-miss: question=%r best_score=%.4f threshold=%.4f",
            question, best, threshold,
        )
    return matched


async def resolve_page_context(
    page_ctx: PageContext | None, question: str, tenant_id: str
) -> str:
    """If pageContext is present and the question is page-relative, fetch context data."""
    if not page_ctx:
        return ""

    page_relative = await is_page_relative(question)
    if not page_relative:
        return ""

    if page_ctx.selectedEntity:
        entity_type = page_ctx.selectedEntity.get("type")
        entity_id = page_ctx.selectedEntity.get("id")
        if entity_type and entity_id:
            ctx = await get_entity_context(entity_type, entity_id, tenant_id)
            return f"\n\nPAGE CONTEXT (entity):\n{json.dumps(ctx, default=str)}"

    if page_ctx.route:
        ctx = await get_page_context(page_ctx.route, tenant_id)
        page_info = f"Route: {page_ctx.route}"
        if page_ctx.title:
            page_info += f"\nTitle: {page_ctx.title}"
        if page_ctx.userRole:
            page_info += f"\nUser role: {page_ctx.userRole}"
        return f"\n\nPAGE CONTEXT:\n{page_info}\nData:\n{json.dumps(ctx, default=str)}"

    return ""
