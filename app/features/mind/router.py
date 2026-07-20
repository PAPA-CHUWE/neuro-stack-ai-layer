"""Mind feature — FastAPI router.

Thin orchestration layer. All business logic lives in:
- service.py (intent classification, retrieval)
- tools.py (page context handlers)
- context.py (page context resolution)
- confidence.py (grounding thresholds)
- composer.py (fallback copy)
- traces.py (response trace storage)
"""

import json
import logging
import re
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.features.mind.schemas import (
    MindQueryBody, MindStreamBody, MindResponse,
    Confidence, MindMode, SourceItem,
)
from app.features.mind.prompts import (
    KNOWLEDGE_SYSTEM_PROMPT, CONVERSATION_SYSTEM_PROMPT,
    PLATFORM_SYSTEM_PROMPT, REASONING_SYSTEM_PROMPT,
)
from app.features.mind.composer import FALLBACK_COPY
from app.features.mind.confidence import derive_confidence
from app.features.mind.context import resolve_page_context
from app.features.mind.service import classify_intent, route_category, retrieve_context
from app.features.mind.traces import store_trace
from app.features.mind.conversation_state import (
    has_introduced as check_introduced, mark_introduced, append_message,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Fast-path intent (regex, no LLM call) ────────────────

GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank\s*you|bye|cheers|ok|okay|got\s*it)\b",
    re.IGNORECASE,
)

PLATFORM_RE = re.compile(
    r"(how many|number of|count|total|list all|show me all|what('s| is) the (total|count|number))\s+"
    r"(published |active )?(courses?|learners?|users?|enrollments?|students?|instructors?|departments?)",
    re.IGNORECASE,
)

ACTION_RE = re.compile(
    r"\b(go to|open|take me to|navigate to|show me the)\s+(.+)",
    re.IGNORECASE,
)

# ── Tier 1: Navigation (read-only, client-side route push) ──
NAVIGATION_REGISTRY: dict[str, str] = {
    "courses": "/admin/courses",
    "users": "/admin/users",
    "tenants": "/admin/tenants",
    "reports": "/admin/reports",
    "roles": "/admin/roles",
    "permissions": "/admin/permissions",
    "dashboard": "/admin",
    "certificates": "/student/certificates",
    "learning-paths": "/student/learning-paths",
}

# ── Tier 2: Mutating actions — STUB, DO NOT POPULATE ────────
# Mutating actions require RBAC check + confirmation flow + audit log
# before any entry is added here. Do not wire this to real endpoints
# without that scaffolding.
MUTATE_REGISTRY: dict[str, str] = {}


def _fast_path_intent(message: str) -> MindMode | None:
    """Cheap regex check before any LLM call."""
    stripped = message.strip()
    if GREETING_RE.match(stripped):
        return MindMode.conversation
    if PLATFORM_RE.search(stripped):
        return MindMode.platform
    if ACTION_RE.search(stripped):
        target = _resolve_action_target(stripped)
        if target:
            return MindMode.action
    return None


def _resolve_action_target(question: str) -> str | None:
    """Extract the navigation target from an action-mode question and match to registry."""
    m = ACTION_RE.search(question)
    if not m:
        return None
    phrase = m.group(2).strip().lower().rstrip("?.!").strip()
    # Direct match
    if phrase in NAVIGATION_REGISTRY:
        return phrase
    # Token containment: "courses" in "let's go to courses"
    for key in NAVIGATION_REGISTRY:
        if key in phrase or phrase in key:
            return key
    return None


def _select_platform_tool(question: str) -> str:
    """Derive the correct platform tool from keywords in the question."""
    q = question.lower()
    if "skill" in q or "lagging" in q or "weak" in q or "gap" in q:
        return "skill_gap"
    if "learner" in q or "student" in q or "user" in q:
        return "learner_count"
    if "enrollment" in q or "enrolled" in q:
        return "enrollment_count"
    if "instructor" in q:
        return "instructor_count"
    if "department" in q:
        return "department_count"
    if "list" in q or "show me all" in q:
        return "course_list"
    return "course_count"


# ── Platform tool execution ───────────────────────────────

async def _execute_platform_tool(tool: str | None, tenant_id: str) -> str:
    """Fetch live data from Postgres for platform-mode queries."""
    from app.shared.database import get_pg_pool
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        if tool == "course_count":
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM courses WHERE status = 'PUBLISHED'")
            return f"Published courses: {row['count']}" if row else "No data"
        elif tool == "enrollment_count":
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM enrollments")
            return f"Total enrollments: {row['count']}" if row else "No data"
        elif tool == "learner_count":
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM users WHERE status = 'ACTIVE'")
            return f"Active learners: {row['count']}" if row else "No data"
        elif tool == "course_list":
            rows = await conn.fetch('SELECT title, status FROM courses ORDER BY "createdAt" DESC LIMIT 10')
            if rows:
                listing = "\n".join(f"- {r['title']} ({r['status']})" for r in rows)
                return f"Recent courses:\n{listing}"
            return "No courses found"
        elif tool == "instructor_count":
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM users WHERE role = 'INSTRUCTOR' AND status = 'ACTIVE'")
            return f"Active instructors: {row['count']}" if row else "No data"
        elif tool == "department_count":
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM departments")
            return f"Total departments: {row['count']}" if row else "No data"
        elif tool == "skill_gap":
            rows = await conn.fetch(
                """SELECT s.name, sp.current_level, sp.target_level
                   FROM skill_profiles sp JOIN skills s ON s.id = sp.skill_id
                   WHERE sp.tenant_id = $1
                   ORDER BY (sp.target_level - sp.current_level) DESC LIMIT 10""",
                tenant_id,
            )
            if not rows:
                return "No skill profile data available for this tenant."
            return "\n".join(
                f"{r['name']}: current {r['current_level']}% → target {r['target_level']}%"
                for r in rows
            )
        else:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM courses")
            return f"Total courses: {row['count']}" if row else "No data"


# ── Action tool execution ────────────────────────────────

async def _execute_action(target_key: str) -> dict:
    """Resolve a navigation target from the allow-listed registry."""
    route = NAVIGATION_REGISTRY.get(target_key)
    if not route:
        return {"resolved": False}
    return {"resolved": True, "route": route, "label": target_key.replace("-", " ").title()}


# ── Non-streaming endpoint ────────────────────────────────

@router.post("/query", response_model=MindResponse)
async def mind_query(body: MindQueryBody):
    start = time.monotonic()
    if body.conversation_id:
        await append_message(body.conversation_id, body.tenant_id, "user", body.question)
    try:
        response = await _mind_query_inner(body, start)
        if body.conversation_id:
            await append_message(body.conversation_id, body.tenant_id, "assistant", response.answer)
        return response
    except Exception as e:
        logger.error(
            "Unhandled mind_query error for tenant=%s question=%r: %s",
            body.tenant_id, body.question, e, exc_info=True,
        )
        return MindResponse(
            answer="I wasn't able to process that — could you try rephrasing?",
            sources=[],
            confidence=Confidence.ungrounded,
            needs_review=True,
            model="error",
            mode=MindMode.conversation,
        )


async def _mind_query_inner(body: MindQueryBody, start: float) -> MindResponse:
    # Step 1: Fast-path check
    mode = _fast_path_intent(body.question)

    # Step 2: Intent classification (if fast path didn't match)
    categories = []
    tool = None
    needs_clarification = False
    if mode is None:
        intent = await classify_intent(body.question, body.tenant_id)
        mode = MindMode(intent["mode"])
        categories = intent["categories"]
        tool = intent.get("tool")
        needs_clarification = intent.get("needs_clarification", False)
    elif mode == MindMode.platform:
        tool = _select_platform_tool(body.question)

    # Step 2.5: Resolve page context if present and question is page-relative
    page_context_block = await resolve_page_context(body.pageContext, body.question, body.tenant_id)

    # ── ACTION MODE (Tier 1: navigation) ───────────────────
    if mode == MindMode.action:
        target_key = _resolve_action_target(body.question)
        result = await _execute_action(target_key or "")
        if not result["resolved"]:
            answer = "I'm not able to navigate there yet — try Courses, Users, Tenants, Reports, or Certificates."
            actions: list[dict] = []
        else:
            answer = f"Taking you to {result['label']}."
            actions = [{"type": "NAVIGATE", "label": f"Open {result['label']}", "target": result["route"]}]
        latency_ms = int((time.monotonic() - start) * 1000)
        trace_id = await store_trace(
            body.tenant_id, "action", body.question, "registry",
            "grounded", [], latency_ms,
        )
        return MindResponse(
            answer=answer,
            sources=[],
            confidence=Confidence.grounded,
            needs_review=False,
            model="registry",
            mode=MindMode.action,
            actions=actions,
            trace_id=trace_id,
        )

    # ── CLARIFICATION (classifier failure) ─────────────────
    if needs_clarification:
        completion = await llm_provider.complete(
            CompletionRequest(
                system_prompt=CONVERSATION_SYSTEM_PROMPT,
                user_prompt=f"User message: {body.question}\n\nNote: You had trouble classifying this message. Ask the user to rephrase or clarify what they need help with.",
                temperature=0.3,
                max_tokens=256,
            )
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        trace_id = await store_trace(
            body.tenant_id, "conversation", body.question, completion.model,
            "grounded", [], latency_ms,
        )
        return MindResponse(
            answer=completion.content,
            sources=[],
            confidence=Confidence.grounded,
            needs_review=False,
            model=completion.model,
            mode=MindMode.conversation,
            trace_id=trace_id,
        )

    # ── CONVERSATION MODE ──────────────────────────────────
    if mode == MindMode.conversation:
        greeting_context = ""
        if body.user_name:
            greeting_context = f"The user's name is {body.user_name}. "

        conv_system = CONVERSATION_SYSTEM_PROMPT
        if body.conversation_id:
            already_introduced = await check_introduced(body.conversation_id)
            logger.debug(
                "conversation_mode: conversation_id=%s has_introduced=%s",
                body.conversation_id, already_introduced,
            )
            if already_introduced:
                conv_system = (
                    "You are Neuro, a helpful and friendly AI learning assistant. "
                    "You have already introduced yourself to this user in this conversation. "
                    "Do NOT repeat your introduction or say 'I'm Neuro' again. "
                    "Just respond naturally and helpfully to their message."
                )

        user_prompt = f"{greeting_context}User message: {body.question}"
        if page_context_block:
            user_prompt += page_context_block
        completion = await llm_provider.complete(
            CompletionRequest(
                system_prompt=conv_system,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=256,
            )
        )
        if body.conversation_id:
            await mark_introduced(body.conversation_id, body.tenant_id)
        latency_ms = int((time.monotonic() - start) * 1000)
        trace_id = await store_trace(
            body.tenant_id, "conversation", body.question, completion.model,
            "grounded", [], latency_ms,
        )
        return MindResponse(
            answer=completion.content,
            sources=[],
            confidence=Confidence.grounded,
            needs_review=False,
            model=completion.model,
            mode=MindMode.conversation,
            trace_id=trace_id,
        )

    # ── PLATFORM MODE ──────────────────────────────────────
    if mode == MindMode.platform:
        tool_result = await _execute_platform_tool(tool, body.tenant_id)
        user_prompt = f"TOOL RESULT:\n{tool_result}\n\nUser question: {body.question}"
        if page_context_block:
            user_prompt += page_context_block
        completion = await llm_provider.complete(
            CompletionRequest(
                system_prompt=PLATFORM_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=256,
            )
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        trace_id = await store_trace(
            body.tenant_id, "platform", body.question, completion.model,
            "grounded", [], latency_ms,
        )
        return MindResponse(
            answer=completion.content,
            sources=[],
            confidence=Confidence.grounded,
            needs_review=False,
            model=completion.model,
            mode=MindMode.platform,
            trace_id=trace_id,
        )

    # ── KNOWLEDGE MODE ─────────────────────────────────────
    if mode == MindMode.knowledge:
        if not categories and not body.category:
            categories, route_confidence = await route_category(body.question)
            if route_confidence == "low" and not categories:
                latency_ms = int((time.monotonic() - start) * 1000)
                trace_id = await store_trace(
                    body.tenant_id, "knowledge", body.question, "router",
                    "ungrounded", [], latency_ms,
                )
                return MindResponse(
                    answer=FALLBACK_COPY["knowledge_unrouteable"],
                    sources=[],
                    confidence=Confidence.ungrounded,
                    needs_review=True,
                    model="router",
                    mode=MindMode.knowledge,
                    trace_id=trace_id,
                )
        elif body.category:
            categories = [body.category]

        context_parts, sources, top_score, search_results = await retrieve_context(
            body.tenant_id, body.question, categories, body.collection, body.top_k
        )
        confidence = derive_confidence(search_results, top_score)

        # Patch A — hard gate: no context → no fabricated answer
        if not context_parts:
            latency_ms = int((time.monotonic() - start) * 1000)
            trace_id = await store_trace(
                body.tenant_id, "knowledge", body.question, "none",
                "ungrounded", [], latency_ms,
            )
            return MindResponse(
                answer=FALLBACK_COPY["knowledge_empty_retrieval"],
                sources=[],
                confidence=Confidence.ungrounded,
                needs_review=True,
                model="none",
                mode=MindMode.knowledge,
                trace_id=trace_id,
            )

        context_block = "\n\n---\n\n".join(context_parts)
        user_content = f"CONTEXT:\n\n{context_block}\n\n---\n\nQuestion: {body.question}"
        if body.user_context:
            user_content = f"ADDITIONAL CONTEXT:\n{body.user_context}\n\n{user_content}"
        if page_context_block:
            user_content += page_context_block

        completion = await llm_provider.complete(
            CompletionRequest(
                system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
                user_prompt=user_content,
                temperature=0.2,
                max_tokens=1024,
            )
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        trace_id = await store_trace(
            body.tenant_id, "knowledge", body.question, completion.model,
            confidence.value, sources, latency_ms,
        )
        return MindResponse(
            answer=completion.content,
            sources=sources,
            confidence=confidence,
            needs_review=confidence != Confidence.grounded,
            model=completion.model,
            mode=MindMode.knowledge,
            trace_id=trace_id,
        )

    # ── REASONING MODE ─────────────────────────────────────
    tool_result = None
    if tool:
        try:
            tool_result = await _execute_platform_tool(tool, body.tenant_id)
        except Exception as e:
            logger.warning("Reasoning tool execution failed: %s", e)

    context_parts, sources, top_score, search_results = await retrieve_context(
        body.tenant_id, body.question, categories, body.collection, body.top_k
    )
    confidence = derive_confidence(search_results, top_score)

    if not context_parts and not tool_result:
        latency_ms = int((time.monotonic() - start) * 1000)
        trace_id = await store_trace(
            body.tenant_id, "reasoning", body.question, "none",
            "ungrounded", [], latency_ms,
        )
        return MindResponse(
            answer=FALLBACK_COPY["reasoning_insufficient_data"],
            sources=[],
            confidence=Confidence.ungrounded,
            needs_review=True,
            model="none",
            mode=MindMode.reasoning,
            trace_id=trace_id,
        )

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else ""
    if tool_result:
        context_block = f"LEARNER DATA:\n{tool_result}\n\n{context_block}" if context_block else f"LEARNER DATA:\n{tool_result}"
    user_content = f"CONTEXT:\n\n{context_block}\n\n---\n\nQuestion: {body.question}"
    if page_context_block:
        user_content += page_context_block

    completion = await llm_provider.complete(
        CompletionRequest(
            system_prompt=REASONING_SYSTEM_PROMPT,
            user_prompt=user_content,
            temperature=0.3,
            max_tokens=1024,
        )
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    trace_id = await store_trace(
        body.tenant_id, "reasoning", body.question, completion.model,
        confidence.value, sources, latency_ms,
    )
    return MindResponse(
        answer=completion.content,
        sources=sources,
        confidence=confidence,
        needs_review=confidence != Confidence.grounded,
        model=completion.model,
        mode=MindMode.reasoning,
        trace_id=trace_id,
    )


# ── Streaming endpoint ────────────────────────────────────

async def _persist_assistant_reply(body_iterator, conversation_id: str, tenant_id: str):
    """Tee a streaming response: pass every event through untouched, and persist the
    assembled assistant reply once the stream completes."""
    chunks: list[str] = []
    async for event in body_iterator:
        yield event
        if isinstance(event, str) and event.startswith("data: "):
            try:
                payload = json.loads(event[len("data: "):].strip())
                if payload.get("type") == "chunk":
                    chunks.append(payload.get("content", ""))
            except json.JSONDecodeError:
                pass
    if chunks:
        await append_message(conversation_id, tenant_id, "assistant", "".join(chunks))


@router.post("/stream")
async def mind_stream(body: MindStreamBody):
    """Streaming variant with 5-mode intent orchestration."""
    start = time.monotonic()

    if body.conversation_id:
        await append_message(body.conversation_id, body.tenant_id, "user", body.question)

    try:
        response = await _mind_stream_inner(body, start)
        if body.conversation_id:
            response.body_iterator = _persist_assistant_reply(
                response.body_iterator, body.conversation_id, body.tenant_id
            )
        return response
    except Exception as e:
        logger.error(
            "Unhandled mind_stream error for tenant=%s question=%r: %s",
            body.tenant_id, body.question, e, exc_info=True,
        )

        async def error_stream():
            yield f"data: {json.dumps({'type': 'mode', 'mode': 'conversation'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', 'confidence': 'ungrounded', 'needs_review': True})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'content': 'I wasn\u2019t able to process that \u2014 could you try rephrasing?'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )


async def _mind_stream_inner(body: MindStreamBody, start: float):
    mode = _fast_path_intent(body.question)
    categories = []
    tool = None
    needs_clarification = False
    if mode is None:
        intent = await classify_intent(body.question, body.tenant_id)
        mode = MindMode(intent["mode"])
        categories = intent["categories"]
        tool = intent.get("tool")
        needs_clarification = intent.get("needs_clarification", False)
    elif mode == MindMode.platform:
        tool = _select_platform_tool(body.question)

    mode_payload = json.dumps({"type": "mode", "mode": mode.value})
    page_context_block = await resolve_page_context(body.pageContext, body.question, body.tenant_id)

    # ── ACTION MODE (Tier 1: navigation) ───────────────────
    if mode == MindMode.action:
        target_key = _resolve_action_target(body.question)
        result = await _execute_action(target_key or "")
        if not result["resolved"]:
            answer = "I'm not able to navigate there yet — try Courses, Users, Tenants, Reports, or Certificates."
            actions: list[dict] = []
        else:
            answer = f"Taking you to {result['label']}."
            actions = [{"type": "NAVIGATE", "label": f"Open {result['label']}", "target": result["route"]}]

        async def action_stream():
            yield f"data: {mode_payload}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', 'confidence': 'grounded', 'needs_review': False})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'content': answer})}\n\n"
            if actions:
                yield f"data: {json.dumps({'type': 'actions', 'actions': actions})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        latency_ms = int((time.monotonic() - start) * 1000)
        await store_trace(body.tenant_id, "action", body.question, "registry", "grounded", [], latency_ms)
        return StreamingResponse(
            action_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── CLARIFICATION (classifier failure) ─────────────────
    if needs_clarification:
        request = CompletionRequest(
            system_prompt=CONVERSATION_SYSTEM_PROMPT,
            user_prompt=f"User message: {body.question}\n\nNote: You had trouble classifying this message. Ask the user to rephrase or clarify what they need help with.",
            temperature=0.3,
            max_tokens=256,
        )

        async def clarify_stream():
            yield f"data: {mode_payload}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', 'confidence': 'grounded', 'needs_review': False})}\n\n"
            try:
                async for chunk in llm_provider.stream(request):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            clarify_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── CONVERSATION MODE ──────────────────────────────────
    if mode == MindMode.conversation:
        greeting_context = ""
        if body.user_name:
            greeting_context = f"The user's name is {body.user_name}. "

        conv_system = CONVERSATION_SYSTEM_PROMPT
        if body.conversation_id:
            already_introduced = await check_introduced(body.conversation_id)
            logger.debug(
                "conversation_mode_stream: conversation_id=%s has_introduced=%s",
                body.conversation_id, already_introduced,
            )
            if already_introduced:
                conv_system = (
                    "You are Neuro, a helpful and friendly AI learning assistant. "
                    "You have already introduced yourself to this user in this conversation. "
                    "Do NOT repeat your introduction or say 'I'm Neuro' again. "
                    "Just respond naturally and helpfully to their message."
                )

        user_prompt = f"{greeting_context}User message: {body.question}"
        if page_context_block:
            user_prompt += page_context_block
        request = CompletionRequest(
            system_prompt=conv_system,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=256,
        )

        async def conv_stream():
            yield f"data: {mode_payload}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', 'confidence': 'grounded', 'needs_review': False})}\n\n"
            try:
                async for chunk in llm_provider.stream(request):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                if body.conversation_id:
                    await mark_introduced(body.conversation_id, body.tenant_id)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            conv_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── PLATFORM MODE ──────────────────────────────────────
    if mode == MindMode.platform:
        tool_result = await _execute_platform_tool(tool, body.tenant_id)
        user_prompt = f"TOOL RESULT:\n{tool_result}\n\nUser question: {body.question}"
        if page_context_block:
            user_prompt += page_context_block
        request = CompletionRequest(
            system_prompt=PLATFORM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=256,
        )

        async def platform_stream():
            yield f"data: {mode_payload}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'confidence', 'confidence': 'grounded', 'needs_review': False})}\n\n"
            try:
                async for chunk in llm_provider.stream(request):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            platform_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── KNOWLEDGE / REASONING MODE ────────────────────────
    tool_result = None
    if mode == MindMode.reasoning and tool:
        try:
            tool_result = await _execute_platform_tool(tool, body.tenant_id)
        except Exception as e:
            logger.warning("Reasoning tool execution failed: %s", e)

    if not categories and not body.category:
        categories, route_confidence = await route_category(body.question)
        if route_confidence == "low" and not categories:
            fallback_msg = FALLBACK_COPY["reasoning_insufficient_data"] if mode == MindMode.reasoning else FALLBACK_COPY["knowledge_unrouteable"]
            async def unroutable_stream():
                yield f"data: {mode_payload}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'confidence', 'confidence': 'ungrounded', 'needs_review': True})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'content': fallback_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return StreamingResponse(
                unroutable_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
    elif body.category:
        categories = [body.category]

    context_parts, sources, top_score, search_results = await retrieve_context(
        body.tenant_id, body.question, categories, body.collection, body.top_k
    )
    confidence = derive_confidence(search_results, top_score)

    sources_payload = json.dumps({"type": "sources", "sources": [s.model_dump() for s in sources]})
    confidence_payload = json.dumps({
        "type": "confidence",
        "confidence": confidence.value,
        "needs_review": confidence != Confidence.grounded,
    })

    # Patch A — no context → ungrounded gate
    if not context_parts and not tool_result:
        fallback_msg = FALLBACK_COPY["reasoning_insufficient_data"] if mode == MindMode.reasoning else FALLBACK_COPY["knowledge_empty_retrieval"]
        async def empty_stream():
            yield f"data: {mode_payload}\n\n"
            yield f"data: {sources_payload}\n\n"
            yield f"data: {confidence_payload}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'content': fallback_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(
            empty_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else ""
    if tool_result:
        context_block = f"LEARNER DATA:\n{tool_result}\n\n{context_block}" if context_block else f"LEARNER DATA:\n{tool_result}"
    user_content = f"CONTEXT:\n\n{context_block}\n\n---\n\nQuestion: {body.question}"
    if body.user_context:
        user_content = f"ADDITIONAL CONTEXT:\n{body.user_context}\n\n{user_content}"
    if page_context_block:
        user_content += page_context_block

    system = REASONING_SYSTEM_PROMPT if mode == MindMode.reasoning else KNOWLEDGE_SYSTEM_PROMPT
    request = CompletionRequest(
        system_prompt=system,
        user_prompt=user_content,
        temperature=0.2,
        max_tokens=1024,
    )

    async def knowledge_stream():
        yield f"data: {mode_payload}\n\n"
        yield f"data: {sources_payload}\n\n"
        yield f"data: {confidence_payload}\n\n"
        try:
            async for chunk in llm_provider.stream(request):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error("Mind stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        knowledge_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
