"""Streaming — FastAPI router."""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.features.streaming.schemas import StreamBody

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat")
async def stream_chat(body: StreamBody):
    request = CompletionRequest(
        system_prompt=body.system_prompt,
        user_prompt=body.user_prompt,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        json_mode=body.json_mode,
    )

    async def event_generator():
        try:
            async for chunk in llm_provider.stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/tutor")
async def stream_tutor(body: dict):
    question = body.get("question", "")
    context = body.get("context", "")
    lesson_content = body.get("lesson_content", "")

    user_content = question
    if lesson_content:
        user_content = f"Lesson content:\n{lesson_content}\n\nQuestion: {question}"
    elif context:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

    request = CompletionRequest(
        system_prompt=(
            "You are an AI tutor in an LMS platform. Help learners understand course material. "
            "Be concise but thorough. Use plain language."
        ),
        user_prompt=user_content,
        temperature=0.3, max_tokens=1024,
    )

    async def event_generator():
        try:
            async for chunk in llm_provider.stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error("Stream tutor error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
