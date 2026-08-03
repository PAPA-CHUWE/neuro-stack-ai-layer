"""Video — FastAPI router.

Generates a narrated slideshow video from lesson content: Mistral writes a
slide-by-slide script, Piper narrates it (free, local TTS), Pillow renders
each slide, and ffmpeg assembles narration + slides into an mp4. Returns the
raw video bytes — no paid video-generation provider involved.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Response

from app.features.video.assembler import build_video
from app.features.video.schemas import GenerateVideoBody
from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.utils.json_parse import extract_json

logger = logging.getLogger(__name__)

router = APIRouter()

VIDEO_SCRIPT_SYSTEM_PROMPT = """You are a course video scriptwriter for an LMS platform.

Given a lesson title and content, break it into a small number of slides that teach the material clearly.

Rules:
1. Produce between 3 and {max_slides} slides.
2. Each slide has a short title (max 8 words), 2-4 short bullet points (max 12 words each), and a narration script (2-4 sentences) a voice narrator will read aloud for that slide.
3. The narration should read naturally when spoken — no bullet symbols, no markdown.
4. Return ONLY a valid JSON array. No markdown, no surrounding text.

Output format:
[
  {{"title": "Slide Title", "bullets": ["point one", "point two"], "narration": "Spoken narration text for this slide."}}
]"""


@router.post("/generate")
async def generate_video(body: GenerateVideoBody):
    if len(body.lesson_content.strip()) < 20:
        raise HTTPException(status_code=422, detail="lesson_content is too short to generate a video from")

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=VIDEO_SCRIPT_SYSTEM_PROMPT.format(max_slides=body.max_slides),
                user_prompt=f"Lesson Title: {body.lesson_title}\n\nLesson Content:\n{body.lesson_content}",
                temperature=0.3,
                max_tokens=2048,
                json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        slides = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    if not isinstance(slides, list) or len(slides) == 0:
        raise HTTPException(status_code=502, detail="LLM did not return a usable slide script")

    slides = slides[: body.max_slides]
    for slide in slides:
        slide.setdefault("bullets", [])
        slide.setdefault("narration", slide.get("title", ""))

    try:
        video_bytes = await asyncio.to_thread(build_video, slides)
    except Exception as e:
        logger.exception("Video assembly failed")
        raise HTTPException(status_code=502, detail=f"Video assembly failed: {e}")

    return Response(content=video_bytes, media_type="video/mp4")
