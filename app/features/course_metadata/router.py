"""Course Metadata — FastAPI router."""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Tuple

from fastapi import APIRouter, HTTPException, Request

from app.features.course_metadata.domain.entities import (
    CourseMetadataGenerationRequest,
    CourseModulesGenerationRequest,
)
from app.features.course_metadata.infrastructure.adapters import get_ai_provider
from app.features.course_metadata.schemas import (
    CourseModulesResponse,
    CourseMetadataResponse,
    ErrorResponse,
    GenerateCourseMetadataBody,
    GenerateCourseModulesBody,
    ModuleItem,
    LessonItem,
)

logger = logging.getLogger(__name__)

router = APIRouter()

RATE_LIMIT = 20
RATE_WINDOW_SECONDS = 60

_rate_limit_store: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0.0))


def _check_rate_limit(key: str) -> bool:
    now = time.time()
    count, window_start = _rate_limit_store[key]
    if now - window_start > RATE_WINDOW_SECONDS:
        _rate_limit_store[key] = (1, now)
        return True
    if count >= RATE_LIMIT:
        return False
    _rate_limit_store[key] = (count + 1, window_start)
    return True


@router.post(
    "/generate-metadata",
    response_model=CourseMetadataResponse,
    responses={
        502: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def generate_course_metadata(
    body: GenerateCourseMetadataBody, request: Request
):
    tenant_id = body.tenantId
    client_key = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")

    if not _check_rate_limit(client_key):
        raise HTTPException(
            status_code=429,
            detail={"error": "RATE_LIMIT_EXCEEDED", "message": "Too many requests. Please try again later."},
        )

    if len(body.courseTitle.strip()) < 4:
        raise HTTPException(
            status_code=422,
            detail={"error": "INVALID_INPUT", "message": "Course title must be at least 4 characters."},
        )

    provider = get_ai_provider()

    req = CourseMetadataGenerationRequest(
        tenant_id=tenant_id,
        course_title=body.courseTitle.strip(),
        existing_categories=[],
        existing_course_codes=[],
    )

    try:
        result = await provider.generate_course_metadata(req)
    except RuntimeError as exc:
        error_msg = str(exc)
        if error_msg == "AI_GENERATION_FAILED":
            raise HTTPException(
                status_code=502,
                detail={"error": "AI_GENERATION_FAILED", "message": "AI generation unavailable — you can enter these manually."},
            )
        if error_msg == "SCHEMA_VALIDATION_FAILED":
            raise HTTPException(
                status_code=422,
                detail={"error": "SCHEMA_VALIDATION_FAILED", "message": "AI returned invalid data after retry. Please try again."},
            )
        raise HTTPException(
            status_code=502,
            detail={"error": "AI_GENERATION_FAILED", "message": error_msg},
        )

    return CourseMetadataResponse(
        courseCode=result.course_code,
        shortDescription=result.short_description,
        difficulty=result.difficulty,
        category=result.category,
        learningObjectives=result.learning_objectives,
        skillsCovered=result.skills_covered,
        estimatedDurationHours=result.estimated_duration_hours,
        recommendedPrerequisites=result.recommended_prerequisites,
        suggestedTags=result.suggested_tags,
        generatedAt=datetime.now(timezone.utc).isoformat(),
        provider="mistral",
    )


@router.post(
    "/generate-modules",
    response_model=CourseModulesResponse,
    responses={
        502: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def generate_course_modules(
    body: GenerateCourseModulesBody, request: Request
):
    client_key = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")

    if not _check_rate_limit(client_key):
        raise HTTPException(
            status_code=429,
            detail={"error": "RATE_LIMIT_EXCEEDED", "message": "Too many requests. Please try again later."},
        )

    if len(body.courseTitle.strip()) < 4:
        raise HTTPException(
            status_code=422,
            detail={"error": "INVALID_INPUT", "message": "Course title must be at least 4 characters."},
        )

    provider = get_ai_provider()

    req = CourseModulesGenerationRequest(
        tenant_id=body.tenantId,
        course_title=body.courseTitle.strip(),
        short_description=body.shortDescription,
        category=body.category,
        difficulty=body.difficulty,
        learning_objectives=body.learningObjectives,
        skills_covered=body.skillsCovered,
    )

    try:
        result = await provider.generate_course_modules(req)
    except RuntimeError as exc:
        error_msg = str(exc)
        if error_msg == "AI_GENERATION_FAILED":
            raise HTTPException(
                status_code=502,
                detail={"error": "AI_GENERATION_FAILED", "message": "AI generation unavailable — you can build the outline manually."},
            )
        if error_msg == "SCHEMA_VALIDATION_FAILED":
            raise HTTPException(
                status_code=422,
                detail={"error": "SCHEMA_VALIDATION_FAILED", "message": "AI returned invalid data after retry. Please try again."},
            )
        raise HTTPException(
            status_code=502,
            detail={"error": "AI_GENERATION_FAILED", "message": error_msg},
        )

    return CourseModulesResponse(
        modules=[
            ModuleItem(
                title=m.title,
                description=m.description,
                lessons=[
                    LessonItem(
                        title=l.title,
                        contentType=l.content_type,
                        durationMinutes=l.duration_minutes,
                    )
                    for l in m.lessons
                ],
            )
            for m in result.modules
        ],
        generatedAt=datetime.now(timezone.utc).isoformat(),
        provider="mistral",
    )
