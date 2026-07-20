import json
import logging
import time
from typing import List

from app.features.course_metadata.application.prompt_builder import build_prompt, build_modules_prompt
from app.features.course_metadata.application.schema import COURSE_METADATA_SCHEMA, COURSE_MODULES_SCHEMA
from app.features.course_metadata.domain.entities import (
    CourseMetadataGenerationRequest,
    CourseMetadataResult,
    CourseModulesGenerationRequest,
    CourseModulesResult,
    GeneratedLesson,
    GeneratedModule,
)
from app.shared.providers.base import CompletionRequest
from app.shared.utils.json_parse import extract_json
from app.validators.json_schema import validate

logger = logging.getLogger(__name__)


def _match_category(generated: str, existing: List[str]) -> str:
    if not existing:
        return generated
    gen_lower = generated.lower()
    for cat in existing:
        if gen_lower == cat.lower() or gen_lower in cat.lower() or cat.lower() in gen_lower:
            return cat
    return generated


def _ensure_unique_code(code: str, existing: List[str]) -> str:
    if code not in existing:
        return code
    base = code.upper()
    suffix = 1
    while f"{base}-{suffix}" in existing:
        suffix += 1
    return f"{base}-{suffix}"


class GenerateCourseMetadataUseCase:
    def __init__(self, ai_provider):
        self._provider = ai_provider

    async def execute(
        self, request: CourseMetadataGenerationRequest
    ) -> CourseMetadataResult:
        start = time.monotonic()
        tenant_id = request.tenant_id
        title = request.course_title

        system_prompt, user_prompt = build_prompt(
            title, request.existing_categories, retry=False
        )

        try:
            result = await self._provider.complete(
                CompletionRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2,
                    max_tokens=1024,
                    json_mode=True,
                )
            )
        except Exception as exc:
            logger.error(
                "Course metadata generation failed for tenant=%s title=%r: %s",
                tenant_id, title, exc,
            )
            raise RuntimeError(
                "AI_GENERATION_FAILED"
            ) from exc

        try:
            data = extract_json(result.content)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse course metadata response for tenant=%s: %s",
                tenant_id, exc,
            )
            raise RuntimeError(
                "AI_GENERATION_FAILED"
            ) from exc

        validation = validate(data, COURSE_METADATA_SCHEMA)
        if not validation["valid"]:
            logger.warning(
                "Schema validation failed for tenant=%s, retrying: %s",
                tenant_id, validation["errors"],
            )
            system_prompt, user_prompt = build_prompt(
                title, request.existing_categories, retry=True
            )
            try:
                result = await self._provider.complete(
                    CompletionRequest(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.1,
                        max_tokens=1024,
                        json_mode=True,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Course metadata retry failed for tenant=%s: %s",
                    tenant_id, exc,
                )
                raise RuntimeError(
                    "SCHEMA_VALIDATION_FAILED"
                ) from exc

            try:
                data = extract_json(result.content)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error(
                    "Retry returned invalid JSON for tenant=%s: %s",
                    tenant_id, exc,
                )
                raise RuntimeError(
                    "SCHEMA_VALIDATION_FAILED"
                ) from exc

            validation = validate(data, COURSE_METADATA_SCHEMA)
            if not validation["valid"]:
                logger.error(
                    "Schema validation failed after retry for tenant=%s: %s",
                    tenant_id, validation["errors"],
                )
                raise RuntimeError(
                    "SCHEMA_VALIDATION_FAILED"
                )

        category = _match_category(data["category"], request.existing_categories)
        course_code = _ensure_unique_code(data["courseCode"], request.existing_course_codes)

        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "Course metadata generated tenant=%s title=%r provider=%s latency_ms=%.1f",
            tenant_id, title, result.model, latency_ms,
        )

        return CourseMetadataResult(
            course_code=course_code,
            short_description=data["shortDescription"],
            difficulty=data["difficulty"],
            category=category,
            learning_objectives=data["learningObjectives"],
            skills_covered=data["skillsCovered"],
            estimated_duration_hours=float(data["estimatedDurationHours"]),
            recommended_prerequisites=data.get("recommendedPrerequisites", ""),
            suggested_tags=data["suggestedTags"],
        )


class GenerateCourseModulesUseCase:
    def __init__(self, ai_provider):
        self._provider = ai_provider

    async def execute(
        self, request: CourseModulesGenerationRequest
    ) -> CourseModulesResult:
        start = time.monotonic()
        tenant_id = request.tenant_id
        title = request.course_title

        def _build(retry: bool) -> tuple[str, str]:
            return build_modules_prompt(
                title,
                short_description=request.short_description,
                category=request.category,
                difficulty=request.difficulty,
                learning_objectives=request.learning_objectives,
                skills_covered=request.skills_covered,
                retry=retry,
            )

        system_prompt, user_prompt = _build(retry=False)

        try:
            result = await self._provider.complete(
                CompletionRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=2048,
                    json_mode=True,
                )
            )
        except Exception as exc:
            logger.error(
                "Course modules generation failed for tenant=%s title=%r: %s",
                tenant_id, title, exc,
            )
            raise RuntimeError(
                "AI_GENERATION_FAILED"
            ) from exc

        try:
            data = extract_json(result.content)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse course modules response for tenant=%s: %s",
                tenant_id, exc,
            )
            raise RuntimeError(
                "AI_GENERATION_FAILED"
            ) from exc

        validation = validate(data, COURSE_MODULES_SCHEMA)
        if not validation["valid"]:
            logger.warning(
                "Modules schema validation failed for tenant=%s, retrying: %s",
                tenant_id, validation["errors"],
            )
            system_prompt, user_prompt = _build(retry=True)
            try:
                result = await self._provider.complete(
                    CompletionRequest(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.1,
                        max_tokens=2048,
                        json_mode=True,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Course modules retry failed for tenant=%s: %s",
                    tenant_id, exc,
                )
                raise RuntimeError(
                    "SCHEMA_VALIDATION_FAILED"
                ) from exc

            try:
                data = extract_json(result.content)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error(
                    "Retry returned invalid JSON for tenant=%s: %s",
                    tenant_id, exc,
                )
                raise RuntimeError(
                    "SCHEMA_VALIDATION_FAILED"
                ) from exc

            validation = validate(data, COURSE_MODULES_SCHEMA)
            if not validation["valid"]:
                logger.error(
                    "Modules schema validation failed after retry for tenant=%s: %s",
                    tenant_id, validation["errors"],
                )
                raise RuntimeError(
                    "SCHEMA_VALIDATION_FAILED"
                )

        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "Course modules generated tenant=%s title=%r provider=%s latency_ms=%.1f",
            tenant_id, title, result.model, latency_ms,
        )

        modules = [
            GeneratedModule(
                title=m["title"],
                description=m["description"],
                lessons=[
                    GeneratedLesson(
                        title=l["title"],
                        content_type=l["contentType"],
                        duration_minutes=l.get("durationMinutes"),
                    )
                    for l in m["lessons"]
                ],
            )
            for m in data["modules"]
        ]
        return CourseModulesResult(modules=modules)
