import json
from unittest.mock import AsyncMock
import pytest

from app.features.course_metadata.application.use_case import GenerateCourseMetadataUseCase
from app.features.course_metadata.domain.entities import (
    CourseMetadataGenerationRequest,
    CourseMetadataResult,
)
from app.shared.providers.base import CompletionRequest, CompletionResponse


def _make_provider(response_content: str, model: str = "test-model"):
    provider = AsyncMock()
    provider.complete.return_value = CompletionResponse(
        content=response_content,
        model=model,
        prompt_tokens=10,
        completion_tokens=20,
    )
    return provider


VALID_RESPONSE = json.dumps({
    "courseCode": "CCS-101",
    "shortDescription": "This course covers cloud security fundamentals including IAM, encryption, and network security best practices.",
    "difficulty": "BEGINNER",
    "category": "Cybersecurity",
    "learningObjectives": [
        "Understand IAM concepts",
        "Implement encryption",
        "Configure network security",
    ],
    "skillsCovered": ["IAM", "Encryption", "Network Security"],
    "estimatedDurationHours": 4.5,
    "recommendedPrerequisites": "Basic networking knowledge",
    "suggestedTags": ["cloud", "security", "beginner"],
})


class TestGenerateCourseMetadataUseCase:
    async def test_success(self):
        provider = _make_provider(VALID_RESPONSE)
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Introduction to Cloud Security",
            existing_categories=["Cybersecurity", "DevOps"],
            existing_course_codes=[],
        )

        result = await use_case.execute(request)

        assert result.course_code == "CCS-101"
        assert result.difficulty == "BEGINNER"
        assert result.category == "Cybersecurity"
        assert len(result.learning_objectives) == 3
        assert len(result.skills_covered) == 3
        assert result.estimated_duration_hours == 4.5
        provider.complete.assert_called_once()

    async def test_category_matching(self):
        provider = _make_provider(VALID_RESPONSE)
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Introduction to Cloud Security",
            existing_categories=["cybersecurity", "DevOps"],
            existing_course_codes=[],
        )

        result = await use_case.execute(request)
        assert result.category == "cybersecurity"

    async def test_category_fuzzy_match(self):
        response = json.dumps({
            **json.loads(VALID_RESPONSE),
            "category": "Cloud Security Advanced",
        })
        provider = _make_provider(response)
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Advanced Cloud Security",
            existing_categories=["Cloud Security"],
            existing_course_codes=[],
        )

        result = await use_case.execute(request)
        assert result.category == "Cloud Security"

    async def test_course_code_collision(self):
        response = json.dumps({
            **json.loads(VALID_RESPONSE),
            "courseCode": "CCS-101",
        })
        provider = _make_provider(response)
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Introduction to Cloud Security",
            existing_categories=[],
            existing_course_codes=["CCS-101", "CCS-102"],
        )

        result = await use_case.execute(request)
        assert result.course_code == "CCS-103"

    async def test_schema_validation_failure_then_retry_success(self):
        invalid_response = json.dumps({"courseCode": "CCS-101"})
        provider = _make_provider(invalid_response)
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Introduction to Cloud Security",
            existing_categories=[],
            existing_course_codes=[],
        )

        provider.complete.side_effect = [
            CompletionResponse(
                content=invalid_response,
                model="test-model",
                prompt_tokens=5,
                completion_tokens=5,
            ),
            CompletionResponse(
                content=VALID_RESPONSE,
                model="test-model",
                prompt_tokens=10,
                completion_tokens=20,
            ),
        ]

        result = await use_case.execute(request)
        assert result.course_code == "CCS-101"
        assert provider.complete.call_count == 2

    async def test_schema_validation_failure_then_retry_failure(self):
        provider = _make_provider("invalid json")
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Introduction to Cloud Security",
            existing_categories=[],
            existing_course_codes=[],
        )

        with pytest.raises(RuntimeError) as exc_info:
            await use_case.execute(request)

        assert "SCHEMA_VALIDATION_FAILED" in str(exc_info.value)

    async def test_provider_timeout(self):
        import httpx

        provider = AsyncMock()
        provider.complete.side_effect = httpx.TimeoutException("timeout")
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Introduction to Cloud Security",
            existing_categories=[],
            existing_course_codes=[],
        )

        with pytest.raises(RuntimeError) as exc_info:
            await use_case.execute(request)

        assert "AI_GENERATION_FAILED" in str(exc_info.value)

    async def test_provider_generic_error(self):
        provider = AsyncMock()
        provider.complete.side_effect = RuntimeError("provider down")
        use_case = GenerateCourseMetadataUseCase(provider)

        request = CourseMetadataGenerationRequest(
            tenant_id="tenant-1",
            course_title="Introduction to Cloud Security",
            existing_categories=[],
            existing_course_codes=[],
        )

        with pytest.raises(RuntimeError) as exc_info:
            await use_case.execute(request)

        assert "AI_GENERATION_FAILED" in str(exc_info.value)
