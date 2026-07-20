import json
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.features.course_metadata.router import router
from app.main import app as fastapi_app


client = TestClient(fastapi_app)


VALID_METADATA = {
    "courseCode": "CCS-101",
    "shortDescription": "This course covers cloud security fundamentals including IAM, encryption, and network security best practices for enterprise environments.",
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
}


class TestGenerateCourseMetadataRouter:
    def test_success(self):
        mock_result = AsyncMock()
        mock_result.course_code = "CCS-101"
        mock_result.short_description = VALID_METADATA["shortDescription"]
        mock_result.difficulty = "BEGINNER"
        mock_result.category = "Cybersecurity"
        mock_result.learning_objectives = VALID_METADATA["learningObjectives"]
        mock_result.skills_covered = VALID_METADATA["skillsCovered"]
        mock_result.estimated_duration_hours = 4.5
        mock_result.recommended_prerequisites = "Basic networking knowledge"
        mock_result.suggested_tags = ["cloud", "security", "beginner"]

        with patch("app.features.course_metadata.router.get_ai_provider") as mock_get:
            mock_provider = AsyncMock()
            mock_provider.generate_course_metadata.return_value = mock_result
            mock_get.return_value = mock_provider

            response = client.post(
                "/api/v1/ai/courses/generate-metadata",
                json={"tenantId": "tenant-1", "courseTitle": "Introduction to Cloud Security"},
                headers={"X-Tenant-Id": "tenant-1"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["courseCode"] == "CCS-101"
        assert data["difficulty"] == "BEGINNER"
        assert data["category"] == "Cybersecurity"
        assert "generatedAt" in data
        assert data["provider"] == "mistral"

    def test_title_too_short(self):
        response = client.post(
            "/api/v1/ai/courses/generate-metadata",
            json={"tenantId": "tenant-1", "courseTitle": "AB"},
            headers={"X-Tenant-Id": "tenant-1"},
        )

        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "INVALID_INPUT"

    def test_ai_generation_failure(self):
        with patch("app.features.course_metadata.router.get_ai_provider") as mock_get:
            mock_provider = AsyncMock()
            mock_provider.generate_course_metadata.side_effect = RuntimeError("AI down")
            mock_get.return_value = mock_provider

            response = client.post(
                "/api/v1/ai/courses/generate-metadata",
                json={"tenantId": "tenant-1", "courseTitle": "Introduction to Cloud Security"},
                headers={"X-Tenant-Id": "tenant-1"},
            )

        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "AI_GENERATION_FAILED"

    def test_schema_validation_failure(self):
        mock_result = AsyncMock()
        mock_result.course_code = "CCS-101"
        mock_result.short_description = VALID_METADATA["shortDescription"]
        mock_result.difficulty = "BEGINNER"
        mock_result.category = "Cybersecurity"
        mock_result.learning_objectives = VALID_METADATA["learningObjectives"]
        mock_result.skills_covered = VALID_METADATA["skillsCovered"]
        mock_result.estimated_duration_hours = 4.5
        mock_result.recommended_prerequisites = "Basic networking knowledge"
        mock_result.suggested_tags = ["cloud", "security", "beginner"]

        with patch("app.features.course_metadata.router.get_ai_provider") as mock_get:
            mock_provider = AsyncMock()
            mock_provider.generate_course_metadata.side_effect = RuntimeError("SCHEMA_VALIDATION_FAILED")
            mock_get.return_value = mock_provider

            response = client.post(
                "/api/v1/ai/courses/generate-metadata",
                json={"tenantId": "tenant-1", "courseTitle": "Introduction to Cloud Security"},
                headers={"X-Tenant-Id": "tenant-1"},
            )

        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "SCHEMA_VALIDATION_FAILED"

    def test_rate_limit(self):
        with patch("app.features.course_metadata.router._check_rate_limit") as mock_rate:
            mock_rate.return_value = False

            response = client.post(
                "/api/v1/ai/courses/generate-metadata",
                json={"tenantId": "tenant-1", "courseTitle": "Introduction to Cloud Security"},
                headers={"X-Tenant-Id": "tenant-1"},
            )

        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "RATE_LIMIT_EXCEEDED"
