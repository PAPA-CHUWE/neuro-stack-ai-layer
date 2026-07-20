from abc import ABC, abstractmethod

from app.features.course_metadata.domain.entities import (
    CourseMetadataGenerationRequest,
    CourseMetadataResult,
    CourseModulesGenerationRequest,
    CourseModulesResult,
)


class AiProvider(ABC):
    @abstractmethod
    async def generate_course_metadata(
        self, request: CourseMetadataGenerationRequest
    ) -> CourseMetadataResult:
        raise NotImplementedError

    @abstractmethod
    async def generate_course_modules(
        self, request: CourseModulesGenerationRequest
    ) -> CourseModulesResult:
        raise NotImplementedError
