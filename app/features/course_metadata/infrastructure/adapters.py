import logging
from typing import Optional

from app.features.course_metadata.application.use_case import (
    GenerateCourseMetadataUseCase,
    GenerateCourseModulesUseCase,
)
from app.features.course_metadata.domain.entities import (
    CourseMetadataGenerationRequest,
    CourseMetadataResult,
    CourseModulesGenerationRequest,
    CourseModulesResult,
)
from app.features.course_metadata.domain.ports import AiProvider
from app.shared.providers.base import CompletionRequest

logger = logging.getLogger(__name__)


class BaseLlmAiProvider(AiProvider):
    def __init__(self, provider_name: str, llm_provider):
        self._provider_name = provider_name
        self._llm = llm_provider

    async def generate_course_metadata(
        self, request: CourseMetadataGenerationRequest
    ) -> CourseMetadataResult:
        use_case = GenerateCourseMetadataUseCase(self._llm)
        return await use_case.execute(request)

    async def generate_course_modules(
        self, request: CourseModulesGenerationRequest
    ) -> CourseModulesResult:
        use_case = GenerateCourseModulesUseCase(self._llm)
        return await use_case.execute(request)


class MistralAiProvider(BaseLlmAiProvider):
    def __init__(self):
        from app.shared.providers.mistral import llm_provider
        super().__init__("mistral", llm_provider)


class OpenAiAiProvider(BaseLlmAiProvider):
    def __init__(self):
        # Placeholder: requires OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL
        # When implemented, instantiate an OpenAI LlmProvider here.
        raise NotImplementedError(
            "OpenAI provider not yet configured. Set OPENAI_API_KEY to enable."
        )


class AzureOpenAiAdapter(BaseLlmAiProvider):
    def __init__(self):
        # Placeholder: requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
        raise NotImplementedError(
            "Azure OpenAI provider not yet configured. Set AZURE_OPENAI_API_KEY to enable."
        )


class NvidiaNimAdapter(BaseLlmAiProvider):
    def __init__(self):
        # Placeholder: requires NVIDIA_NIM_API_KEY, NVIDIA_NIM_API_URL, NVIDIA_NIM_MODEL
        raise NotImplementedError(
            "NVIDIA NIM provider not yet configured. Set NVIDIA_NIM_API_KEY to enable."
        )


class OllamaAdapter(BaseLlmAiProvider):
    def __init__(self):
        # Placeholder: requires OLLAMA_API_URL, OLLAMA_MODEL
        raise NotImplementedError(
            "Ollama provider not yet configured. Set OLLAMA_API_URL to enable."
        )


_PROVIDER_MAP = {
    "mistral": MistralAiProvider,
    "openai": OpenAiAiProvider,
    "azure_openai": AzureOpenAiAdapter,
    "nvidia_nim": NvidiaNimAdapter,
    "ollama": OllamaAdapter,
}


def get_ai_provider() -> AiProvider:
    import os

    provider_name = os.getenv("AI_COURSE_METADATA_PROVIDER", "mistral").lower()
    provider_cls = _PROVIDER_MAP.get(provider_name)
    if provider_cls is None:
        available = ", ".join(sorted(_PROVIDER_MAP.keys()))
        raise RuntimeError(
            f"Unknown AI_COURSE_METADATA_PROVIDER={provider_name!r}. Available: {available}"
        )
    return provider_cls()
