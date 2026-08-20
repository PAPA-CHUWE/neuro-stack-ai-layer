from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pydantic import BaseModel


class CompletionRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False
    messages: list[dict] | None = None


class CompletionResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class EmbeddingRequest(BaseModel):
    input: str
    model: str | None = None


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    model: str
    prompt_tokens: int


class LlmProvider(ABC):
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        result = await self.complete(request)
        yield result.content
