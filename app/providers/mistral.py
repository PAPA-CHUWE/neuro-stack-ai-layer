import logging

import httpx

from app.config import settings
from app.providers.base import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LlmProvider,
)

logger = logging.getLogger(__name__)


class MistralProvider(LlmProvider):
    def __init__(self):
        self._cfg = settings.mistral
        if not self._cfg.api_key:
            logger.warning("MISTRAL_API_KEY is not set — LLM calls will fail")

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if not self._cfg.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")

        body: dict = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(
                self._cfg.api_url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._cfg.api_key}",
                },
            )

        if res.status_code != 200:
            logger.error("Mistral API error %s: %s", res.status_code, res.text)
            raise RuntimeError(f"Mistral completion failed: {res.status_code}")

        data = res.json()
        choice = data["choices"][0]
        return CompletionResponse(
            content=choice["message"]["content"],
            model=data["model"],
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if not self._cfg.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")

        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(
                self._cfg.embed_url,
                json={
                    "model": request.model or self._cfg.embed_model,
                    "input": request.input,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._cfg.api_key}",
                },
            )

        if res.status_code != 200:
            logger.error("Mistral embed error %s: %s", res.status_code, res.text)
            raise RuntimeError(f"Mistral embed failed: {res.status_code}")

        data = res.json()
        return EmbeddingResponse(
            embedding=data["data"][0]["embedding"],
            model=data["model"],
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
        )


llm_provider = MistralProvider()
