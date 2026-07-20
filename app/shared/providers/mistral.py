import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.shared.providers.base import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LlmProvider,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = [0.5, 1.0, 2.0]
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


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

        last_err = None
        for attempt in range(MAX_RETRIES):
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    res = await client.post(
                        self._cfg.api_url,
                        json=body,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self._cfg.api_key}",
                        },
                    )

                latency_ms = round((time.monotonic() - start) * 1000, 1)

                if res.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "Mistral %s retryable status %d (attempt %d/%d) latency_ms=%.1f",
                        "complete", res.status_code, attempt + 1, MAX_RETRIES, latency_ms,
                    )
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue

                if res.status_code != 200:
                    logger.error(
                        "Mistral complete error %d: %s latency_ms=%.1f",
                        res.status_code, res.text[:500], latency_ms,
                    )
                    raise RuntimeError(f"Mistral completion failed: {res.status_code}")

                data = res.json()
                choice = data["choices"][0]
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                logger.info(
                    "Mistral complete model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%.1f",
                    data.get("model", "unknown"), prompt_tokens, completion_tokens, latency_ms,
                )
                return CompletionResponse(
                    content=choice["message"]["content"],
                    model=data["model"],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except httpx.TimeoutException as exc:
                last_err = exc
                logger.warning(
                    "Mistral complete timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
            except httpx.ConnectError as exc:
                last_err = exc
                logger.warning(
                    "Mistral connect error (attempt %d/%d)", attempt + 1, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue

        raise RuntimeError(f"Mistral completion failed after {MAX_RETRIES} retries: {last_err}")

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if not self._cfg.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")

        last_err = None
        for attempt in range(MAX_RETRIES):
            start = time.monotonic()
            try:
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

                latency_ms = round((time.monotonic() - start) * 1000, 1)

                if res.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "Mistral %s retryable status %d (attempt %d/%d) latency_ms=%.1f",
                        "embed", res.status_code, attempt + 1, MAX_RETRIES, latency_ms,
                    )
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue

                if res.status_code != 200:
                    logger.error(
                        "Mistral embed error %d: %s latency_ms=%.1f",
                        res.status_code, res.text[:500], latency_ms,
                    )
                    raise RuntimeError(f"Mistral embed failed: {res.status_code}")

                data = res.json()
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                logger.info(
                    "Mistral embed model=%s prompt_tokens=%d latency_ms=%.1f",
                    data.get("model", "unknown"), prompt_tokens, latency_ms,
                )
                return EmbeddingResponse(
                    embedding=data["data"][0]["embedding"],
                    model=data["model"],
                    prompt_tokens=prompt_tokens,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_err = exc
                logger.warning(
                    "Mistral embed error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue

        raise RuntimeError(f"Mistral embed failed after {MAX_RETRIES} retries: {last_err}")

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
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
            "stream": True,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
                async with client.stream(
                    "POST",
                    self._cfg.api_url,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._cfg.api_key}",
                    },
                ) as res:
                    if res.status_code != 200:
                        text = await res.aread()
                        logger.error("Mistral stream error %d: %s", res.status_code, text[:500])
                        raise RuntimeError(f"Mistral stream failed: {res.status_code}")

                    async for line in res.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

            latency_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info("Mistral stream completed latency_ms=%.1f", latency_ms)
        except httpx.TimeoutException as exc:
            logger.error("Mistral stream timeout: %s", exc)
            raise RuntimeError(f"Mistral stream failed: {exc}") from exc
        except httpx.ConnectError as exc:
            logger.error("Mistral stream connect error: %s", exc)
            raise RuntimeError(f"Mistral stream failed: {exc}") from exc


llm_provider = MistralProvider()
