"""LLM — FastAPI router."""

from fastapi import APIRouter, HTTPException

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.features.llm.schemas import CompleteBody, CompleteResponse

router = APIRouter()


@router.post("/complete", response_model=CompleteResponse)
async def complete(body: CompleteBody):
    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=body.system_prompt,
                user_prompt=body.user_prompt,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                json_mode=body.json_mode,
            )
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
