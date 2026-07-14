from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from providers.base import CompletionRequest
from providers.mistral import llm_provider

router = APIRouter()


class CompleteBody(BaseModel):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False


class CompleteResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


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
