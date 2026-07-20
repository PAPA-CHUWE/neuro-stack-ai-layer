"""CV Extractions — FastAPI router."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.shared.database import get_pg_pool
from app.features.cv_extractions.schemas import SaveExtractionBody, ExtractionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _now():
    return datetime.now(timezone.utc)


def _row_to_response(row) -> ExtractionResponse:
    return ExtractionResponse(
        id=row["id"], user_id=row["user_id"], file_name=row["file_name"],
        mime_type=row["mime_type"], cv_text=row["cv_text"],
        skills=json.loads(row["skills"]) if isinstance(row["skills"], str) else row["skills"],
        model=row["model"], prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        validation=json.loads(row["validation"]) if isinstance(row["validation"], str) else row["validation"],
        created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
    )


async def _get_extraction(pool, extraction_id: str, user_id: str) -> ExtractionResponse:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM cv_extractions WHERE id = $1 AND user_id = $2",
            extraction_id, user_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Extraction not found")
    return _row_to_response(row)


@router.post("", response_model=ExtractionResponse)
async def save_extraction(body: SaveExtractionBody):
    pool = await get_pg_pool()
    extraction_id = str(uuid.uuid4())
    now = _now()

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO cv_extractions
               (id, user_id, file_name, mime_type, cv_text, skills, model, prompt_tokens, completion_tokens, validation, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
            extraction_id, body.user_id, body.file_name, body.mime_type, body.cv_text,
            json.dumps(body.skills), body.model, body.prompt_tokens, body.completion_tokens,
            json.dumps(body.validation or {}), now,
        )

    return await _get_extraction(pool, extraction_id, body.user_id)


@router.get("", response_model=list[ExtractionResponse])
async def list_extractions(user_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM cv_extractions WHERE user_id = $1 ORDER BY created_at DESC", user_id,
        )
    return [_row_to_response(r) for r in rows]


@router.get("/{extraction_id}", response_model=ExtractionResponse)
async def get_extraction(extraction_id: str, user_id: str):
    pool = await get_pg_pool()
    return await _get_extraction(pool, extraction_id, user_id)


@router.delete("/{extraction_id}")
async def delete_extraction(extraction_id: str, user_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM cv_extractions WHERE id = $1 AND user_id = $2",
            extraction_id, user_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Extraction not found")
    return {"ok": True}
