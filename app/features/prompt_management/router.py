"""Prompt Template Versioning — FastAPI router."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.shared.database import get_pg_pool
from app.features.prompt_management.schemas import PromptVersionCreate, PromptVersionUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


def _now():
    return datetime.now(timezone.utc)


@router.post("/versions")
async def create_prompt_version(body: PromptVersionCreate):
    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        max_version = await conn.fetchval(
            """SELECT MAX(version) FROM prompt_template_versions
               WHERE code = $1 AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))""",
            body.code, body.tenant_id,
        )
    next_version = (max_version or 0) + 1

    version_id = str(uuid.uuid4())
    now = _now()

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO prompt_template_versions
                   (id, tenant_id, code, version, name, description, content,
                    status, change_note, created_by, created_at, updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,'draft',$8,$9,$10,$11)""",
                version_id, body.tenant_id, body.code, next_version,
                body.name, body.description, body.content,
                body.change_note, body.created_by, now, now,
            )
    except Exception as e:
        logger.error("Failed to create prompt version: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create prompt version")

    return {"id": version_id, "version": next_version, "status": "draft"}


@router.get("/versions")
async def list_prompt_versions(
    tenant_id: str | None = None,
    code: str | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    pool = await get_pg_pool()
    conditions = []
    params: list = []
    idx = 1

    if tenant_id:
        conditions.append(f"tenant_id = ${idx}")
        params.append(tenant_id)
        idx += 1
    if code:
        conditions.append(f"code = ${idx}")
        params.append(code)
        idx += 1
    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT * FROM prompt_template_versions
        WHERE {where}
        ORDER BY code, version DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return {"items": [dict(r) for r in rows]}


@router.get("/versions/{version_id}")
async def get_prompt_version(version_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM prompt_template_versions WHERE id = $1", version_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return dict(row)


@router.get("/active/{code}")
async def get_active_prompt(code: str, tenant_id: str | None = None):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT * FROM prompt_template_versions
               WHERE code = $1 AND status = 'active'
                 AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))
               ORDER BY version DESC LIMIT 1""",
            code, tenant_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail=f"No active prompt for code '{code}'")
    return dict(row)


@router.patch("/versions/{version_id}/status")
async def update_prompt_status(version_id: str, new_status: str, approved_by: str | None = None):
    valid_transitions = {
        "draft": ["under_review"],
        "under_review": ["approved", "draft"],
        "approved": ["active"],
        "active": ["retired"],
        "retired": ["draft"],
    }

    pool = await get_pg_pool()
    now = _now()

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM prompt_template_versions WHERE id = $1", version_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Prompt version not found")

        current_status = existing["status"]
        if new_status not in valid_transitions.get(current_status, []):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot transition from '{current_status}' to '{new_status}'",
            )

        if new_status == "active":
            await conn.execute(
                """UPDATE prompt_template_versions
                   SET status = 'retired', retired_at = $1, updated_at = $1
                   WHERE code = $2 AND status = 'active' AND id != $3
                     AND (tenant_id = $4 OR (tenant_id IS NULL AND $4 IS NULL))""",
                now, existing["code"], version_id, existing["tenant_id"],
            )

        update_fields = {"status": new_status, "updated_at": now}
        if approved_by:
            update_fields["approved_by"] = approved_by
        if new_status == "active":
            update_fields["activated_at"] = now
        if new_status == "retired":
            update_fields["retired_at"] = now

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(update_fields))
        values = [version_id] + list(update_fields.values())

        await conn.execute(
            f"UPDATE prompt_template_versions SET {set_clause} WHERE id = $1", *values,
        )

    return {"ok": True, "status": new_status}


@router.patch("/versions/{version_id}")
async def update_prompt_version(version_id: str, body: PromptVersionUpdate):
    pool = await get_pg_pool()
    now = _now()

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM prompt_template_versions WHERE id = $1", version_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Prompt version not found")
        if existing["status"] not in ("draft", "under_review"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot edit prompt in '{existing['status']}' status",
            )

        updates = {}
        if body.name is not None:
            updates["name"] = body.name
        if body.description is not None:
            updates["description"] = body.description
        if body.content is not None:
            updates["content"] = body.content
        if body.change_note is not None:
            updates["change_note"] = body.change_note
        updates["updated_at"] = now

        set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates))
        values = [version_id] + list(updates.values())
        await conn.execute(
            f"UPDATE prompt_template_versions SET {set_clause} WHERE id = $1", *values,
        )

    return {"ok": True}


@router.delete("/versions/{version_id}")
async def delete_prompt_version(version_id: str):
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT status FROM prompt_template_versions WHERE id = $1", version_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Prompt version not found")
        if existing["status"] != "draft":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete prompt in '{existing['status']}' status",
            )
        await conn.execute(
            "DELETE FROM prompt_template_versions WHERE id = $1", version_id,
        )
    return {"ok": True}
