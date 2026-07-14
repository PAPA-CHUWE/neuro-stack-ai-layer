from fastapi import APIRouter
from pydantic import BaseModel

from app.validators.json_schema import (
    validate,
    validate_cv_extraction,
    validate_feedback_draft,
    validate_gap_narrative,
)

router = APIRouter()


class ValidateBody(BaseModel):
    data: dict | list
    schema: dict


@router.post("/")
async def validate_generic(body: ValidateBody):
    return validate(body.data, body.schema)


@router.post("/cv-extraction")
async def validate_cv(body: dict | list):
    return validate_cv_extraction(body)


@router.post("/gap-narrative")
async def validate_gap(body: dict):
    return validate_gap_narrative(body)


@router.post("/feedback")
async def validate_feedback(body: dict):
    return validate_feedback_draft(body)
