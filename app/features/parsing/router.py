"""Parsing — FastAPI router."""

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.shared.parsers.cv_parser import SUPPORTED_MIME_TYPES, cv_parser

router = APIRouter()


@router.post("/cv")
async def parse_cv(
    file: UploadFile = File(...),
    mime_type: str = Form(...),
):
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported mime type: {mime_type}")

    suffix = "." + SUPPORTED_MIME_TYPES[mime_type]
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        suffix = ".docx"
    elif mime_type == "application/pdf":
        suffix = ".pdf"
    else:
        suffix = ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = await cv_parser.parse(tmp_path, mime_type)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


class ParseFilePathBody(BaseModel):
    file_path: str
    mime_type: str


@router.post("/cv-path")
async def parse_cv_by_path(body: ParseFilePathBody):
    if body.mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported mime type: {body.mime_type}")

    try:
        text = await cv_parser.parse(body.file_path, body.mime_type)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
