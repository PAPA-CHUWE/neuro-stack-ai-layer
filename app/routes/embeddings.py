from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.providers.base import EmbeddingRequest
from app.providers.mistral import llm_provider
from app.providers.vector_store import vector_store, CollectionConfig, Document

router = APIRouter()


class EmbedBody(BaseModel):
    input: str
    model: str | None = None


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    prompt_tokens: int


@router.post("/", response_model=EmbedResponse)
async def embed(body: EmbedBody):
    try:
        return await llm_provider.embed(
            EmbeddingRequest(input=body.input, model=body.model)
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


class SearchBody(BaseModel):
    tenant_id: str
    config: dict
    query_vector: list[float]
    top_k: int = 10
    filter: str | None = None


@router.post("/search")
async def search(body: SearchBody):
    try:
        cfg = CollectionConfig(**body.config)
        results = await vector_store.search(
            body.tenant_id, cfg, body.query_vector, body.top_k, body.filter
        )
        return {"results": [r.model_dump() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpsertBody(BaseModel):
    tenant_id: str
    config: dict
    documents: list[dict]


@router.post("/upsert")
async def upsert(body: UpsertBody):
    try:
        cfg = CollectionConfig(**body.config)
        docs = [Document(**d) for d in body.documents]
        await vector_store.upsert(body.tenant_id, cfg, docs)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteBody(BaseModel):
    tenant_id: str
    collection_name: str
    ids: list[str]


@router.post("/delete")
async def delete_documents(body: DeleteBody):
    try:
        await vector_store.delete_documents(body.tenant_id, body.collection_name, body.ids)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteCollectionBody(BaseModel):
    tenant_id: str
    collection_name: str


@router.delete("/collection")
async def delete_collection(body: DeleteCollectionBody):
    try:
        await vector_store.delete_collection(body.tenant_id, body.collection_name)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
