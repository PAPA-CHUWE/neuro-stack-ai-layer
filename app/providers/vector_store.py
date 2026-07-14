import json
import logging
import shutil
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


class CollectionConfig(BaseModel):
    name: str
    dimension: int
    metric: str = "cosine"


class Document(BaseModel):
    id: str
    vector: list[float]
    fields: dict | None = None


class SearchResult(BaseModel):
    id: str
    score: float
    fields: dict | None = None


class VectorStore:
    def __init__(self):
        self._base = Path(settings.zvec.base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, _Collection] = {}

    def _tenant_dir(self, tenant_id: str) -> Path:
        return self._base / f"tenant_{tenant_id}"

    def _coll_dir(self, tenant_id: str, name: str) -> Path:
        return self._tenant_dir(tenant_id) / name

    def _get_or_create(self, tenant_id: str, config: CollectionConfig) -> "_Collection":
        key = f"{tenant_id}:{config.name}"
        if key in self._collections:
            return self._collections[key]

        coll_dir = self._coll_dir(tenant_id, config.name)
        coll_dir.mkdir(parents=True, exist_ok=True)
        coll = _Collection(coll_dir, config)
        self._collections[key] = coll
        return coll

    async def upsert(
        self, tenant_id: str, config: CollectionConfig, documents: list[Document]
    ):
        coll = self._get_or_create(tenant_id, config)
        coll.upsert(documents)

    async def search(
        self,
        tenant_id: str,
        config: CollectionConfig,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        coll = self._get_or_create(tenant_id, config)
        return coll.search(query_vector, top_k, filter_expr)

    async def delete_documents(self, tenant_id: str, collection_name: str, ids: list[str]):
        key = f"{tenant_id}:{collection_name}"
        if key in self._collections:
            self._collections[key].delete(ids)

    async def delete_collection(self, tenant_id: str, collection_name: str):
        key = f"{tenant_id}:{collection_name}"
        self._collections.pop(key, None)
        coll_dir = self._coll_dir(tenant_id, collection_name)
        if coll_dir.exists():
            shutil.rmtree(coll_dir)


class _Collection:
    def __init__(self, path: Path, config: CollectionConfig):
        self._path = path
        self._config = config
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._fields: list[dict | None] = []
        self._load()

    def _load(self):
        ids_path = self._path / "ids.json"
        vectors_path = self._path / "vectors.npy"
        fields_path = self._path / "fields.json"

        if ids_path.exists():
            self._ids = json.loads(ids_path.read_text())
        if vectors_path.exists():
            self._vectors = np.load(str(vectors_path))
        if fields_path.exists():
            self._fields = json.loads(fields_path.read_text())

    def _save(self):
        (self._path / "ids.json").write_text(json.dumps(self._ids))
        if self._vectors is not None:
            np.save(str(self._path / "vectors.npy"), self._vectors)
        (self._path / "fields.json").write_text(json.dumps(self._fields))

    def upsert(self, documents: list[Document]):
        for doc in documents:
            if doc.id in self._ids:
                idx = self._ids.index(doc.id)
                if self._vectors is not None:
                    self._vectors[idx] = np.array(doc.vector, dtype=np.float32)
                self._fields[idx] = doc.fields
            else:
                self._ids.append(doc.id)
                vec = np.array([doc.vector], dtype=np.float32)
                if self._vectors is None:
                    self._vectors = vec
                else:
                    self._vectors = np.vstack([self._vectors, vec])
                self._fields.append(doc.fields)
        self._save()

    def search(
        self, query_vector: list[float], top_k: int, filter_expr: str | None
    ) -> list[SearchResult]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = self._vectors / norms

        scores = normalized @ query
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            fields = self._fields[idx] if idx < len(self._fields) else None
            if filter_expr and fields and not self._eval_filter(filter_expr, fields):
                continue
            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=float(scores[idx]),
                    fields=fields,
                )
            )
        return results

    def delete(self, ids: list[str]):
        keep = [i for i, id_ in enumerate(self._ids) if id_ not in ids]
        self._ids = [self._ids[i] for i in keep]
        if self._vectors is not None and len(keep) > 0:
            self._vectors = self._vectors[keep]
        else:
            self._vectors = None
        self._fields = [self._fields[i] for i in keep]
        self._save()

    @staticmethod
    def _eval_filter(expr: str, fields: dict) -> bool:
        try:
            if "==" in expr:
                left, right = expr.split("==", 1)
                key = left.strip()
                val = right.strip().strip("'\"")
                return fields.get(key) == val
            return True
        except Exception:
            return True


vector_store = VectorStore()
