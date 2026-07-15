from __future__ import annotations

from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings


@lru_cache(maxsize=1)
def _get_client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo.uri)


async def get_db() -> AsyncIOMotorDatabase:
    return _get_client()[settings.mongo.db_name]
