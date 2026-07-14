import os
from pydantic import BaseModel


class MistralConfig(BaseModel):
    api_key: str = os.getenv("MISTRAL_API_KEY", "")
    api_url: str = os.getenv(
        "MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions"
    )
    model: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    embed_url: str = os.getenv(
        "MISTRAL_EMBED_URL", "https://api.mistral.ai/v1/embeddings"
    )
    embed_model: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")


class ZvecConfig(BaseModel):
    base_path: str = os.getenv("ZVEC_BASE_PATH", "/tmp/zvec-data")


class MongoConfig(BaseModel):
    uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name: str = os.getenv("MONGODB_DB_NAME", "e-lms")


class Settings(BaseModel):
    mistral: MistralConfig = MistralConfig()
    zvec: ZvecConfig = ZvecConfig()
    mongo: MongoConfig = MongoConfig()
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")


settings = Settings()
