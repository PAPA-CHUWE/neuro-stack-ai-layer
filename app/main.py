from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.config import settings
from app.routes import llm, embeddings, parsing, validation

app = FastAPI(
    title="NeuroStack AI Layer",
    version="0.0.1",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(llm.router, prefix="/llm", tags=["LLM"])
app.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
app.include_router(parsing.router, prefix="/parsing", tags=["Parsing"])
app.include_router(validation.router, prefix="/validation", tags=["Validation"])


@app.get("/")
async def root():
    return {"service": "neurostack-ai-layer", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
