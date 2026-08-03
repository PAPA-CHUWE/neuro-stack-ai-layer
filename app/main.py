import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from app.config import settings
from app.shared.middleware.request_context import RequestContextMiddleware
from app.features.llm.router import router as llm_router
from app.features.embeddings.router import router as embeddings_router
from app.features.parsing.router import router as parsing_router
from app.features.validation.router import router as validation_router
from app.features.skill_goals.router import router as skill_goals_router
from app.features.extraction.router import router as extraction_router
from app.features.recommendations.router import router as recommendations_router
from app.features.cv_extractions.router import router as cv_extractions_router
from app.features.knowledge.router import router as rag_router
from app.features.streaming.router import router as streaming_router
from app.features.mind.router import router as mind_router
from app.features.feedback.router import router as feedback_router
from app.features.traces.router import router as traces_router
from app.features.evaluations.router import router as evaluations_router
from app.features.prompt_management.router import router as prompts_router
from app.features.course_metadata.router import router as course_metadata_router
from app.features.video.router import router as video_router
from app.features.grading.router import router as grading_router
from app.features.career_readiness.router import router as career_readiness_router
from app.features.insights.router import router as insights_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.shared.database import init_pg_tables
        await init_pg_tables()
    except Exception as e:
        logging.getLogger(__name__).warning("Postgres init skipped: %s", e)
    yield


app = FastAPI(
    title="NeuroStack AI Layer",
    version="0.1.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(llm_router, prefix="/llm", tags=["LLM"])
app.include_router(embeddings_router, prefix="/embeddings", tags=["Embeddings"])
app.include_router(parsing_router, prefix="/parsing", tags=["Parsing"])
app.include_router(validation_router, prefix="/validation", tags=["Validation"])
app.include_router(skill_goals_router, prefix="/skill-goals", tags=["Skill Goals"])
app.include_router(extraction_router, prefix="/extraction", tags=["Extraction"])
app.include_router(recommendations_router, prefix="/recommendations", tags=["Recommendations"])
app.include_router(cv_extractions_router, prefix="/cv-extractions", tags=["CV Extractions"])
app.include_router(rag_router, prefix="/rag", tags=["RAG"])
app.include_router(streaming_router, prefix="/stream", tags=["Streaming"])
app.include_router(mind_router, prefix="/mind", tags=["Mind"])
app.include_router(feedback_router, prefix="/mind/feedback", tags=["Mind Feedback"])
app.include_router(traces_router, prefix="/traces", tags=["AI Traces"])
app.include_router(evaluations_router, prefix="/evaluations", tags=["AI Evaluations"])
app.include_router(prompts_router, prefix="/prompts", tags=["Prompt Versioning"])
app.include_router(course_metadata_router, prefix="/api/v1/ai/courses", tags=["Course Metadata"])
app.include_router(video_router, prefix="/video", tags=["Video"])
app.include_router(grading_router, prefix="/grading", tags=["Grading"])
app.include_router(career_readiness_router, prefix="/career-readiness", tags=["Career Readiness"])
app.include_router(insights_router, prefix="/insights", tags=["Insights"])


@app.get("/")
async def root():
    return {"service": "neurostack-ai-layer", "status": "ok", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness():
    mistral_ok = bool(settings.mistral.api_key)
    return {"status": "ready" if mistral_ok else "degraded", "mistral_configured": mistral_ok}
