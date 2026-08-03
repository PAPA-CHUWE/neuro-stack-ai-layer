"""Career Readiness Agent — FastAPI router.

Trigger: a learner's SkillGoal reaches 'achieved' (milestone reached).
Readiness level is classified deterministically upstream (NestJS) from real
assessment performance tied to the goal's courses — this agent only explains
the level and proposes a next stage; it never re-derives or second-guesses
the level itself.
"""

import json
import logging

from fastapi import APIRouter, HTTPException

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.utils.json_parse import extract_json
from app.features.career_readiness.schemas import AssessReadinessBody, AssessReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter()


ASSESS_READINESS_SYSTEM_PROMPT = """You are a career readiness agent for an LMS platform.

A learner has just completed every course in their learning plan for a target role.
The READINESS LEVEL has already been classified deterministically from their real
assessment performance — never restate it as uncertain or re-derive it yourself.

Rules:
1. Write a short (2-3 sentence) assessment addressed to the learner, explaining what
   their readiness level means, grounded in the specific signals provided.
2. Write a short (2-3 sentence) recommendation for their next stage — a specific next
   role, specialization, or focus area to pursue from here.
3. Be encouraging and specific, never generic ("keep learning!") — name real things
   grounded in the role and the courses they just completed.
4. Return ONLY valid JSON. No markdown, no conversational text.

Output format:
{
  "assessment": "...",
  "nextStage": "..."
}"""


@router.post("/assess", response_model=AssessReadinessResponse)
async def assess_readiness(body: AssessReadinessBody):
    user_prompt = (
        f"Target role just completed: {body.role_name}\n"
        f"Classified readiness level: {body.readiness_level}\n"
        f"Signals: {json.dumps(body.signals)}\n"
        f"Completed courses: {', '.join(body.course_titles) or '(none)'}\n\n"
        "Write the assessment and next-stage recommendation."
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=ASSESS_READINESS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3, max_tokens=512, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse assess-readiness response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return AssessReadinessResponse(
        assessment=data.get("assessment", ""),
        next_stage=data.get("nextStage", ""),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
