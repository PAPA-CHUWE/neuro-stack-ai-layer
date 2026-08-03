"""Insights Agent — FastAPI router.

Cohort-level reporting for administrators. Every metric is computed
deterministically upstream (NestJS) from real Risk Intervention, Career
Readiness, and Behavioral Memory data — this agent only narrates the
numbers it's given; it never invents or re-derives a statistic itself.
"""

import json
import logging

from fastapi import APIRouter, HTTPException

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.utils.json_parse import extract_json
from app.features.insights.schemas import SummarizeCohortBody, SummarizeCohortResponse

logger = logging.getLogger(__name__)

router = APIRouter()


SUMMARIZE_COHORT_SYSTEM_PROMPT = """You are a cohort insights agent for an LMS platform's admin dashboard.

You are given real, pre-computed metrics about a cohort of learners. Never invent
a number that isn't in the provided metrics, and never restate a metric as
uncertain — it was computed deterministically, not guessed.

Rules:
1. Write ONE headline sentence using a specific real number from the metrics
   (e.g. "18% of active learners are currently flagged as at risk").
2. Write a short narrative (2-4 sentences) highlighting what the admin should
   pay attention to first, grounded in the specific metrics provided.
3. If a metric signals a genuine problem (high at-risk rate, low readiness),
   say so plainly — don't soften real risk into vague positivity.
4. If the cohort is small or metrics are sparse, say so rather than
   overstating confidence in a small sample.
5. Return ONLY valid JSON. No markdown, no conversational text.

Output format:
{
  "headline": "...",
  "narrative": "..."
}"""


@router.post("/summarize", response_model=SummarizeCohortResponse)
async def summarize_cohort(body: SummarizeCohortBody):
    user_prompt = f"Cohort metrics:\n{json.dumps(body.metrics, indent=2)}\n\nWrite the headline and narrative."

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=SUMMARIZE_COHORT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2, max_tokens=512, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse cohort summary response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return SummarizeCohortResponse(
        headline=data.get("headline", ""),
        narrative=data.get("narrative", ""),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
