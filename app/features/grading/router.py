"""Grading — FastAPI router. LLM-graded scoring + explanation for subjective answers."""

import json
import logging

from fastapi import APIRouter, HTTPException

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.utils.json_parse import extract_json
from app.features.grading.schemas import AnswerGradingBody, AnswerGradingResponse

logger = logging.getLogger(__name__)

router = APIRouter()


ANSWER_GRADING_SYSTEM_PROMPT = """You are an assessment grading engine for an LMS platform.

Given a question, an optional model/rubric answer, and a learner's response, grade the response.

Rules:
1. If a model/rubric answer is provided, grade how well the learner's response matches its meaning
   (not exact wording — accept paraphrases and equivalent reasoning).
2. If no model/rubric answer is provided, grade the response on correctness and understanding of the question.
3. Award partial credit when the response is partially correct.
4. Write a short, specific explanation (2-4 sentences) of why the learner earned or lost points.
   Address the learner directly ("You correctly identified..." / "You missed...").
5. Return ONLY valid JSON. No markdown, no conversational text.

Output format:
{
  "isCorrect": true,
  "pointsAwarded": 2.5,
  "explanation": "You correctly identified X, but missed Y which is required for full credit."
}"""


@router.post("/answer", response_model=AnswerGradingResponse)
async def grade_answer(body: AnswerGradingBody):
    user_prompt = (
        f"Question: {body.question_text}\n"
        f"Question type: {body.question_type}\n"
        f"Model/rubric answer: {body.correct_answer or 'N/A — grade on understanding'}\n"
        f"Max points: {body.max_points}\n"
    )
    if body.course_context:
        user_prompt += f"Course context:\n{body.course_context}\n"
    user_prompt += (
        f"Learner's response: {body.learner_response}\n\n"
        "Return JSON: { isCorrect, pointsAwarded, explanation }."
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=ANSWER_GRADING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1, max_tokens=512, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse answer grading response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    points_awarded = min(max(float(data.get("pointsAwarded", 0)), 0), body.max_points)

    return AnswerGradingResponse(
        is_correct=bool(data.get("isCorrect", False)),
        points_awarded=points_awarded,
        explanation=data.get("explanation", ""),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
