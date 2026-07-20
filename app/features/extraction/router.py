"""Extraction — CV skill extraction, gap narratives, instructor feedback, skill gap analysis, weak topics."""

import json
import logging

from fastapi import APIRouter, HTTPException

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.prompts.cv_extraction import (
    CV_EXTRACTION_SYSTEM_PROMPT, CV_EXTRACTION_USER_TEMPLATE, CV_EXTRACTION_JSON_SCHEMA,
)
from app.shared.prompts.gap_narrative import (
    GapData, GAP_NARRATIVE_SYSTEM_PROMPT, GAP_NARRATIVE_USER_TEMPLATE, GAP_NARRATIVE_JSON_SCHEMA,
)
from app.shared.prompts.instructor_feedback import (
    FeedbackData, INSTRUCTOR_FEEDBACK_SYSTEM_PROMPT, INSTRUCTOR_FEEDBACK_USER_TEMPLATE,
    INSTRUCTOR_FEEDBACK_JSON_SCHEMA,
)
from app.shared.utils.json_parse import extract_json
from app.validators.json_schema import validate
from app.features.extraction.schemas import (
    CvExtractBody, CvExtractResponse,
    GapNarrativeBody, GapNarrativeResponse,
    InstructorFeedbackBody, InstructorFeedbackResponse,
    SkillGapAnalysisBody, SkillGapAnalysisResponse,
    WeakTopicBody, WeakTopicAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/cv", response_model=CvExtractResponse)
async def extract_cv_skills(body: CvExtractBody):
    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=CV_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=CV_EXTRACTION_USER_TEMPLATE.format(cv_text=body.cv_text),
                temperature=0.1, max_tokens=2048, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse CV extraction response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    vr = validate(data, CV_EXTRACTION_JSON_SCHEMA)
    if not vr["valid"]:
        logger.warning("CV extraction schema invalid: %s", vr["errors"])

    return CvExtractResponse(
        skills=data if isinstance(data, list) else [],
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        validation=vr,
    )


@router.post("/gap-narrative", response_model=GapNarrativeResponse)
async def generate_gap_narrative(body: GapNarrativeBody):
    gap_data = GapData(
        role_name=body.target_role,
        current_skills=body.strengths,
        required_skills=[],
        gaps=body.gaps,
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=GAP_NARRATIVE_SYSTEM_PROMPT,
                user_prompt=GAP_NARRATIVE_USER_TEMPLATE.format(
                    role_name=gap_data.role_name,
                    current_skills=json.dumps(gap_data.current_skills, indent=2),
                    required_skills="N/A",
                    gaps=json.dumps(gap_data.gaps, indent=2),
                ),
                temperature=0.3, max_tokens=2048, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse gap narrative response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    vr = validate(data, GAP_NARRATIVE_JSON_SCHEMA)

    return GapNarrativeResponse(
        narrative=data, model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, validation=vr,
    )


@router.post("/feedback", response_model=InstructorFeedbackResponse)
async def generate_instructor_feedback(body: InstructorFeedbackBody):
    from app.shared.prompts.instructor_feedback import Question

    questions = [Question(**q) for q in body.questions]
    feedback_data = FeedbackData(
        learner_name=body.learner_name,
        assessment_title=body.assessment_title,
        score=body.score,
        questions=questions,
    )

    questions_text = "\n".join(
        f"Q{i+1}: {q.text} (correct: {q.correct_answer or 'N/A'})"
        for i, q in enumerate(questions)
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=INSTRUCTOR_FEEDBACK_SYSTEM_PROMPT,
                user_prompt=INSTRUCTOR_FEEDBACK_USER_TEMPLATE.format(
                    learner_name=feedback_data.learner_name,
                    assessment_title=feedback_data.assessment_title,
                    score=feedback_data.score,
                    questions_text=questions_text,
                ),
                temperature=0.3, max_tokens=2048, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse feedback response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    vr = validate(data, INSTRUCTOR_FEEDBACK_JSON_SCHEMA)

    return InstructorFeedbackResponse(
        feedback=data, model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, validation=vr,
    )


SKILL_GAP_SYSTEM_PROMPT = """You are a skill gap analysis engine for an LMS platform.

Given a candidate's CV/resume text and a target role, produce a structured skill gap analysis.

Rules:
1. Compare the candidate's current skills (estimated from CV) against the target role's requirements.
2. Only include skills where there IS a gap (current level < required level).
3. Identify strengths — skills where the candidate meets or exceeds the requirement.
4. Compute an overall readiness score (0-100) based on how many required skills the candidate has.
5. Provide actionable recommendations for each gap.
6. Return ONLY valid JSON. No markdown, no conversational text.

Output format:
{
  "gaps": [{"skill": "Skill Name", "skillId": "camelCaseId", "currentLevel": "Beginner|Intermediate|Advanced|Expert", "requiredLevel": "Beginner|Intermediate|Advanced|Expert", "gap": 1}],
  "strengths": [{"skill": "Skill Name", "skillId": "camelCaseId", "currentLevel": "Intermediate"}],
  "overallScore": 65,
  "recommendations": [{"skill": "Skill Name", "action": "Enroll in course X", "priority": "high|medium|low"}]
}"""


@router.post("/skill-gap", response_model=SkillGapAnalysisResponse)
async def analyze_skill_gap(body: SkillGapAnalysisBody):
    role_ctx = ""
    if body.role_requirements:
        role_ctx = "\n\nRole requirements:\n" + "\n".join(
            f"- {r.get('skill', 'Unknown')}: required level {r.get('requiredLevel', 'Intermediate')}"
            for r in body.role_requirements
        )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=SKILL_GAP_SYSTEM_PROMPT,
                user_prompt=(
                    f"CV Text:\n{body.cv_text}\n\nTarget Role: {body.target_role}"
                    f"{role_ctx}\n\nReturn a JSON object with gaps, strengths, overallScore, and recommendations."
                ),
                temperature=0.1, max_tokens=2048, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse skill gap response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return SkillGapAnalysisResponse(
        gaps=data.get("gaps", []),
        strengths=data.get("strengths", []),
        overall_score=data.get("overallScore", 0),
        recommendations=data.get("recommendations", []),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


WEAK_TOPIC_SYSTEM_PROMPT = """You are an educational analytics engine. Analyze a learner's assessment results to identify weak topics.

Rules:
1. Group questions by topic.
2. For each topic, compute the average score and count of questions.
3. Determine mastery level: "mastered" (>80%), "developing" (50-80%), "needs_work" (<50%).
4. Return topics sorted by mastery level (needs_work first).
5. Return ONLY valid JSON.

Output format:
{
  "weak_topics": [{"topic": "Topic Name", "averageScore": 45.0, "questionCount": 3, "masteryLevel": "needs_work"}]
}"""


@router.post("/weak-topics", response_model=WeakTopicAnalysisResponse)
async def analyze_weak_topics(body: WeakTopicBody):
    questions_text = "\n".join(
        f"Q{i+1} [{q.get('topic', 'unknown')}]: correct={q.get('isCorrect', False)} score={q.get('score', 0)}"
        for i, q in enumerate(body.questions)
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=WEAK_TOPIC_SYSTEM_PROMPT,
                user_prompt=f"Assessment results:\n{questions_text}",
                temperature=0.1, max_tokens=1024, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse weak topics response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return WeakTopicAnalysisResponse(
        weak_topics=data.get("weak_topics", []),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
