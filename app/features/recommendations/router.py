"""Recommendations — FastAPI router."""

import json
import logging

from fastapi import APIRouter, HTTPException

from app.shared.providers.base import CompletionRequest
from app.shared.providers.mistral import llm_provider
from app.shared.utils.json_parse import extract_json
from app.features.recommendations.schemas import (
    CourseRecommendationBody, CourseRecommendationResponse,
    LearningPathBody, LearningPathResponse,
    TutorQueryBody, TutorQueryResponse,
    SummaryBody, SummaryResponse,
    QuizGenerationBody, QuizGenerationResponse,
    LearningObjectivesBody, LearningObjectivesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


COURSE_REC_SYSTEM_PROMPT = """You are a learning recommendation engine for an LMS platform.

Given a learner's current skills, a target role, and a catalog of available courses,
recommend the most relevant courses to help the learner close their skill gaps.

Rules:
1. Prioritize courses that address the largest skill gaps.
2. Each recommendation must include the courseId, title, reason, and a priority score.
3. Do NOT recommend courses for skills the learner already has at the required level.
4. Return at most max_recommendations courses.
5. Return ONLY valid JSON array. No markdown.

Output format:
[
  {
    "courseId": "id",
    "courseTitle": "Title",
    "reason": "This course addresses the gap in X",
    "priority": "high|medium|low",
    "skillsAddressed": ["skill1", "skill2"]
  }
]"""


@router.post("/courses", response_model=CourseRecommendationResponse)
async def recommend_courses(body: CourseRecommendationBody):
    skills_text = "\n".join(
        f"- {s.get('skill', 'Unknown')}: {s.get('currentLevel', 'Beginner')} (required: {s.get('requiredLevel', 'Intermediate')})"
        for s in body.user_skills
    )
    courses_text = "\n".join(
        f"- {c.get('courseId', 'unknown')}: {c.get('title', 'Untitled')} — covers: {', '.join(c.get('skills', []))}"
        for c in body.available_courses
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=COURSE_REC_SYSTEM_PROMPT,
                user_prompt=(
                    f"Target Role: {body.target_role}\n\nLearner Skills:\n{skills_text}\n\n"
                    f"Available Courses:\n{courses_text}\n\nRecommend up to {body.max_recommendations} courses."
                ),
                temperature=0.2, max_tokens=1024, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse course recommendation response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return CourseRecommendationResponse(
        recommendations=data if isinstance(data, list) else [],
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


LEARNING_PATH_SYSTEM_PROMPT = """You are a learning path designer for an LMS platform.

Given a learner's skill gaps and available courses, design an optimized learning path.

Rules:
1. Order courses from foundational to advanced.
2. Each course must address at least one identified gap.
3. Include a brief reason for why each course is in the path.
4. Estimate total duration based on course durations.
5. Return ONLY valid JSON. No markdown.

Output format:
{
  "title": "Learning Path for [Role]",
  "description": "A personalized path to close your skill gaps",
  "estimated_duration": "4 weeks",
  "courses": [
    {
      "courseId": "id",
      "courseTitle": "Title",
      "sortOrder": 1,
      "isRequired": true,
      "reason": "Addresses gap in X",
      "gapSkillId": "skillId"
    }
  ]
}"""


@router.post("/learning-path", response_model=LearningPathResponse)
async def generate_learning_path(body: LearningPathBody):
    gaps_text = "\n".join(
        f"- {g.get('skill', 'Unknown')}: {g.get('currentLevel', '?')} → {g.get('requiredLevel', '?')} (gap: {g.get('gap', 1)})"
        for g in body.gaps
    )
    courses_text = "\n".join(
        f"- {c.get('courseId', 'unknown')}: {c.get('title', 'Untitled')} — skills: {', '.join(c.get('skills', []))} — duration: {c.get('duration', 'unknown')}"
        for c in body.available_courses
    )

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=LEARNING_PATH_SYSTEM_PROMPT,
                user_prompt=(
                    f"Target Role: {body.target_role}\n\nSkill Gaps:\n{gaps_text}\n\n"
                    f"Available Courses:\n{courses_text}\n\nDesign the optimal learning path."
                ),
                temperature=0.2, max_tokens=1536, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse learning path response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return LearningPathResponse(
        title=data.get("title", "Learning Path"),
        description=data.get("description", ""),
        estimated_duration=data.get("estimated_duration", "Unknown"),
        courses=data.get("courses", []),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


TUTOR_SYSTEM_PROMPT = """You are an AI tutor embedded in an LMS platform. You help learners understand course material, answer questions, explain concepts, and provide study guidance.

Rules:
1. Be concise but thorough. Aim for clarity over length.
2. If the question is about a specific topic in the course material, reference the relevant content.
3. If you don't know the answer, say so honestly.
4. Use plain language. Avoid jargon unless necessary.
5. If appropriate, suggest related topics the learner should review.
6. Never fabricate information or cite sources you cannot verify."""


@router.post("/tutor", response_model=TutorQueryResponse)
async def tutor_query(body: TutorQueryBody):
    user_content = body.question
    if body.lesson_content:
        user_content = f"Lesson content:\n{body.lesson_content}\n\nQuestion: {body.question}"
    elif body.context:
        user_content = f"Context:\n{body.context}\n\nQuestion: {body.question}"

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=TUTOR_SYSTEM_PROMPT,
                user_prompt=user_content,
                temperature=0.3, max_tokens=1024,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return TutorQueryResponse(
        answer=result.content, model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


SUMMARY_SYSTEM_PROMPT = "You are a summarization engine. Create a clear, accurate summary of the provided content."


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_content(body: SummaryBody):
    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=(
                    f"Summarize the following content in a {body.style} style, "
                    f"keeping it under {body.max_length} words:\n\n{body.content}"
                ),
                temperature=0.2, max_tokens=1024,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return SummaryResponse(
        summary=result.content, model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


QUIZ_GEN_SYSTEM_PROMPT = """You are a quiz generation engine for an LMS platform. Generate assessment questions from the provided content.

Rules:
1. Questions should be based on the content provided.
2. Mix difficulty levels as requested.
3. Each MCQ must have exactly 4 options with one correct answer.
4. Each question must include the correctAnswer field.
5. Return ONLY valid JSON array. No markdown.

Output format:
[
  {
    "text": "Question text",
    "questionType": "MCQ|TRUE_FALSE|ESSAY|FILL_BLANK",
    "options": ["A", "B", "C", "D"],
    "correctAnswer": "A",
    "points": 1,
    "topic": "Topic name"
  }
]"""


@router.post("/quiz", response_model=QuizGenerationResponse)
async def generate_quiz(body: QuizGenerationBody):
    types = ", ".join(body.question_types) if body.question_types else "MCQ, TRUE_FALSE"

    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=QUIZ_GEN_SYSTEM_PROMPT,
                user_prompt=(
                    f"Generate {body.num_questions} questions ({types}) "
                    f"at {body.difficulty} difficulty from the following content:\n\n{body.content}"
                ),
                temperature=0.3, max_tokens=2048, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse quiz generation response: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return QuizGenerationResponse(
        questions=data if isinstance(data, list) else [],
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


LO_SYSTEM_PROMPT = "You are a curriculum designer. Generate clear, measurable learning objectives from the provided content. Return ONLY a valid JSON array of strings."


@router.post("/learning-objectives", response_model=LearningObjectivesResponse)
async def generate_learning_objectives(body: LearningObjectivesBody):
    try:
        result = await llm_provider.complete(
            CompletionRequest(
                system_prompt=LO_SYSTEM_PROMPT,
                user_prompt=(
                    f"Generate {body.num_objectives} learning objectives from:\n\n{body.content}\n\n"
                    "Return a JSON array of objective strings."
                ),
                temperature=0.2, max_tokens=512, json_mode=True,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        data = extract_json(result.content)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    return LearningObjectivesResponse(
        objectives=data if isinstance(data, list) else [],
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
