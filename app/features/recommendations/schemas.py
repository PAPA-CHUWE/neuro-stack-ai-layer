"""Recommendations — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CourseRecommendationBody(BaseModel):
    user_skills: list[dict]
    target_role: str
    available_courses: list[dict]
    max_recommendations: int = 5


class CourseRecommendationResponse(BaseModel):
    recommendations: list[dict]
    model: str
    prompt_tokens: int
    completion_tokens: int


class LearningPathBody(BaseModel):
    user_id: str
    target_role: str
    gaps: list[dict]
    available_courses: list[dict]
    strengths: list[dict] | None = None


class LearningPathResponse(BaseModel):
    title: str
    description: str
    estimated_duration: str
    courses: list[dict]
    model: str
    prompt_tokens: int
    completion_tokens: int


class TutorQueryBody(BaseModel):
    question: str
    context: str | None = None
    course_id: str | None = None
    lesson_content: str | None = None


class TutorQueryResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class SummaryBody(BaseModel):
    content: str
    style: str = "concise"
    max_length: int = 500


class SummaryResponse(BaseModel):
    summary: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class QuizGenerationBody(BaseModel):
    content: str
    num_questions: int = 5
    question_types: list[str] | None = None
    difficulty: str = "medium"


class QuizGenerationResponse(BaseModel):
    questions: list[dict]
    model: str
    prompt_tokens: int
    completion_tokens: int


class LearningObjectivesBody(BaseModel):
    content: str
    num_objectives: int = 5


class LearningObjectivesResponse(BaseModel):
    objectives: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
