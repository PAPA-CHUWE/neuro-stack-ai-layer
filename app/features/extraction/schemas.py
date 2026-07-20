"""Extraction — schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CvExtractBody(BaseModel):
    cv_text: str


class ExtractedSkill(BaseModel):
    skill: str
    proficiencyLevel: str
    yearsExperience: float
    evidence: str
    confidence: float


class CvExtractResponse(BaseModel):
    skills: list[ExtractedSkill]
    model: str
    prompt_tokens: int
    completion_tokens: int
    validation: dict


class GapNarrativeBody(BaseModel):
    learner_name: str
    target_role: str
    overall_score: int
    gaps: list[dict]
    strengths: list[dict]
    recommended_courses: list[dict]
    sequenced_path: list[dict]
    tone: str = "professional"


class GapNarrativeResponse(BaseModel):
    narrative: dict
    model: str
    prompt_tokens: int
    completion_tokens: int
    validation: dict


class InstructorFeedbackBody(BaseModel):
    assessment_title: str
    learner_name: str
    score: int
    passing_score: int
    weak_topics: list[str]
    questions: list[dict]
    tone: str = "professional"


class InstructorFeedbackResponse(BaseModel):
    feedback: dict
    model: str
    prompt_tokens: int
    completion_tokens: int
    validation: dict


class SkillGapAnalysisBody(BaseModel):
    cv_text: str
    target_role: str
    role_requirements: list[dict] | None = None


class SkillGapAnalysisResponse(BaseModel):
    gaps: list[dict]
    strengths: list[dict]
    overall_score: int
    recommendations: list[dict]
    model: str
    prompt_tokens: int
    completion_tokens: int


class WeakTopicBody(BaseModel):
    questions: list[dict]
    course_topics: list[str] | None = None


class WeakTopicAnalysisResponse(BaseModel):
    weak_topics: list[dict]
    model: str
    prompt_tokens: int
    completion_tokens: int
