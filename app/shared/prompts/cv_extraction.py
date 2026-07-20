"""CV extraction prompts, schema, and RAG context."""

from pydantic import BaseModel


class CvExtractionResult(BaseModel):
    skills: list[dict]
    experience_years: float | None = None
    education: list[dict] | None = None


CV_EXTRACTION_SYSTEM_PROMPT = """You are an expert technical recruiter and skills analyst.
Extract ALL technical and soft skills from this CV/resume.
For each skill provide:
- name: the skill name
- level: one of "beginner", "intermediate", "advanced", "expert"
- years: estimated years of experience (null if unknown)
- category: "technical", "soft", "domain", "tool", "language"

Be thorough. Include implicit skills (e.g., if they used React, they know JavaScript).
Return valid JSON only."""

CV_EXTRACTION_USER_TEMPLATE = """Extract skills from this CV text:

---
{cv_text}
---

Return JSON: {{"skills": [...], "experience_years": number_or_null, "education": [...]}}"""

CV_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced", "expert"]},
                    "years": {"type": ["number", "null"]},
                    "category": {"type": "string", "enum": ["technical", "soft", "domain", "tool", "language"]},
                },
                "required": ["name", "level", "category"],
            },
        },
        "experience_years": {"type": ["number", "null"]},
        "education": {"type": "array"},
    },
    "required": ["skills"],
}

# RAG context for CV extraction (used when enriching with company skill taxonomy)
CV_RAG_SYSTEM_PROMPT = """You are a skill taxonomy expert. Given extracted CV skills and company context,
map the candidate's skills to the company's competency framework.
For each skill, indicate:
- match: "exact", "partial", or "none"
- mapped_skill: the closest company skill name (if any)
- gap: whether this represents a gap or strength"""
