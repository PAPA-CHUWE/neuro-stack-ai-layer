"""Gap narrative prompts and schema."""

from pydantic import BaseModel


class GapData(BaseModel):
    role_name: str
    current_skills: list[dict]
    required_skills: list[dict]
    gaps: list[dict]


GAP_NARRATIVE_SYSTEM_PROMPT = """You are an expert learning experience designer.
Given a candidate's current skills, the target role requirements, and identified gaps,
write a concise, actionable narrative explaining:
1. What skills the candidate already has that transfer well
2. The key gaps that need to be addressed
3. A prioritized learning approach

Be specific and encouraging. Use the candidate's actual skill names."""

GAP_NARRATIVE_USER_TEMPLATE = """Target role: {role_name}

Current skills:
{current_skills}

Required skills for role:
{required_skills}

Identified gaps:
{gaps}

Write a personalized gap analysis narrative."""

GAP_NARRATIVE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative": {"type": "string", "minLength": 50},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "priority_gaps": {"type": "array", "items": {"type": "string"}},
        "learning_approach": {"type": "string"},
    },
    "required": ["narrative"],
}
