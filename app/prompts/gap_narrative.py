from pydantic import BaseModel


class GapData(BaseModel):
    learner_name: str
    target_role: str
    overall_score: int
    gaps: list[dict]
    strengths: list[dict]
    recommended_courses: list[dict]
    sequenced_path: list[dict]


GAP_NARRATIVE_SYSTEM_PROMPT = lambda tone: (
    f"You are a learning advisor writing in a {tone} tone. "
    "Your task is to explain a learner's skill gap analysis and recommended learning path "
    "using ONLY the provided deterministic data.\n\n"
    "Rules:\n"
    "1. Every claim must reference a specific gap or course from the provided data.\n"
    "2. Do NOT invent statistics, percentages, or outcomes not present in the input.\n"
    "3. Do NOT mention that you are an AI or language model.\n"
    "4. Structure the output as structured JSON with summary and sections.\n"
    "5. Be encouraging but factual \u2014 the learner must understand exactly what to study and why.\n"
    f'6. If a gap is marked as "critical", emphasize it appropriately in the {tone} tone.'
)

GAP_NARRATIVE_USER_PROMPT = lambda data: (
    f"Generate a gap narrative for {data.learner_name} targeting the {data.target_role} role.\n\n"
    f"Overall readiness score: {data.overall_score}/100\n\n"
    "Gaps identified:\n"
    + "\n".join(
        f"- {g['skill']}: current {g['currentLevel']}, required {g['requiredLevel']}, gap {g['gap']}"
        for g in data.gaps
    )
    + "\n\nStrengths:\n"
    + "\n".join(f"- {s['skill']} ({s['currentLevel']})" for s in data.strengths)
    + "\n\nRecommended courses:\n"
    + "\n".join(f"- {c['title']}: {c['reason']}" for c in data.recommended_courses)
    + "\n\nSequenced learning path:\n"
    + "\n".join(
        f"{p['sortOrder']}. {p['courseTitle']} {'(required)' if p['isRequired'] else '(optional)'}"
        for p in data.sequenced_path
    )
)

GAP_NARRATIVE_JSON_SCHEMA = {
    "type": "object",
    "required": ["summary", "sections"],
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["heading", "body"],
                "properties": {
                    "heading": {"type": "string", "minLength": 1},
                    "body": {"type": "string", "minLength": 1},
                    "referencedGaps": {"type": "array", "items": {"type": "string"}},
                    "referencedCourses": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "tone": {"type": "string"},
    },
}
