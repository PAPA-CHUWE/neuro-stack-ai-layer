CV_EXTRACTION_SYSTEM_PROMPT = """You are an expert HR skills analyst. Your task is to extract structured skills from a candidate's CV/resume.

Rules:
1. Extract ONLY skills that are explicitly mentioned or strongly implied by the content.
2. Do NOT infer skills that are not supported by evidence in the text.
3. Proficiency levels must be one of: BEGINNER, INTERMEDIATE, ADVANCED, EXPERT.
4. yearsExperience must be a non-negative number (use 0 if not explicitly stated but skill is present).
5. evidence must quote or paraphrase the specific passage supporting each skill.
6. confidence must be between 0.0 and 1.0 based on how explicit the evidence is.
7. Return ONLY valid JSON array. No markdown, no explanations, no conversational text.
8. If the CV contains no extractable skills, return an empty array [].

Output format:
[
  {
    "skill": "Skill Name",
    "proficiencyLevel": "INTERMEDIATE",
    "yearsExperience": 3,
    "evidence": "Developed backend APIs using...",
    "confidence": 0.87
  }
]"""

CV_EXTRACTION_USER_PROMPT = lambda cv_text: (
    f"Extract skills from the following CV/resume text:\n\n{cv_text}"
)

CV_EXTRACTION_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["skill", "proficiencyLevel", "yearsExperience", "evidence", "confidence"],
        "properties": {
            "skill": {"type": "string", "minLength": 1},
            "proficiencyLevel": {
                "type": "string",
                "enum": ["BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"],
            },
            "yearsExperience": {"type": "number", "minimum": 0},
            "evidence": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    },
    "minItems": 0,
}

CV_RAG_CONTEXT_PROMPT = lambda context_documents: (
    "Use the following reference materials to improve skill extraction accuracy. "
    "If a skill appears in the references, normalize its name to match the reference:\n\n"
    + "\n\n---\n\n".join(context_documents)
)
