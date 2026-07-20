BASE_SYSTEM_PROMPT = """You are a curriculum design assistant for an LMS platform. Generate structured course metadata from a course title.

Rules:
1. Return ONLY valid JSON matching the exact schema below. No markdown, no explanations, no extra text.
2. courseCode: short prefix derived from category/title + sequence number (e.g. CCS-101). Use uppercase letters and hyphens only.
3. difficulty: one of BEGINNER, INTERMEDIATE, ADVANCED, EXPERT.
4. learningObjectives: exactly 3-6 measurable objectives.
5. skillsCovered: exactly 3-8 specific skills.
6. estimatedDurationHours: realistic number based on scope (0.5 to 200).
7. recommendedPrerequisites: empty string if none needed.
8. suggestedTags: exactly 3-6 tags.

Output schema:
{
  "courseCode": "string",
  "shortDescription": "string (50-120 words)",
  "difficulty": "BEGINNER|INTERMEDIATE|ADVANCED|EXPERT",
  "category": "string",
  "learningObjectives": ["string", ...],
  "skillsCovered": ["string", ...],
  "estimatedDurationHours": "number",
  "recommendedPrerequisites": "string",
  "suggestedTags": ["string", ...]
}"""

RETRY_SYSTEM_PROMPT = """You are a curriculum design assistant. Your previous response failed validation. You must return ONLY valid JSON matching this exact schema. No markdown, no explanations, no extra text.

Schema:
{
  "courseCode": "string",
  "shortDescription": "string (50-120 words)",
  "difficulty": "BEGINNER|INTERMEDIATE|ADVANCED|EXPERT",
  "category": "string",
  "learningObjectives": ["string", ...],
  "skillsCovered": ["string", ...],
  "estimatedDurationHours": "number",
  "recommendedPrerequisites": "string",
  "suggestedTags": ["string", ...]
}"""


def build_prompt(
    course_title: str,
    existing_categories: list[str],
    retry: bool = False,
) -> tuple[str, str]:
    system_prompt = RETRY_SYSTEM_PROMPT if retry else BASE_SYSTEM_PROMPT

    category_hint = ""
    if existing_categories:
        category_hint = (
            "\n\nExisting tenant categories (match one of these if possible):\n"
            + "\n".join(f"- {c}" for c in existing_categories)
        )

    user_prompt = (
        f"Course Title: {course_title}"
        f"{category_hint}\n\n"
        "Generate the metadata now."
    )
    return system_prompt, user_prompt


MODULES_BASE_SYSTEM_PROMPT = """You are a curriculum design assistant for an LMS platform. Generate a structured module/lesson outline for a course.

Rules:
1. Return ONLY valid JSON matching the exact schema below. No markdown, no explanations, no extra text.
2. Produce about 5 modules (never fewer than 3, never more than 8), ordered so each builds on the last.
3. Each module needs 1-6 lessons.
4. contentType must be one of: video, text, pdf, external_link.
5. durationMinutes is optional but should be a realistic estimate (5-60) when included.
6. If learning objectives and/or skills covered are supplied, the modules together must build toward all of them — do not ignore the supplied context.

Output schema:
{
  "modules": [
    {
      "title": "string",
      "description": "string",
      "lessons": [
        {"title": "string", "contentType": "video|text|pdf|external_link", "durationMinutes": "number (optional)"}
      ]
    }
  ]
}"""

MODULES_RETRY_SYSTEM_PROMPT = """You are a curriculum design assistant. Your previous response failed validation. You must return ONLY valid JSON matching this exact schema. No markdown, no explanations, no extra text.

Schema:
{
  "modules": [
    {
      "title": "string",
      "description": "string",
      "lessons": [
        {"title": "string", "contentType": "video|text|pdf|external_link", "durationMinutes": "number (optional)"}
      ]
    }
  ]
}"""


def build_modules_prompt(
    course_title: str,
    short_description: str = "",
    category: str = "",
    difficulty: str = "",
    learning_objectives: list[str] | None = None,
    skills_covered: list[str] | None = None,
    retry: bool = False,
) -> tuple[str, str]:
    system_prompt = MODULES_RETRY_SYSTEM_PROMPT if retry else MODULES_BASE_SYSTEM_PROMPT

    context_lines = [f"Course Title: {course_title}"]
    if short_description:
        context_lines.append(f"Short Description: {short_description}")
    if category:
        context_lines.append(f"Category: {category}")
    if difficulty:
        context_lines.append(f"Difficulty: {difficulty}")
    if learning_objectives:
        context_lines.append(
            "Learning Objectives:\n" + "\n".join(f"- {o}" for o in learning_objectives)
        )
    if skills_covered:
        context_lines.append(
            "Skills Covered:\n" + "\n".join(f"- {s}" for s in skills_covered)
        )

    user_prompt = "\n\n".join(context_lines) + "\n\nGenerate the module outline now."
    return system_prompt, user_prompt
