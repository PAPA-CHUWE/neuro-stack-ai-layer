"""Instructor feedback prompts and schema."""

from pydantic import BaseModel


class Question(BaseModel):
    id: str
    text: str
    correct_answer: str | None = None
    learner_answer: str | None = None
    is_correct: bool | None = None


class FeedbackData(BaseModel):
    learner_name: str
    assessment_title: str
    score: float
    questions: list[Question]


INSTRUCTOR_FEEDBACK_SYSTEM_PROMPT = """You are an expert instructional designer writing constructive feedback
for a learner who just completed an assessment.

Your feedback must:
1. Be encouraging but honest
2. Highlight what the learner did well
3. Identify specific areas for improvement
4. Provide actionable next steps
5. Reference specific questions they got wrong and explain why

Tone: Professional, supportive, educational."""

INSTRUCTOR_FEEDBACK_USER_TEMPLATE = """Generate constructive feedback for {learner_name} on "{assessment_title}".

Score: {score}%

Questions:
{questions_text}

Provide detailed, question-by-question feedback."""

INSTRUCTOR_FEEDBACK_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_feedback": {"type": "string", "minLength": 20},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "question_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "questionId": {"type": "string"},
                    "feedback": {"type": "string", "minLength": 1},
                },
                "required": ["questionId", "feedback"],
            },
        },
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overall_feedback", "question_feedback"],
}
