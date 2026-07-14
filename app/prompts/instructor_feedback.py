from pydantic import BaseModel


class Question(BaseModel):
    questionId: str
    questionText: str
    isCorrect: bool
    learnerAnswer: str
    correctAnswer: str
    topic: str


class FeedbackData(BaseModel):
    assessment_title: str
    learner_name: str
    score: int
    passing_score: int
    weak_topics: list[str]
    questions: list[Question]


INSTRUCTOR_FEEDBACK_SYSTEM_PROMPT = lambda tone: (
    f"You are an expert instructional designer writing constructive feedback "
    f"for a learner's assessment submission in a {tone} tone.\n\n"
    "Rules:\n"
    "1. Feedback must be actionable, specific, and encouraging.\n"
    "2. Reference specific weak topics and gaps identified by the deterministic analysis.\n"
    "3. Do NOT reveal the correct answer directly for incorrect responses \u2014 "
    "guide the learner toward the right understanding.\n"
    "4. Every feedback claim must be traceable to a specific assessment question or weak topic.\n"
    "5. Return ONLY valid JSON. No markdown, no explanations."
)

INSTRUCTOR_FEEDBACK_USER_PROMPT = lambda data: (
    f'Generate constructive feedback for {data.learner_name} on "{data.assessment_title}".\n'
    f"Score: {data.score}/{data.passing_score} required to pass.\n"
    f"Weak topics to address: {', '.join(data.weak_topics)}\n\n"
    "Questions and answers:\n"
    + "\n\n".join(
        f"Q[{q.questionId}] ({q.topic}): {q.questionText}\n"
        f"Learner answered: {q.learnerAnswer}\n"
        f"Correct: {'Yes' if q.isCorrect else 'No'} \u2014 {q.correctAnswer}"
        for q in data.questions
    )
)

INSTRUCTOR_FEEDBACK_JSON_SCHEMA = {
    "type": "object",
    "required": ["overallFeedback", "perQuestionFeedback"],
    "properties": {
        "overallFeedback": {"type": "string", "minLength": 1},
        "perQuestionFeedback": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["questionId", "feedback"],
                "properties": {
                    "questionId": {"type": "string", "minLength": 1},
                    "feedback": {"type": "string", "minLength": 1},
                    "isCorrect": {"type": "boolean"},
                    "improvementTip": {"type": "string"},
                },
            },
        },
        "weakTopicsAddressed": {"type": "array", "items": {"type": "string"}},
    },
}
