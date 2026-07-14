import logging

import jsonschema

logger = logging.getLogger(__name__)


def validate(data: dict | list, schema: dict) -> dict:
    try:
        jsonschema.validate(data, schema, format_checker=jsonschema.FormatChecker())
        return {"valid": True, "data": data}
    except jsonschema.ValidationError as e:
        errors = [f"{e.json_path} {e.message}"]
        logger.warning("Schema validation failed: %s", errors)
        return {"valid": False, "errors": errors}


CV_EXTRACTION_SCHEMA = {
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
    "minItems": 1,
}

GAP_NARRATIVE_SCHEMA = {
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

FEEDBACK_SCHEMA = {
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


def validate_cv_extraction(data) -> dict:
    return validate(data, CV_EXTRACTION_SCHEMA)


def validate_gap_narrative(data) -> dict:
    return validate(data, GAP_NARRATIVE_SCHEMA)


def validate_feedback_draft(data) -> dict:
    return validate(data, FEEDBACK_SCHEMA)
