COURSE_METADATA_SCHEMA = {
    "type": "object",
    "required": [
        "courseCode",
        "shortDescription",
        "difficulty",
        "category",
        "learningObjectives",
        "skillsCovered",
        "estimatedDurationHours",
        "recommendedPrerequisites",
        "suggestedTags",
    ],
    "properties": {
        "courseCode": {"type": "string", "minLength": 1, "maxLength": 50},
        "shortDescription": {"type": "string", "minLength": 50},
        "difficulty": {
            "type": "string",
            "enum": ["BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"],
        },
        "category": {"type": "string", "minLength": 1},
        "learningObjectives": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
        },
        "skillsCovered": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 8,
        },
        "estimatedDurationHours": {"type": "number", "minimum": 0.5},
        "recommendedPrerequisites": {"type": "string"},
        "suggestedTags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
        },
    },
    "additionalProperties": False,
}

COURSE_MODULES_SCHEMA = {
    "type": "object",
    "required": ["modules"],
    "properties": {
        "modules": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "required": ["title", "description", "lessons"],
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 200},
                    "description": {"type": "string", "minLength": 1},
                    "lessons": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "required": ["title", "contentType"],
                            "properties": {
                                "title": {"type": "string", "minLength": 1, "maxLength": 200},
                                "contentType": {
                                    "type": "string",
                                    "enum": ["video", "text", "pdf", "external_link"],
                                },
                                "durationMinutes": {"type": "number", "minimum": 1},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}
