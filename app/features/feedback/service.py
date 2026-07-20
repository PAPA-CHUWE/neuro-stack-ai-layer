"""Structured AI feedback — triage classification and severity derivation."""

from __future__ import annotations

from app.features.feedback.schemas import SEVERITY_MAP


def classify_triage(reason_codes: list[str], confidence: str | None) -> tuple[str, float, str]:
    """Classify feedback into triage category based on reason codes.

    Returns (category, confidence_score, suggested_remediation).
    """
    reasons = set(reason_codes)

    if "HALLUCINATED" in reasons:
        return "HALLUCINATION", 0.9, "Add prohibited-claims check or strengthen grounding"
    if "WRONG_TENANT_CONTEXT" in reasons or "WRONG_ORGANIZATION_CONTEXT" in reasons:
        return "TENANT_CONTEXT_FAILURE", 0.85, "Improve tenant context injection in prompt"
    if "WRONG_POLICY" in reasons:
        return "KNOWLEDGE_GAP", 0.8, "Add or update policy document in knowledge base"
    if "FAILED_TO_USE_KNOWLEDGE_BASE" in reasons:
        return "RETRIEVAL_FAILURE", 0.8, "Improve retrieval or grounding prompt"
    if "FAILED_TO_USE_LIVE_DATA" in reasons:
        return "TOOL_FAILURE", 0.8, "Check tool availability and routing"
    if "OUTDATED" in reasons:
        return "OUTDATED_DOCUMENT", 0.75, "Update or replace outdated document"
    if "INCOMPLETE" in reasons:
        return "KNOWLEDGE_GAP", 0.7, "Add missing information to knowledge base"
    if "TOO_GENERIC" in reasons:
        return "PROMPT_FAILURE", 0.65, "Strengthen grounding requirements in system prompt"
    if "POOR_CITATION" in reasons:
        return "RETRIEVAL_FAILURE", 0.6, "Improve chunk attribution in prompt"
    if "REPETITIVE" in reasons:
        return "STYLE_FAILURE", 0.5, "Add deduplication instruction to prompt"
    if "TOO_VERBOSE" in reasons or "TOO_SHORT" in reasons:
        return "STYLE_FAILURE", 0.5, "Adjust length instructions in prompt"
    if "UNPROFESSIONAL_TONE" in reasons:
        return "STYLE_FAILURE", 0.5, "Update tone instructions in prompt"
    if "FORMAT_ERROR" in reasons or "STREAMING_DUPLICATION" in reasons:
        return "STREAMING_OR_UI_FAILURE", 0.7, "Frontend rendering issue"
    if "TOOL_FAILURE" in reasons:
        return "TOOL_FAILURE", 0.7, "Investigate tool execution"
    if "IRRELEVANT" in reasons:
        return "RETRIEVAL_FAILURE", 0.6, "Improve query understanding or retrieval"

    return "PROMPT_FAILURE", 0.3, "Review prompt and grounding"


def derive_severity(triage_category: str, reason_codes: list[str]) -> str:
    """Derive severity from triage category and reason codes."""
    for code in reason_codes:
        if code in SEVERITY_MAP:
            severity = SEVERITY_MAP[code]
            if severity in ("critical", "high"):
                return severity

    category_severity = {
        "HALLUCINATION": "high",
        "TENANT_CONTEXT_FAILURE": "critical",
        "KNOWLEDGE_GAP": "medium",
        "RETRIEVAL_FAILURE": "medium",
        "TOOL_FAILURE": "medium",
        "DATA_ACCESS_FAILURE": "medium",
        "OUTDATED_DOCUMENT": "medium",
        "PROMPT_FAILURE": "low",
        "STYLE_FAILURE": "low",
        "STREAMING_OR_UI_FAILURE": "low",
        "MODEL_LIMITATION": "low",
        "USER_MISUNDERSTANDING": "low",
    }
    return category_severity.get(triage_category, "low")
