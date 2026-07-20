"""Deduplicated JSON parsing and date helpers used across feature packages."""

import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response text, handling markdown fences and extra text."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to extract JSON from LLM response: %s", text[:200])
    return {}


def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)
