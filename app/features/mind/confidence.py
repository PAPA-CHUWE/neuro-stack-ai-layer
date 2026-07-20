"""Mind feature — Confidence derivation."""

from app.features.mind.schemas import Confidence

GROUNDED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.55


def derive_confidence(results: list, top_score: float) -> Confidence:
    """Derive confidence from retrieval quality. Never trust the model to grade itself."""
    if not results or top_score < PARTIAL_THRESHOLD:
        return Confidence.ungrounded
    if top_score < GROUNDED_THRESHOLD:
        return Confidence.partial
    return Confidence.grounded
