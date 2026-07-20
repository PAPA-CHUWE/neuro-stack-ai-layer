"""Mind feature — Response Composer (fallback copy for ungrounded states)."""

FALLBACK_COPY = {
    "knowledge_unrouteable": (
        "I couldn't find enough verified information to answer that accurately. "
        "My current knowledge is limited to approved organizational documents, "
        "course materials, and live platform data. If this should be supported, "
        "an administrator can add it to the knowledge base."
    ),
    "knowledge_empty_retrieval": (
        "I couldn't find an approved document on this topic. "
        "An administrator can add it to the Enterprise Knowledge Base "
        "to enable an accurate answer."
    ),
    "reasoning_insufficient_data": (
        "I don't have enough learner or course data yet to answer that. "
        "Make sure skill profiles and enrollment data are available, "
        "or try rephrasing your question."
    ),
}
