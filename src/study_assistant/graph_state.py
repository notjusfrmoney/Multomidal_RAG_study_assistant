from typing import Any, TypedDict


class StudyAssistantState(TypedDict, total=False):
    user_message: str
    conversation_history: list[dict[str, Any]]
    intent: str
    rewritten_query: str
    subqueries: list[str]
    candidates: list[list[Any]]
    reranked_results: list[Any]
    evidence_sufficient: bool
    retry_count: int
    calculation_result: dict[str, Any] | None
    final_answer: str
    sources: list[Any]
    chapter: str | None
    debug: dict[str, Any]
