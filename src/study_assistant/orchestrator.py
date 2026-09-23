from typing import Any

from .agent import analyze_query, decompose_query, rewrite_query
from .calculator import calculate_electric_field
from .config import settings
from .generation import generate_answer
from .groq_retry import call_with_retry
from .retrieval import search
from .store import qdrant_client


MAX_RETRIEVAL_ATTEMPTS = 2


def assess_evidence(records: list[Any] | None, query: str) -> dict[str, Any]:
    if not records:
        return {
            "sufficient": False,
            "reason": "No textbook records were retrieved.",
            "record_count": 0,
        }

    usable_count = 0
    for record in records:
        if isinstance(record, dict):
            text = record.get("text")
            page_image = record.get("page_image")
            page_start = record.get("page_start")
        else:
            text = getattr(record, "text", None)
            page_image = getattr(record, "page_image", None)
            page_start = getattr(record, "page_start", None)
        if (isinstance(text, str) and text.strip()) or (
            isinstance(page_image, str)
            and page_image.strip()
            and page_start is not None
        ):
            usable_count += 1

    if usable_count:
        return {
            "sufficient": True,
            "reason": "Usable textbook records were retrieved.",
            "record_count": len(records),
        }
    return {
        "sufficient": False,
        "reason": "Retrieved records contained no usable textbook evidence.",
        "record_count": len(records),
    }


def _retry_query(query: str) -> str:
    normalized = " ".join(query.split()).strip()
    suffix = " textbook context"
    if normalized.lower().endswith(suffix):
        return normalized
    return f"{normalized}{suffix}"


def _record_key(record: Any) -> tuple:
    source_id = getattr(record, "source_id", None)
    if source_id is not None:
        return ("source_id", str(source_id))
    if isinstance(record, dict) and record.get("source_id") is not None:
        return ("source_id", str(record["source_id"]))
    if isinstance(record, dict):
        return (
            "metadata",
            record.get("source_file", ""),
            record.get("page_start"),
            record.get("page_end"),
            record.get("record_type", ""),
            record.get("text", ""),
        )
    return (
        "metadata",
        getattr(record, "source_file", ""),
        getattr(record, "page_start", None),
        getattr(record, "page_end", None),
        getattr(record, "record_type", ""),
        getattr(record, "text", ""),
    )


def _merge_records(
    record_groups: list[list[Any] | None],
    limit: int | None = None,
) -> list[Any]:
    merged = []
    positions = {}
    max_length = max((len(records or []) for records in record_groups), default=0)
    for index in range(max_length):
        for records in record_groups:
            if index >= len(records or []):
                continue
            record = records[index]
            key = _record_key(record)
            position = positions.get(key)
            if position is None:
                positions[key] = len(merged)
                merged.append(record)
            elif getattr(record, "score", float("-inf")) > getattr(
                merged[position], "score", float("-inf")
            ):
                merged[position] = record
            if limit is not None and len(merged) >= limit:
                return merged
    return merged


def _conversational_answer(question: str, history: list[dict]) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    try:
        from groq import Groq, RateLimitError
    except ImportError as exc:
        raise RuntimeError("groq is required for conversational generation") from exc

    try:
        response = call_with_retry(
            lambda: Groq(api_key=settings.groq_api_key).chat.completions.create(
                model=settings.generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful Class 12 Physics study assistant. "
                        "Answer casual conversation and study guidance briefly and naturally.",
                    },
                    *history[-8:],
                    {"role": "user", "content": question},
                ],
                temperature=0.2,
            )
        )
    except RateLimitError as exc:
        raise RuntimeError(
            "Groq rate limit reached while generating the conversational response. "
            "Please retry later."
        ) from exc
    return response.choices[0].message.content or ""


def process_student_query(
    question: str,
    chat_history: list[dict],
    chapter: str | None = None,
) -> dict[str, Any]:
    route = analyze_query(question, chat_history)
    intent = route["intent"]
    if not route["needs_retrieval"]:
        return {
            "answer": _conversational_answer(question, chat_history),
            "intent": intent,
            "used_retrieval": False,
            "sources": [],
            "debug": {
                "standalone_query": route["standalone_query"],
                "retrieval_attempts": 0,
                "calculation_used": False,
            },
        }

    standalone_query = route["standalone_query"]
    if intent == "follow_up_question":
        standalone_query = rewrite_query(question, chat_history)

    client = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    results = None
    evidence = {"sufficient": False, "record_count": 0}
    retrieval_attempts = 0
    subqueries = [standalone_query]
    decomposed = False
    subquery_record_counts = []
    for retrieval_attempts in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        decomposition = decompose_query(standalone_query, chat_history)
        subqueries = decomposition["queries"]
        decomposed = decomposition["decompose"]
        record_groups = []
        for subquery in subqueries:
            records = search(
                client,
                settings.collection_name,
                subquery,
                settings.embedding_model,
                settings.top_k,
            )
            if chapter and records:
                records = [
                    result
                    for result in records
                    if getattr(result, "source_file", "")
                    and chapter.lower() in result.source_file.lower()
                ]
            record_groups.append(records)
        subquery_record_counts = [len(records or []) for records in record_groups]
        results = _merge_records(record_groups, settings.top_k)
        evidence = assess_evidence(results, standalone_query)
        if evidence["sufficient"]:
            calculation_result = None
            calculation_used = False
            calculation_error = None
            if intent == "calculation":
                calculation_result = calculate_electric_field(question)
                calculation_used = calculation_result["success"]
                if not calculation_used:
                    calculation_error = calculation_result["reason"]
            answer = generate_answer(
                standalone_query,
                results,
                settings.groq_api_key,
                settings.generation_model,
                calculation_result if calculation_used else None,
            )
            return {
                "answer": answer,
                "intent": intent,
                "used_retrieval": True,
                "sources": results,
                "debug": {
                    "standalone_query": standalone_query,
                    "decomposed": decomposed,
                    "subqueries": subqueries,
                    "subquery_record_counts": subquery_record_counts,
                    "retrieval_attempts": retrieval_attempts,
                    "evidence_sufficient": True,
                    "calculation_used": calculation_used,
                    **({"calculation_error": calculation_error} if calculation_error else {}),
                },
            }
        if retrieval_attempts < MAX_RETRIEVAL_ATTEMPTS:
            standalone_query = _retry_query(standalone_query)

    return {
        "answer": (
            "I couldn't find enough relevant information in the textbook to answer "
            "that confidently. Try rephrasing the question or asking about a "
            "specific topic from the chapter."
        ),
        "intent": intent,
        "used_retrieval": True,
        "sources": [],
        "debug": {
            "standalone_query": standalone_query,
            "decomposed": decomposed,
            "subqueries": subqueries,
            "subquery_record_counts": subquery_record_counts,
            "retrieval_attempts": retrieval_attempts,
            "evidence_sufficient": False,
            "calculation_used": False,
        },
    }
