import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .agent import analyze_query, decompose_query, rewrite_query
from .calculator import calculate_electric_field
from .config import settings
from .generation import generate_answer
from .retrieval import qdrant_vector_store, search


MAX_RETRIEVAL_ATTEMPTS = 2


def _with_latency(debug: dict[str, Any], started: float) -> dict[str, Any]:
    debug["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return debug


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
    messages = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Class 12 Physics study assistant. "
                "Answer casual conversation and study guidance briefly and naturally.",
            ),
            *[
                (message.get("role", "user"), message.get("content", ""))
                for message in history[-8:]
                if isinstance(message, dict)
            ],
            ("user", "{question}"),
        ]
    )
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.generation_model,
        temperature=0.2,
        max_retries=0,
    ).with_retry(
        retry_if_exception_type=(ConnectionError, TimeoutError),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )
    response = (messages | llm).invoke({"question": question})
    return str(response.content or "")


def process_student_query(
    question: str,
    chat_history: list[dict],
    chapter: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    from .graph import graph

    result = graph.invoke({
        "user_message": question,
        "conversation_history": chat_history,
        "chapter": chapter,
        "retry_count": 0,
    })
    intent = result.get("intent", "")
    used_retrieval = intent not in {"casual_chat", "study_guidance"}
    calculation_result = result.get("calculation_result")
    debug = {
        "standalone_query": result.get("rewritten_query", question),
        "subqueries": result.get("subqueries", []),
        "decomposed": len(result.get("subqueries", [])) > 1,
        "retrieval_attempts": 0 if not used_retrieval else result.get("retry_count", 0) + 1,
        "evidence_sufficient": result.get("evidence_sufficient", False),
        "calculation_used": bool(calculation_result and calculation_result.get("success")),
    }
    return {
        "answer": result.get("final_answer", ""),
        "intent": intent,
        "used_retrieval": used_retrieval,
        "sources": result.get("sources", []),
        "debug": _with_latency(debug, started),
    }
