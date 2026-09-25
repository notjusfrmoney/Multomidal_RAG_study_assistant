from typing import Any

from .agent import analyze_query, decompose_query, rewrite_query
from .calculator import calculate_electric_field
from .config import settings
from .generation import generate_answer
from .graph_state import StudyAssistantState
from .orchestrator import _conversational_answer, _merge_records, assess_evidence
from .retrieval import search
from .store import qdrant_client


def route_node(state: StudyAssistantState) -> dict[str, Any]:
    route = analyze_query(state["user_message"], state.get("conversation_history", []))
    return {
        "intent": route["intent"],
        "rewritten_query": route["standalone_query"],
    }


def rewrite_followup_node(state: StudyAssistantState) -> dict[str, str]:
    return {
        "rewritten_query": rewrite_query(
            state["user_message"],
            state.get("conversation_history", []),
        )
    }


def decompose_node(state: StudyAssistantState) -> dict[str, Any]:
    decomposition = decompose_query(
        state["rewritten_query"],
        state.get("conversation_history", []),
    )
    return {"subqueries": decomposition["queries"]}


def retrieve_node(state: StudyAssistantState) -> dict[str, Any]:
    client = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    query = state["rewritten_query"]
    subqueries = state.get("subqueries", [query])
    groups = []
    for subquery in subqueries:
        records = search(
            client,
            settings.collection_name,
            subquery,
            settings.embedding_model,
            settings.top_k,
        )
        chapter = state.get("chapter")
        if chapter and records:
            records = [
                record
                for record in records
                if chapter.lower() in getattr(record, "source_file", "").lower()
            ]
        groups.append(records)
    return {"candidates": groups}


def rerank_merge_node(state: StudyAssistantState) -> dict[str, Any]:
    return {
        "reranked_results": _merge_records(
            state.get("candidates", []),
            settings.top_k,
        )
    }


def evidence_check_node(state: StudyAssistantState) -> dict[str, Any]:
    records = state.get("reranked_results", [])
    evidence = assess_evidence(records, state["rewritten_query"])
    return {"evidence_sufficient": evidence["sufficient"]}


def calculate_node(state: StudyAssistantState) -> dict[str, Any]:
    return {"calculation_result": calculate_electric_field(state["user_message"])}


def generate_node(state: StudyAssistantState) -> dict[str, Any]:
    intent = state.get("intent")
    records = state.get("reranked_results", [])
    if intent in {"casual_chat", "study_guidance"}:
        answer = _conversational_answer(
            state["user_message"],
            state.get("conversation_history", []),
        )
    else:
        answer = generate_answer(
            state["rewritten_query"],
            records,
            settings.groq_api_key,
            settings.generation_model,
            state.get("calculation_result"),
        )
    return {"final_answer": answer, "sources": records}
