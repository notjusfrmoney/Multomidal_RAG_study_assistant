from typing import Literal

from langgraph.graph import END, START, StateGraph

from .graph_nodes import (
    calculate_node,
    decompose_node,
    evidence_check_node,
    generate_node,
    rerank_merge_node,
    retrieve_node,
    rewrite_followup_node,
    route_node,
)
from .graph_state import StudyAssistantState
from .orchestrator import _retry_query


def _retry_node(state: StudyAssistantState) -> dict:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "rewritten_query": _retry_query(state["rewritten_query"]),
        "subqueries": [],
    }


def _insufficient_node(state: StudyAssistantState) -> dict:
    return {
        "final_answer": (
            "I couldn't find enough relevant information in the textbook to answer "
            "that confidently. Try rephrasing the question or asking about a "
            "specific topic from the chapter."
        ),
        "sources": [],
    }


def _route_branch(state: StudyAssistantState) -> Literal[
    "generate", "calculate", "rewrite_followup", "decompose"
]:
    intent = state.get("intent")
    if intent in {"casual_chat", "study_guidance"}:
        return "generate"
    if intent == "calculation":
        return "calculate"
    if intent == "follow_up_question":
        return "rewrite_followup"
    return "decompose"


def _after_calculate(_: StudyAssistantState) -> str:
    return "decompose"


def _evidence_branch(state: StudyAssistantState) -> Literal[
    "generate", "retry", "insufficient"
]:
    if state.get("evidence_sufficient"):
        return "generate"
    if state.get("retry_count", 0) < 1:
        return "retry"
    return "insufficient"


builder = StateGraph(StudyAssistantState)
builder.add_node("route", route_node)
builder.add_node("rewrite_followup", rewrite_followup_node)
builder.add_node("decompose", decompose_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rerank_merge", rerank_merge_node)
builder.add_node("evidence_check", evidence_check_node)
builder.add_node("calculate", calculate_node)
builder.add_node("generate", generate_node)
builder.add_node("retry", _retry_node)
builder.add_node("insufficient", _insufficient_node)

builder.add_edge(START, "route")
builder.add_conditional_edges(
    "route",
    _route_branch,
    {
        "generate": "generate",
        "calculate": "calculate",
        "rewrite_followup": "rewrite_followup",
        "decompose": "decompose",
    },
)
builder.add_edge("rewrite_followup", "retrieve")
builder.add_edge("decompose", "retrieve")
builder.add_edge("calculate", "decompose")
builder.add_edge("retrieve", "rerank_merge")
builder.add_edge("rerank_merge", "evidence_check")
builder.add_conditional_edges(
    "evidence_check",
    _evidence_branch,
    {
        "generate": "generate",
        "retry": "retry",
        "insufficient": "insufficient",
    },
)
builder.add_edge("retry", "retrieve")
builder.add_edge("generate", END)
builder.add_edge("insufficient", END)

graph = builder.compile()
