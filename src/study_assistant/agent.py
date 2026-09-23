import json
from typing import Any

from .groq_retry import call_with_retry

from .config import settings


INTENTS = {
    "casual_chat",
    "textbook_question",
    "follow_up_question",
    "calculation",
    "study_guidance",
}
MAX_HISTORY_MESSAGES = 8
MAX_SUBQUERIES = 3
OBVIOUS_CASUAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
}


def _recent_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in history[-MAX_HISTORY_MESSAGES:]
        if isinstance(message, dict) and message.get("content")
    ]


def _groq_json(prompt: str) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    try:
        from groq import Groq, RateLimitError
    except ImportError as exc:
        raise RuntimeError("groq is required for query analysis") from exc

    try:
        response = call_with_retry(
            lambda: Groq(api_key=settings.groq_api_key).chat.completions.create(
                model=settings.generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY the required JSON object. Do not include "
                            "markdown, explanation, or any text outside the JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=200,
                response_format={"type": "json_object"},
            )
        )
    except RateLimitError as exc:
        raise RuntimeError(
            "Groq rate limit reached while processing the structured query request. "
            "Please retry later."
        ) from exc
    content = response.choices[0].message.content or ""
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Query router returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Query router returned an invalid object")
    return result


def _validate_route(result: dict[str, Any]) -> dict[str, Any]:
    if (
        result.get("intent") not in INTENTS
        or not isinstance(result.get("needs_retrieval"), bool)
        or not isinstance(result.get("standalone_query"), str)
        or not result["standalone_query"].strip()
        or not isinstance(result.get("requires_calculation"), bool)
    ):
        raise RuntimeError("Query router returned an invalid routing payload")
    return {
        "intent": result["intent"],
        "needs_retrieval": result["needs_retrieval"],
        "standalone_query": result["standalone_query"].strip(),
        "requires_calculation": result["requires_calculation"],
    }


def _obvious_casual_route(question: str) -> dict[str, Any] | None:
    normalized = " ".join(question.lower().strip().split()).rstrip("!?.,")
    if normalized not in OBVIOUS_CASUAL_MESSAGES:
        return None
    return {
        "intent": "casual_chat",
        "needs_retrieval": False,
        "standalone_query": question.strip(),
        "requires_calculation": False,
    }


def analyze_query(question: str, history: list[dict]) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("question must not be empty")
    casual_route = _obvious_casual_route(question)
    if casual_route is not None:
        return casual_route
    prompt = f"""Classify the latest student message and rewrite it as a standalone query.
Use exactly one intent: casual_chat, textbook_question, follow_up_question,
calculation, or study_guidance.
Set needs_retrieval to true only when textbook evidence is needed.
Set requires_calculation to true only for an actual numerical calculation.
Return ONLY valid JSON with exactly these keys. Do not include any explanation,
markdown, or text outside the JSON:
{{"intent": "...", "needs_retrieval": true, "standalone_query": "...",
 "requires_calculation": false}}

Recent conversation:
{json.dumps(_recent_history(history), ensure_ascii=True)}

Latest student message:
{question}
"""
    return _validate_route(_groq_json(prompt))


def rewrite_query(question: str, chat_history: list[dict]) -> str:
    if not question.strip():
        raise ValueError("question must not be empty")
    prompt = f"""Rewrite the context-dependent student question as one standalone
textbook retrieval query. Resolve pronouns and references using the recent
conversation. Preserve the student's actual topic and intent. Return ONLY valid JSON with this key: {{"standalone_query": "..."}}.
Do not include any explanation, markdown, or text outside the JSON.

Recent conversation:
{json.dumps(_recent_history(chat_history), ensure_ascii=True)}

Current question:
{question}
"""
    result = _groq_json(prompt)
    standalone_query = result.get("standalone_query")
    if not isinstance(standalone_query, str) or not standalone_query.strip():
        raise RuntimeError("Query rewriter returned an invalid query")
    return standalone_query.strip()


def decompose_query(query: str, chat_history: list[dict]) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    prompt = f"""Determine whether this textbook retrieval query contains multiple
distinct concepts that need independent searches. Decompose only comparisons or
clearly separate concepts; keep ordinary questions as one query.
Return ONLY valid JSON with exactly these keys. Do not include any explanation,
markdown, or text outside the JSON:
{{"decompose": false, "queries": ["..."]}}
Use at most {MAX_SUBQUERIES} non-empty queries.

Recent conversation:
{json.dumps(_recent_history(chat_history), ensure_ascii=True)}

Retrieval query:
{query}
"""
    result = _groq_json(prompt)
    queries = result.get("queries")
    if not isinstance(queries, list):
        raise RuntimeError("Query decomposer returned invalid queries")
    cleaned_queries = [
        item.strip()
        for item in queries
        if isinstance(item, str) and item.strip()
    ][:MAX_SUBQUERIES]
    if not cleaned_queries:
        raise RuntimeError("Query decomposer returned no usable queries")
    return {
        "decompose": bool(result.get("decompose")) and len(cleaned_queries) > 1,
        "queries": cleaned_queries if bool(result.get("decompose")) else [query.strip()],
    }
