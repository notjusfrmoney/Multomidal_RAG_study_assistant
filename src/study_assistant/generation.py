from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .citations import build_context
from .store import SearchResult


def generate_answer(
    question: str,
    results: list[SearchResult],
    api_key: str,
    model_name: str,
    calculation_result: dict[str, Any] | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    prompt = f"""Answer the student's question using only the textbook evidence below.
Explain the physics clearly, cite the source file and page in your answer, and say when the evidence is insufficient.

Question: {question}

Textbook evidence:
{build_context(results)}
"""
    if calculation_result:
        prompt += f"\nDeterministic calculation result:\n{calculation_result}\n"
    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0,
        max_retries=0,
    ).with_retry(
        retry_if_exception_type=(ConnectionError, TimeoutError),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )
    messages = ChatPromptTemplate.from_messages([("user", "{prompt}")])
    try:
        response = (messages | llm).invoke({"prompt": prompt})
    except Exception as exc:
        try:
            from groq import RateLimitError
        except ImportError:
            RateLimitError = ()
        if RateLimitError and isinstance(exc, RateLimitError):
            raise RuntimeError(
                "Groq rate limit reached while generating the grounded answer. "
                "Please retry later."
            ) from exc
        raise
    return str(response.content or "")
