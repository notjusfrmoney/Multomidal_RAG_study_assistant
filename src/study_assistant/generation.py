from .citations import build_context
from .store import SearchResult


def generate_answer(question: str, results: list[SearchResult], api_key: str, model_name: str) -> str:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is required for answer generation") from exc
    prompt = f"""Answer the student's question using only the textbook evidence below.
Explain the physics clearly, cite the source file and page in your answer, and say when the evidence is insufficient.

Question: {question}

Textbook evidence:
{build_context(results)}
"""
    response = Groq(api_key=api_key).chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""
