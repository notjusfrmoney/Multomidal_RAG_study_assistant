import json
import re
from pathlib import Path

from src.study_assistant.config import settings
from src.study_assistant.generation import generate_answer
from src.study_assistant.retrieval import search
from src.study_assistant.store import qdrant_client


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "evaluation" / "benchmark_chapter_01.json"
RESULTS_PATH = ROOT / "data" / "evaluation" / "results.json"


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark must be a JSON list")
    for index, item in enumerate(data, start=1):
        if not all(field in item for field in ("question", "gold_answer", "gold_sources")):
            raise ValueError(f"Benchmark item {index} must contain question, gold_answer, and gold_sources")
    return data


def source_pages(gold_sources: list) -> set[int]:
    pages = set()
    for source in gold_sources:
        value = source.get("page", source.get("page_start")) if isinstance(source, dict) else source
        if isinstance(value, int):
            pages.add(value)
        elif isinstance(value, str):
            match = re.search(r"\bpage\s+(\d+)\b", value, re.IGNORECASE)
            if match:
                pages.add(int(match.group(1)))
            elif value.isdigit():
                pages.add(int(value))
    return pages


def judge_answer(question: str, gold_answer: str, generated_answer: str, api_key: str, model_name: str) -> dict:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is required for evaluation") from exc

    prompt = f"""You are a strict evaluator for a textbook-grounded Physics RAG system.
Compare the generated answer to the gold answer for the exact question.
Score each dimension from 1 to 5:
- correctness_score: factual accuracy and coverage compared with the gold answer.
- groundedness_score: whether the generated answer stays within the gold answer and does not add unsupported outside knowledge.
Return JSON only, exactly with integer fields correctness_score and groundedness_score.

Question:
{question}

Gold answer:
{gold_answer}

Generated answer:
{generated_answer}
"""
    response = Groq(api_key=api_key).chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("Judge response did not contain a JSON object")
    scores = json.loads(match.group(0))
    for field in ("correctness_score", "groundedness_score"):
        if not isinstance(scores.get(field), int) or not 1 <= scores[field] <= 5:
            raise ValueError(f"Judge returned invalid {field}: {scores.get(field)!r}")
    return scores


def evaluate() -> list[dict]:
    benchmark = load_benchmark()
    client = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    report = []
    for item in benchmark:
        results = search(client, settings.collection_name, item["question"], settings.embedding_model, 5)
        retrieved_pages = [result.page_start for result in results]
        recall = bool(source_pages(item["gold_sources"]) & set(retrieved_pages))
        generated_answer = generate_answer(
            item["question"],
            results,
            settings.groq_api_key,
            settings.generation_model,
        )
        scores = judge_answer(
            item["question"],
            item["gold_answer"],
            generated_answer,
            settings.groq_api_key,
            settings.generation_model,
        )
        report.append(
            {
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "gold_sources": item["gold_sources"],
                "retrieved_pages": retrieved_pages,
                "retrieved_records": [
                    {
                        "source_id": result.source_id,
                        "record_type": result.record_type,
                        "score": result.score,
                        "page_start": result.page_start,
                        "page_end": result.page_end,
                        "source_file": result.source_file,
                        "page_image": result.page_image,
                    }
                    for result in results
                ],
                "generated_answer": generated_answer,
                "recall_at_5": recall,
                **scores,
            }
        )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    results = evaluate()
    recall = sum(result["recall_at_5"] for result in results) / len(results)
    correctness = sum(result["correctness_score"] for result in results) / len(results)
    groundedness = sum(result["groundedness_score"] for result in results) / len(results)
    print(f"Average Recall@5: {recall:.3f}")
    print(f"Average Correctness: {correctness:.3f}")
    print(f"Average Groundedness: {groundedness:.3f}")
    print(f"Saved evaluation report to {RESULTS_PATH}")
