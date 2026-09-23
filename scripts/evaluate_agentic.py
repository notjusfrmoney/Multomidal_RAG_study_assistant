import json
import time
from pathlib import Path

from src.study_assistant.agent import analyze_query
from src.study_assistant.calculator import calculate_electric_field
from src.study_assistant.config import settings
from src.study_assistant.groq_retry import GroqTransientError, call_with_retry
from src.study_assistant.orchestrator import process_student_query
from scripts.evaluation_metrics import (
    category_summary,
    citation_accuracy,
    concept_coverage,
    follow_up_resolution,
    hit_at_k,
    mean,
    page_numbers,
    recall_at_k,
    abstention_success,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "evaluation" / "benchmark.json"
RESULTS_PATH = ROOT / "data" / "evaluation" / "agentic_results.json"


def _expected_source_pages(expected: dict) -> set[int]:
    reconciled = expected.get("source_pages_reconciled")
    return page_numbers(reconciled if reconciled is not None else expected.get("source_pages", []))


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Agentic benchmark must be a JSON list")
    required = {"id", "category", "question", "expected", "evaluation"}
    for item in data:
        if not required <= item.keys():
            raise ValueError(f"Benchmark item is missing fields: {item}")
    return data


def _judge(question: str, gold_answer: str | None, generated: str) -> dict[str, int | None]:
    if not gold_answer:
        return {"correctness": None, "groundedness": None}
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    try:
        from groq import Groq, RateLimitError
    except ImportError as exc:
        raise RuntimeError("groq is required for evaluation") from exc
    prompt = f"""Compare the generated answer with the concise reference answer.
Return ONLY JSON with integer correctness and groundedness scores from 1 to 5.
Do not include explanation or markdown.
Question: {question}
Reference answer: {gold_answer}
Generated answer: {generated}
"""
    try:
        response = call_with_retry(
            lambda: Groq(api_key=settings.groq_api_key).chat.completions.create(
                model=settings.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=100,
                response_format={"type": "json_object"},
            )
        )
    except RateLimitError as exc:
        raise RuntimeError("Groq rate limit reached during evaluation.") from exc
    scores = json.loads(response.choices[0].message.content or "{}")
    return {
        "correctness": scores.get("correctness"),
        "groundedness": scores.get("groundedness"),
    }


def _success(item: dict, result: dict, metrics: dict) -> bool:
    category = item["category"]
    expected = item["expected"]
    if category in {"casual", "study_guidance"}:
        return metrics["intent_correct"] and not result["system"]["used_retrieval"]
    if category == "insufficient_evidence":
        return metrics["abstention_success"] is True
    if category == "numerical":
        return metrics["intent_correct"] and metrics["calculation_correct"] is True
    if category == "follow_up":
        return (
            metrics["intent_correct"]
            and metrics["resolution_correct"]
            and metrics["retrieval_success"]
        )
    if category == "multi_concept":
        return (
            metrics["intent_correct"]
            and metrics["decomposition_correct"]
            and (metrics["concept_coverage"] or 0) >= 0.5
            and metrics["retrieval_success"]
        )
    return (
        metrics["intent_correct"]
        and metrics["retrieval_success"]
        and (metrics["correctness"] is None or metrics["correctness"] >= 3)
        and (metrics["groundedness"] is None or metrics["groundedness"] >= 3)
        and (metrics["citation_accuracy"] in (None, 1))
    )


def _transient_failure_result(item: dict, error: str, started: float) -> dict:
    expected = item["expected"]
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected": expected,
        "routing": {
            "expected_intent": expected.get("intent"),
            "predicted_intent": None,
            "intent_correct": None,
            "expected_needs_retrieval": expected.get("needs_retrieval"),
            "predicted_needs_retrieval": None,
            "decision_correct": None,
        },
        "retrieval": {
            "retrieved_pages": [],
            "expected_pages": sorted(_expected_source_pages(expected)),
            "hit_at_3": None,
            "hit_at_5": None,
            "recall_at_3": None,
            "recall_at_5": None,
        },
        "answer": {
            "correctness": None,
            "groundedness": None,
            "citation_accuracy": None,
        },
        "system": {
            "retrieval_attempts": 0,
            "used_retrieval": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": error,
        },
        "metrics": {
            "intent_correct": None,
            "retrieval_success": None,
            "calculation_correct": None,
            "abstention_success": None,
        },
        "overall": {"success": False},
    }


def evaluate() -> list[dict]:
    results = []
    for item in load_benchmark():
        started = time.perf_counter()
        expected = item["expected"]
        history = item.get("conversation", [])
        try:
            route = analyze_query(item["question"], history)
            try:
                output = process_student_query(item["question"], history)
                error = None
            except GroqTransientError:
                raise
            except RuntimeError as exc:
                output = {
                    "answer": "",
                    "intent": route["intent"],
                    "used_retrieval": False,
                    "sources": [],
                    "debug": {"retrieval_attempts": 0},
                }
                error = str(exc)
            retrieved_pages = [getattr(source, "page_start", 0) for source in output["sources"]]
            expected_pages = _expected_source_pages(expected)
            debug = output.get("debug", {})
            predicted_subqueries = debug.get("subqueries", [])
            expected_concepts = expected.get("concepts", [])
            expected_result = expected.get("calculation", {}).get("expected_result")
            actual_calculation = (
                calculate_electric_field(item["question"])
                if item["category"] == "numerical"
                else {"success": False}
            )
            actual_result = actual_calculation.get("result")
            calculation_correct = None
            if expected_result is not None and actual_result is not None:
                calculation_correct = abs(actual_result - expected_result) / max(abs(expected_result), 1e-12) <= 0.01
            scores = _judge(item["question"], expected.get("gold_answer"), output["answer"])
        except GroqTransientError as exc:
            error = f"Question {item['id']} failed: {exc}"
            print(error)
            results.append(_transient_failure_result(item, error, started))
            continue
        metrics = {
            "intent_correct": route["intent"] == expected.get("intent"),
            "retrieval_decision_correct": route["needs_retrieval"] == expected.get("needs_retrieval"),
            "retrieval_success": bool(set(retrieved_pages) & expected_pages) if expected_pages else bool(output["used_retrieval"]),
            "resolution_correct": follow_up_resolution(
                expected.get("expected_standalone_query"), debug.get("standalone_query", "")
            ),
            "decomposition_correct": debug.get("decomposed") == expected.get("should_decompose"),
            "concept_coverage": concept_coverage(expected_concepts, predicted_subqueries),
            "calculation_correct": calculation_correct,
            "citation_accuracy": citation_accuracy(output["answer"], list(expected_pages)),
            "correctness": scores["correctness"],
            "groundedness": scores["groundedness"],
            "abstention_success": abstention_success(
                output["answer"],
                item["category"] == "insufficient_evidence",
            ),
        }
        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected": expected,
            "routing": {
                "expected_intent": expected.get("intent"),
                "predicted_intent": route["intent"],
                "intent_correct": metrics["intent_correct"],
                "expected_needs_retrieval": expected.get("needs_retrieval"),
                "predicted_needs_retrieval": route["needs_retrieval"],
                "decision_correct": metrics["retrieval_decision_correct"],
            },
            "query_understanding": {
                "expected_standalone_query": expected.get("expected_standalone_query"),
                "predicted_standalone_query": debug.get("standalone_query"),
                "correct": metrics["resolution_correct"],
            },
            "decomposition": {
                "expected": expected.get("should_decompose"),
                "predicted": debug.get("decomposed"),
                "correct": metrics["decomposition_correct"],
                "expected_concepts": expected_concepts,
                "predicted_subqueries": predicted_subqueries,
                "concept_coverage": metrics["concept_coverage"],
            },
            "retrieval": {
                "retrieved_pages": retrieved_pages,
                "expected_pages": sorted(expected_pages),
                "hit_at_3": hit_at_k(retrieved_pages, list(expected_pages), 3) if expected_pages else None,
                "hit_at_5": hit_at_k(retrieved_pages, list(expected_pages), 5) if expected_pages else None,
                "recall_at_3": recall_at_k(retrieved_pages, list(expected_pages), 3),
                "recall_at_5": recall_at_k(retrieved_pages, list(expected_pages), 5),
            },
            "answer": {
                "correctness": scores["correctness"],
                "groundedness": scores["groundedness"],
                "citation_accuracy": metrics["citation_accuracy"],
            },
            "calculation": {
                "used": debug.get("calculation_used", False),
                "correct": calculation_correct,
                "expected_result": expected_result,
                "actual_result": actual_result,
            },
            "system": {
                "retrieval_attempts": debug.get("retrieval_attempts", 0),
                "used_retrieval": output["used_retrieval"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": error,
            },
            "generated_answer": output["answer"],
            "metrics": metrics,
        }
        result["overall"] = {"success": _success(item, result, metrics)}
        results.append(result)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def report(results: list[dict]) -> None:
    completed = [item for item in results if not item["system"]["error"]]
    routing = completed
    expected_retrieval = [item for item in completed if item["expected"].get("needs_retrieval") is not None]
    retrieval = [item for item in completed if item["retrieval"]["expected_pages"]]
    followups = [item for item in completed if item["category"] == "follow_up"]
    multi = [item for item in completed if item["category"] == "multi_concept"]
    numerical = [item for item in completed if item["category"] == "numerical"]
    insufficient = [item for item in completed if item["category"] == "insufficient_evidence"]
    print("=" * 52)
    print("STUDY ASSISTANT EVALUATION")
    print("=" * 52)
    print(f"Total questions: {len(results)}")
    print(f"Failed transient requests: {len(results) - len(completed)}")
    print(f"Intent Accuracy: {mean([int(item['routing']['intent_correct']) for item in routing]) * 100:.1f}%")
    print(f"Retrieval Decision Accuracy: {mean([int(item['routing']['decision_correct']) for item in expected_retrieval]) * 100:.1f}%")
    no_retrieval = [item for item in results if item["expected"].get("needs_retrieval") is False]
    unnecessary = [item for item in no_retrieval if item["system"]["used_retrieval"]]
    print(f"Unnecessary Retrieval Rate: {(len(unnecessary) / len(no_retrieval) * 100) if no_retrieval else 0:.1f}%")
    print(f"Hit@3: {mean([int(item['retrieval']['hit_at_3']) for item in retrieval if item['retrieval']['hit_at_3'] is not None]) * 100:.1f}%")
    print(f"Hit@5: {mean([int(item['retrieval']['hit_at_5']) for item in retrieval if item['retrieval']['hit_at_5'] is not None]) * 100:.1f}%")
    print(f"Recall@3: {mean([item['retrieval']['recall_at_3'] for item in retrieval if item['retrieval']['recall_at_3'] is not None]) * 100:.1f}%")
    print(f"Recall@5: {mean([item['retrieval']['recall_at_5'] for item in retrieval if item['retrieval']['recall_at_5'] is not None]) * 100:.1f}%")
    print(f"Follow-up Resolution: {mean([int(item['metrics']['resolution_correct']) for item in followups if item['metrics']['resolution_correct'] is not None]) * 100 if followups else 0:.1f}%")
    print(f"Contextual Retrieval Success: {mean([int(item['metrics']['retrieval_success']) for item in followups]) * 100 if followups else 0:.1f}%")
    print(f"Decomposition Decision Accuracy: {mean([int(item['metrics']['decomposition_correct']) for item in multi]) * 100 if multi else 0:.1f}%")
    print(f"Concept Coverage: {mean([item['metrics']['concept_coverage'] for item in multi if item['metrics']['concept_coverage'] is not None]) * 100 if multi else 0:.1f}%")
    print(f"Average Correctness: {mean([item['answer']['correctness'] for item in results if item['answer']['correctness'] is not None]) or 0:.2f} / 5")
    print(f"Average Groundedness: {mean([item['answer']['groundedness'] for item in results if item['answer']['groundedness'] is not None]) or 0:.2f} / 5")
    cited = [item["answer"]["citation_accuracy"] for item in results if item["answer"]["citation_accuracy"] is not None]
    print(f"Citation Accuracy: {mean(cited) * 100 if cited else 0:.1f}%")
    print(f"Calculation Accuracy: {mean([int(item['metrics']['calculation_correct']) for item in numerical if item['metrics']['calculation_correct'] is not None]) * 100 if numerical else 0:.1f}%")
    abstention_rate = (
        mean([int(item["metrics"]["abstention_success"]) for item in insufficient]) * 100
        if insufficient
        else 0
    )
    print(
        f"Unsupported-query abstention success: {abstention_rate:.1f}%"
    )
    print(f"End-to-End Success Rate: {mean([int(item['overall']['success']) for item in results]) * 100:.1f}%")
    print("\nCategory                  Cases     Success")
    for category, values in category_summary(results).items():
        print(f"{category:<25} {values['cases']:<9} {values['success_rate'] * 100:.1f}%")


if __name__ == "__main__":
    report(evaluate())
