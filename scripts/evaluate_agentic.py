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
    failure_rate,
    follow_up_resolution,
    hit_at_k,
    latency_summary,
    mean,
    ndcg_at_k,
    page_numbers,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
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


def _judge(
    question: str,
    gold_answer: str | None,
    generated: str,
    context: str,
) -> dict[str, int | None]:
    if not gold_answer:
        return {key: None for key in (
            "correctness",
            "groundedness",
            "answer_relevancy",
            "context_relevancy",
            "citation_completeness",
        )}
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    try:
        from groq import Groq, RateLimitError
    except ImportError as exc:
        raise RuntimeError("groq is required for evaluation") from exc
    prompt = f"""Evaluate the generated answer against the question, reference,
and retrieved context. Return ONLY JSON with integer scores from 1 to 5 for:
correctness, groundedness, answer_relevancy, context_relevancy, and
citation_completeness. Do not include explanation or markdown.
Question: {question}
Reference answer: {gold_answer}
Generated answer: {generated}
Retrieved context: {context}
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
        "answer_relevancy": scores.get("answer_relevancy"),
        "context_relevancy": scores.get("context_relevancy"),
        "citation_completeness": scores.get("citation_completeness"),
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
            "precision_at_3": None,
            "precision_at_5": None,
            "mrr": None,
            "ndcg_at_5": None,
        },
        "answer": {
            "correctness": None,
            "groundedness": None,
            "citation_accuracy": None,
            "answer_relevancy": None,
            "context_relevancy": None,
            "citation_completeness": None,
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
            context = "\n\n".join(
                str(getattr(source, "text", "") or "")
                for source in output["sources"]
            )
            scores = _judge(
                item["question"],
                expected.get("gold_answer"),
                output["answer"],
                context,
            )
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
            "answer_relevancy": scores["answer_relevancy"],
            "context_relevancy": scores["context_relevancy"],
            "citation_completeness": scores["citation_completeness"],
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
                "precision_at_3": precision_at_k(retrieved_pages, list(expected_pages), 3),
                "precision_at_5": precision_at_k(retrieved_pages, list(expected_pages), 5),
                "mrr": reciprocal_rank(retrieved_pages, list(expected_pages)),
                "ndcg_at_5": ndcg_at_k(retrieved_pages, list(expected_pages), 5),
            },
            "answer": {
                "correctness": scores["correctness"],
                "groundedness": scores["groundedness"],
                "citation_accuracy": metrics["citation_accuracy"],
                "answer_relevancy": scores["answer_relevancy"],
                "context_relevancy": scores["context_relevancy"],
                "citation_completeness": scores["citation_completeness"],
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


def _metric(values: list, scale: float = 1.0) -> dict:
    values = [value for value in values if value is not None]
    return {
        "value": mean(values) * scale if values else None,
        "count": len(values),
    }


def aggregate_results(results: list[dict], benchmark: list[dict] | None = None) -> dict:
    benchmark_by_id = {item["id"]: item for item in benchmark or []}
    normalized = []
    for item in results:
        benchmark_item = benchmark_by_id.get(item["id"], item)
        metrics = dict(item.get("metrics", {}))
        if metrics.get("abstention_success") is None:
            metrics["abstention_success"] = abstention_success(
                item.get("generated_answer", ""),
                item["category"] == "insufficient_evidence",
            )
        normalized.append(
            (item, benchmark_item, metrics, _success(item, item, metrics))
        )
    results = [
        {**item, "_offline_success": success}
        for item, _, _, success in normalized
    ]
    completed = [item for item in results if not item["system"].get("error")]
    retrieval = [item for item in completed if item["retrieval"].get("expected_pages")]
    followups = [item for item in completed if item["category"] == "follow_up"]
    multi = [item for item in completed if item["category"] == "multi_concept"]
    numerical = [item for item in completed if item["category"] == "numerical"]
    insufficient = [item for item in completed if item["category"] == "insufficient_evidence"]
    visual_ids = {
        item["id"]
        for item, benchmark_item, _, _ in normalized
        if benchmark_item.get("evaluation", {}).get("visual_evidence_required")
    }
    visual = [item for item in completed if item["id"] in visual_ids]
    retrieval_values = {
        name: [
            function(
                item["retrieval"].get("retrieved_pages", []),
                item["retrieval"].get("expected_pages", []),
                k,
            )
            for item in retrieval
        ]
        for name, function, k in (
            ("precision_at_3", precision_at_k, 3),
            ("precision_at_5", precision_at_k, 5),
            ("recall_at_3", recall_at_k, 3),
            ("recall_at_5", recall_at_k, 5),
            ("hit_at_3", hit_at_k, 3),
            ("hit_at_5", hit_at_k, 5),
        )
    }
    retrieval_values["mrr"] = [
        reciprocal_rank(item["retrieval"]["retrieved_pages"], item["retrieval"]["expected_pages"])
        for item in retrieval
    ]
    retrieval_values["ndcg_at_5"] = [
        ndcg_at_k(item["retrieval"]["retrieved_pages"], item["retrieval"]["expected_pages"], 5)
        for item in retrieval
    ]
    return {
        "retrieval_metrics": {
            name: {**_metric(values, 100), "scope": f"{len(retrieval)} page-grounded retrieval cases"}
            for name, values in retrieval_values.items()
        },
        "rag_context_metrics": {
            "context_precision": {
                **_metric(retrieval_values["precision_at_5"], 100),
                "scope": f"{len(retrieval)} cases; page-level context proxy",
            },
            "context_recall": {
                **_metric(retrieval_values["recall_at_5"], 100),
                "scope": f"{len(retrieval)} cases; page-level context proxy",
            },
            "context_relevancy": {
                **_metric([item["answer"].get("context_relevancy") for item in completed], 1),
                "scope": "cases with live LLM-judge context scores",
            },
        },
        "answer_metrics": {
            name: {
                **_metric([item["answer"].get(name) for item in completed], 1),
                "scope": "cases with live LLM-judge scores",
            }
            for name in ("correctness", "groundedness", "answer_relevancy")
        },
        "citation_metrics": {
            "citation_accuracy": {
                **_metric([item["answer"].get("citation_accuracy") for item in retrieval], 100),
                "scope": f"{len(retrieval)} cases with expected pages",
            },
            "citation_completeness": {
                **_metric([item["answer"].get("citation_completeness") for item in completed], 1),
                "scope": "cases with live LLM-judge scores",
            },
        },
        "agentic_metrics": {
            "intent_accuracy": {
                **_metric([item["routing"].get("intent_correct") for item in completed], 100),
                "scope": f"{len(completed)} completed cases",
            },
            "follow_up_resolution": {
                **_metric([item["metrics"].get("resolution_correct") for item in followups], 100),
                "scope": f"{len(followups)} follow-up cases",
            },
            "decomposition_decision_accuracy": {
                **_metric([item["metrics"].get("decomposition_correct") for item in multi], 100),
                "scope": f"{len(multi)} multi-concept cases",
            },
            "concept_coverage": {
                **_metric([item["metrics"].get("concept_coverage") for item in multi], 100),
                "scope": f"{len(multi)} multi-concept cases",
            },
            "calculation_accuracy": {
                **_metric([item["metrics"].get("calculation_correct") for item in numerical], 100),
                "scope": f"{len(numerical)} numerical cases",
            },
            "abstention_success": {
                **_metric([
                    item["metrics"].get("abstention_success")
                    if item["metrics"].get("abstention_success") is not None
                    else abstention_success(
                        item.get("generated_answer", ""),
                        True,
                    )
                    for item in insufficient
                ], 100),
                "scope": f"{len(insufficient)} insufficient-evidence cases",
            },
            "end_to_end_success": {
                **_metric([item.get("_offline_success") for item in results], 100),
                "scope": f"{len(results)} total benchmark cases",
            },
        },
        "multimodal_metrics": {
            "visual_page_retrieval_hit_at_5": {
                **_metric([
                    hit_at_k(item["retrieval"]["retrieved_pages"], item["retrieval"]["expected_pages"], 5)
                    for item in visual if item["retrieval"].get("expected_pages")
                ], 100),
                "scope": f"{len(visual)} visual-evidence cases with expected pages",
            },
            "visual_evidence_groundedness": {
                **_metric([item["answer"].get("groundedness") for item in visual], 1),
                "scope": "visual cases with live LLM-judge groundedness",
            },
        },
        "system_metrics": {
            **latency_summary(results),
            "failure_rate": failure_rate(results),
            "scope": f"{len(results)} stored evaluation cases",
        },
    }


def report(results: list[dict]) -> None:
    summary = aggregate_results(results, load_benchmark())
    print("=" * 52)
    print("STUDY ASSISTANT EVALUATION")
    print("=" * 52)
    for category, metrics in summary.items():
        print(f"\n{category.replace('_', ' ').title()}")
        for name, value in metrics.items():
            if isinstance(value, dict):
                print(f"- {name}: {value['value']} (n={value['count']}; {value['scope']})")
            else:
                print(f"- {name}: {value}")
    print("\nMachine-readable summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    report(evaluate())
