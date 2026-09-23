import re
from collections import defaultdict
from math import log2
from typing import Iterable


def page_numbers(values: Iterable) -> set[int]:
    pages = set()
    for value in values:
        if isinstance(value, int):
            pages.add(value)
        elif isinstance(value, str):
            match = re.search(r"\bpage\s+(\d+)\b", value, re.IGNORECASE)
            if match:
                pages.add(int(match.group(1)))
            elif value.isdigit():
                pages.add(int(value))
    return pages


def hit_at_k(retrieved_pages: list[int], expected_pages: list[int], k: int) -> bool:
    return bool(set(unique_pages(retrieved_pages, k)) & set(expected_pages))


def recall_at_k(retrieved_pages: list[int], expected_pages: list[int], k: int) -> float | None:
    if not expected_pages:
        return None
    return len(set(unique_pages(retrieved_pages, k)) & set(expected_pages)) / len(set(expected_pages))


def unique_pages(retrieved_pages: Iterable[int], k: int | None = None) -> list[int]:
    pages = []
    seen = set()
    values = list(retrieved_pages)
    if k is not None:
        values = values[:k]
    for page in values:
        if page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return pages


def precision_at_k(retrieved_pages: list[int], expected_pages: list[int], k: int) -> float | None:
    if not expected_pages or k <= 0:
        return None
    relevant = len(set(unique_pages(retrieved_pages, k)) & set(expected_pages))
    return relevant / k


def reciprocal_rank(retrieved_pages: list[int], expected_pages: list[int]) -> float | None:
    if not expected_pages:
        return None
    expected = set(expected_pages)
    for rank, page in enumerate(unique_pages(retrieved_pages), start=1):
        if page in expected:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved_pages: list[int], expected_pages: list[int], k: int) -> float | None:
    if not expected_pages or k <= 0:
        return None
    expected = set(expected_pages)
    retrieved = unique_pages(retrieved_pages, k)
    dcg = sum(
        1 / log2(rank + 1)
        for rank, page in enumerate(retrieved, start=1)
        if page in expected
    )
    ideal_count = min(len(expected), k)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def concept_coverage(expected_concepts: list[str], predicted_subqueries: list[str]) -> float | None:
    if not expected_concepts:
        return None
    combined = " ".join(predicted_subqueries).lower()
    covered = 0
    for concept in expected_concepts:
        terms = {term for term in re.findall(r"[a-z0-9]+", concept.lower()) if len(term) > 2}
        if terms and len(terms & set(re.findall(r"[a-z0-9]+", combined))) / len(terms) >= 0.5:
            covered += 1
    return covered / len(expected_concepts)


def follow_up_resolution(expected_query: str | None, predicted_query: str) -> bool | None:
    if not expected_query:
        return None
    expected_terms = {term for term in re.findall(r"[a-z0-9]+", expected_query.lower()) if len(term) > 2}
    predicted_terms = set(re.findall(r"[a-z0-9]+", predicted_query.lower()))
    return bool(expected_terms) and len(expected_terms & predicted_terms) / len(expected_terms) >= 0.5


def citation_accuracy(answer: str, expected_pages: list[int]) -> int | None:
    if not expected_pages:
        return None
    cited_pages = {
        int(page)
        for page in re.findall(
            r"\*{0,2}\s*\bpage\b\s*:?\s*\*{0,2}\s*(\d+)\b",
            answer,
            re.IGNORECASE,
        )
    }
    return int(bool(cited_pages & set(expected_pages)))


def abstention_success(answer: str, expected_insufficient: bool) -> bool | None:
    if not expected_insufficient:
        return None
    normalized = answer.lower()
    return "insufficient" in normalized and (
        "evidence" in normalized or "information" in normalized
    )


def latency_summary(results: list[dict]) -> dict[str, float | int | None]:
    values = [
        item["system"]["latency_ms"]
        for item in results
        if isinstance(item.get("system", {}).get("latency_ms"), (int, float))
    ]
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int((len(ordered) * 0.95) + 0.999999) - 1))
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": ordered[len(ordered) // 2] if len(ordered) % 2 else (
            ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]
        ) / 2,
        "p95_ms": ordered[p95_index],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def failure_rate(results: list[dict]) -> float | None:
    if not results:
        return None
    return sum(bool(item.get("system", {}).get("error")) for item in results) / len(results)


def mean(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def category_summary(results: list[dict]) -> dict[str, dict[str, float | int]]:
    grouped = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)
    return {
        category: {
            "cases": len(items),
            "success_rate": mean([int(item["overall"]["success"]) for item in items]) or 0.0,
        }
        for category, items in sorted(grouped.items())
    }
