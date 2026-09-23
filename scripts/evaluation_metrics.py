import re
from collections import defaultdict
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
    return bool(set(retrieved_pages[:k]) & set(expected_pages))


def recall_at_k(retrieved_pages: list[int], expected_pages: list[int], k: int) -> float | None:
    if not expected_pages:
        return None
    return len(set(retrieved_pages[:k]) & set(expected_pages)) / len(set(expected_pages))


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
