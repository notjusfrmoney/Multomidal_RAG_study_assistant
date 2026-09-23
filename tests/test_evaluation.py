from pathlib import Path

from scripts.evaluate_agentic import load_benchmark
from scripts.evaluation_metrics import (
    abstention_success,
    category_summary,
    citation_accuracy,
    concept_coverage,
    follow_up_resolution,
    hit_at_k,
    recall_at_k,
)


def test_agentic_benchmark_schema_and_categories():
    benchmark = load_benchmark(Path("evaluation/benchmark.json"))
    assert len(benchmark) >= 20
    categories = {item["category"] for item in benchmark}
    assert {
        "direct_textbook",
        "follow_up",
        "multi_concept",
        "casual",
        "study_guidance",
        "numerical",
        "visual",
        "insufficient_evidence",
    } <= categories
    for item in benchmark:
        assert item["expected"]["intent"]
        assert "needs_retrieval" in item["expected"]
        assert "evaluation" in item


def test_retrieval_hit_and_recall_support_multiple_pages():
    pages = [18, 25, 24]
    assert hit_at_k(pages, [18, 24], 3) is True
    assert recall_at_k(pages, [18, 24], 3) == 1.0
    assert recall_at_k(pages, [18, 24], 2) == 0.5


def test_follow_up_resolution_uses_meaning_not_exact_text():
    assert follow_up_resolution(
        "Explain electric flux again.",
        "Explain the concept of electric flux in more detail.",
    ) is True


def test_concept_coverage_is_explainable():
    coverage = concept_coverage(
        ["point charge electric field", "electric dipole field"],
        ["electric field due to a point charge", "field lines of an electric dipole"],
    )
    assert coverage == 1.0


def test_category_summary_reports_case_count_and_success():
    summary = category_summary(
        [
            {"category": "casual", "overall": {"success": True}},
            {"category": "casual", "overall": {"success": False}},
        ]
    )
    assert summary["casual"] == {"cases": 2, "success_rate": 0.5}


def test_citation_accuracy_accepts_common_page_formats():
    formats = [
        "page 38",
        "Page 38",
        "page: 38",
        "Page: 38",
        "**Page:** 38",
    ]
    for citation in formats:
        assert citation_accuracy(citation, [38]) == 1


def test_abstention_success_is_scoped_to_insufficient_evidence_cases():
    assert abstention_success("The evidence is insufficient to answer.", True) is True
    assert abstention_success("Here is the answer.", True) is False
    assert abstention_success("The evidence is insufficient.", False) is None


def test_reconciled_benchmark_provenance_points_to_verified_chunks():
    benchmark = load_benchmark(Path("evaluation/benchmark.json"))
    by_id = {item["id"]: item["expected"] for item in benchmark}
    assert by_id["direct01"]["source_pages"] == [8]
    assert by_id["direct01"]["source_pages_reconciled"] == [5]
    assert by_id["direct01"]["source_chunks"] == [
        "chapter_01_electric_charges_and_fields-p5-0"
    ]
    assert by_id["direct03"]["source_pages_reconciled"] == [36, 40]
    assert by_id["follow03"]["source_pages_reconciled"] == [36, 40]
    assert by_id["multi01"]["source_pages_reconciled"] == [21]
    assert by_id["visual02"]["source_pages_reconciled"] == [21]
