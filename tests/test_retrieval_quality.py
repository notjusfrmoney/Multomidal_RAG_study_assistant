from src.study_assistant.orchestrator import _merge_records
from src.study_assistant.store import SearchResult


def _records(prefix: str, count: int) -> list[SearchResult]:
    return [
        SearchResult(f"{prefix}{index}", "text", 1 - index / 10, f"{prefix}{index}", index, index)
        for index in range(1, count + 1)
    ]


def test_single_subquery_merge_preserves_order():
    records = _records("A", 3)
    assert _merge_records([records]) == records


def test_two_subquery_merge_interleaves_results():
    first = _records("A", 3)
    second = _records("B", 3)
    merged = _merge_records([first, second])
    assert [record.source_id for record in merged] == ["A1", "B1", "A2", "B2", "A3", "B3"]


def test_three_subquery_merge_round_robins_results():
    groups = [_records(prefix, 2) for prefix in ("A", "B", "C")]
    merged = _merge_records(groups)
    assert [record.source_id for record in merged] == [
        "A1",
        "B1",
        "C1",
        "A2",
        "B2",
        "C2",
    ]


def test_merge_deduplicates_and_keeps_stronger_duplicate():
    first = _records("A", 3)
    duplicate = SearchResult("A2", "page", 2.0, "stronger", 2, 2, page_image="image")
    second = [duplicate, *_records("B", 2)]
    merged = _merge_records([first, second])
    assert [record.source_id for record in merged] == ["A1", "A2", "B1", "A3", "B2"]
    assert merged[1] is duplicate
    assert merged[1].record_type == "page"
    assert merged[1].page_image == "image"


def test_merge_is_bounded_by_limit():
    merged = _merge_records([_records("A", 3), _records("B", 3)], limit=4)
    assert [record.source_id for record in merged] == ["A1", "B1", "A2", "B2"]
