from types import SimpleNamespace

import src.study_assistant.retrieval as retrieval
from src.study_assistant.orchestrator import _merge_records
from src.study_assistant.retrieval import rerank_results
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


def test_reranking_promotes_strong_keyword_overlap():
    records = [
        SearchResult("semantic", "text", 0.90, "electric field and charge", 1, 1),
        SearchResult("keyword", "text", 0.80, "electric flux electric flux", 2, 2),
    ]
    reranked = rerank_results("electric flux", records, 2)
    assert [record.source_id for record in reranked] == ["keyword", "semantic"]


def test_reranking_keeps_semantic_score_dominant():
    records = [
        SearchResult("semantic", "text", 0.95, "related material", 1, 1),
        SearchResult("keyword", "text", 0.60, "electric flux", 2, 2),
    ]
    reranked = rerank_results("electric flux", records, 2)
    assert [record.source_id for record in reranked] == ["semantic", "keyword"]


def test_reranking_handles_empty_or_malformed_text_and_preserves_metadata():
    records = [
        SearchResult(
            "page-1",
            "page",
            0.8,
            None,
            12,
            13,
            page_image="data/pages/page-1.png",
            source_file="chapter.pdf",
        ),
        SearchResult("text-2", "text", 0.7, "electric field", 14, 14),
    ]
    reranked = rerank_results("electric field", records, 1)
    assert len(reranked) == 1
    assert reranked[0].source_id == "text-2"
    assert reranked[0].record_type == "text"
    assert reranked[0].page_start == 14
    assert reranked[0].source_file == ""


def test_search_uses_bounded_candidate_pool_before_reranking(monkeypatch):
    points = [
        SimpleNamespace(
            id=str(index),
            score=0.9 - index / 100,
            payload={
                "record_type": "page",
                "text": "electric field",
                "page_start": index,
                "page_end": index,
                "page_image": f"page-{index}.png",
                "source_file": "chapter.pdf",
            },
        )
        for index in range(1, 11)
    ]

    class Client:
        def __init__(self):
            self.limit = None

        def retrieve(self, **kwargs):
            return points

    class VectorStore:
        def __init__(self):
            self.client = Client()

        def similarity_search_with_score_by_vector(self, vector, k):
            self.client.limit = k
            return [
                (
                    SimpleNamespace(
                        page_content=point.payload["text"],
                        metadata={"_id": point.id},
                    ),
                    point.score,
                )
                for point in points[:k]
            ]

    monkeypatch.setattr(retrieval, "embed_texts", lambda texts, model: [[0.1]])
    client = VectorStore()
    results = retrieval.search(client, "collection", "electric field", "model", top_k=5)

    assert client.client.limit == 10
    assert len(results) == 5
