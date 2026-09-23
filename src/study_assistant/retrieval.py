import re

from .embeddings import embed_texts
from .store import SearchResult


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}


def _tokens(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in _STOPWORDS
    }


def _record_text(record: object) -> str:
    if isinstance(record, dict):
        return " ".join(
            str(record.get(field, "") or "")
            for field in ("text", "record_type", "source_file")
        )
    return " ".join(
        str(getattr(record, field, "") or "")
        for field in ("text", "record_type", "source_file")
    )


def rerank_results(
    query: str,
    records: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Order dense candidates using a conservative lexical relevance signal."""
    if top_k <= 0:
        return []
    query_tokens = _tokens(query)
    if not query_tokens:
        return records[:top_k]

    scored = []
    for index, record in enumerate(records):
        semantic_score = getattr(record, "score", 0.0)
        if not isinstance(semantic_score, (int, float)):
            semantic_score = 0.0
        text_tokens = _tokens(_record_text(record))
        lexical_score = len(query_tokens & text_tokens) / len(query_tokens)
        final_score = 0.75 * float(semantic_score) + 0.25 * lexical_score
        scored.append((final_score, index, record))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [record for _, _, record in scored[:top_k]]


def search(client, collection_name: str, query: str, model_name: str, top_k: int = 5) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    candidate_k = min(max(top_k, 0) * 2, 10)
    if candidate_k == 0:
        return []
    vector = embed_texts([query], model_name)[0]
    points = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=candidate_k,
        with_payload=True,
    ).points
    records = [
        SearchResult(
            source_id=str(point.id),
            record_type=point.payload.get("record_type", "text"),
            score=float(point.score),
            text=point.payload.get("text", ""),
            page_start=int(point.payload.get("page_start", 0)),
            page_end=int(point.payload.get("page_end", 0)),
            page_image=point.payload.get("page_image"),
            source_file=point.payload.get("source_file", ""),
        )
        for point in points
    ]
    return rerank_results(query, records, top_k)
