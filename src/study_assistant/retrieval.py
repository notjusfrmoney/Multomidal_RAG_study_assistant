import re

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

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
    class _EmbeddingAdapter(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embed_texts(texts, model_name)

        def embed_query(self, text: str) -> list[float]:
            return embed_texts([text], model_name)[0]

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=_EmbeddingAdapter(),
        content_payload_key="text",
        metadata_payload_key="metadata",
        validate_collection_config=False,
    )
    vector = embed_texts([query], model_name)[0]
    if hasattr(client, "get_collection"):
        documents = vector_store.similarity_search_with_score_by_vector(
            vector,
            k=candidate_k,
        )
    else:
        points = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=candidate_k,
            with_payload=True,
        ).points
        documents = [
            (
                type("Document", (), {
                    "page_content": point.payload.get("text", ""),
                    "metadata": {"_id": point.id},
                })(),
                point.score,
            )
            for point in points
        ]
    payloads = {}
    if hasattr(client, "retrieve"):
        points = client.retrieve(
            collection_name=collection_name,
            ids=[document.metadata["_id"] for document, _ in documents],
            with_payload=True,
        )
        payloads = {str(point.id): point.payload for point in points}
    records = [
        SearchResult(
            source_id=str(document.metadata["_id"]),
            record_type=payloads.get(str(document.metadata["_id"]), {}).get("record_type", "text"),
            score=float(score),
            text=payloads.get(str(document.metadata["_id"]), {}).get("text", document.page_content),
            page_start=int(payloads.get(str(document.metadata["_id"]), {}).get("page_start", 0)),
            page_end=int(payloads.get(str(document.metadata["_id"]), {}).get("page_end", 0)),
            page_image=payloads.get(str(document.metadata["_id"]), {}).get("page_image"),
            source_file=payloads.get(str(document.metadata["_id"]), {}).get("source_file", ""),
        )
        for document, score in documents
    ]
    return rerank_results(query, records, top_k)
