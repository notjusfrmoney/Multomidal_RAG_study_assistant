from .embeddings import embed_texts
from .store import SearchResult


def search(client, collection_name: str, query: str, model_name: str, top_k: int = 5) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    vector = embed_texts([query], model_name)[0]
    points = client.query_points(collection_name=collection_name, query=vector, limit=top_k, with_payload=True).points
    return [
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
