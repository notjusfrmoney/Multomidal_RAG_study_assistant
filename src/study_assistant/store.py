from dataclasses import dataclass


@dataclass
class SearchResult:
    source_id: str
    record_type: str
    score: float
    text: str
    page_start: int
    page_end: int
    page_image: str | None = None
    source_file: str = ""


def qdrant_client(url: str, api_key: str):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for indexing and retrieval") from exc
    if not url:
        raise RuntimeError("QDRANT_URL is missing from .env")
    return QdrantClient(url=url, api_key=api_key or None)
