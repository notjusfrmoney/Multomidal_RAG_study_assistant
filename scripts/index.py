import json

from qdrant_client.models import Distance, PointStruct, VectorParams

from src.study_assistant.config import settings
from src.study_assistant.embeddings import embed_texts, embedding_dimension
from src.study_assistant.ids import stable_id
from src.study_assistant.store import qdrant_client


def index_records() -> int:
    records = json.loads((settings.processed_dir / "records.json").read_text(encoding="utf-8"))
    client = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    dimension = embedding_dimension(settings.embedding_model)
    collections = {item.name for item in client.get_collections().collections}
    if settings.collection_name not in collections:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
    info = client.get_collection(settings.collection_name)
    configured_size = info.config.params.vectors.size
    if configured_size != dimension:
        raise RuntimeError(
            f"Qdrant collection {settings.collection_name!r} has vector size {configured_size}, "
            f"but {settings.embedding_model} produces {dimension}"
        )
    vectors = embed_texts([record["text"] for record in records], settings.embedding_model)
    points = [
        PointStruct(id=stable_id(record["chunk_id"]), vector=vector, payload=record)
        for record, vector in zip(records, vectors)
    ]
    for start in range(0, len(points), 10):
        client.upsert(
            collection_name=settings.collection_name,
            points=points[start : start + 10],
            wait=True,
        )
    return len(points)


if __name__ == "__main__":
    print(f"Indexed {index_records()} records")
