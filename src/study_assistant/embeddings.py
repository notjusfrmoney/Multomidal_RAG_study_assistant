from functools import lru_cache


@lru_cache(maxsize=1)
def _model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for embeddings") from exc
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model(model_name).encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embedding_dimension(model_name: str) -> int:
    return int(_model(model_name).get_sentence_embedding_dimension())
