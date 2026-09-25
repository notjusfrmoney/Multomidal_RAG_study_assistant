from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def _model(model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        return []
    return _model(model_name).embed_documents(texts)


def embedding_dimension(model_name: str) -> int:
    return len(_model(model_name).embed_query("dimension probe"))
