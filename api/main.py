from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.study_assistant.config import settings
from src.study_assistant.generation import generate_answer
from src.study_assistant.retrieval import search
from src.study_assistant.store import qdrant_client


app = FastAPI(title="Multimodal CBSE Study Assistant")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    chapter: str | None = None


class Source(BaseModel):
    page_number: int
    source_file: str
    image_path: str | None = None
    record_type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    try:
        qdrant_client(settings.qdrant_url, settings.qdrant_api_key).get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        client = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        results = search(
            client,
            settings.collection_name,
            request.question,
            settings.embedding_model,
            settings.top_k,
        )
        if request.chapter:
            results = [
                result
                for result in results
                if request.chapter.lower() in result.source_file.lower()
            ]
        answer = generate_answer(
            request.question,
            results,
            settings.groq_api_key,
            settings.generation_model,
        )
        return QueryResponse(
            answer=answer,
            sources=[
                Source(
                    page_number=result.page_start,
                    source_file=result.source_file,
                    image_path=result.page_image,
                    record_type=result.record_type,
                )
                for result in results
            ],
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
