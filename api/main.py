from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.study_assistant.config import settings
from src.study_assistant.orchestrator import process_student_query
from src.study_assistant.retrieval import qdrant_vector_store


app = FastAPI(title="Multimodal CBSE Study Assistant")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    chapter: str | None = None
    chat_history: List[dict] = Field(default_factory=list)


class Source(BaseModel):
    page_number: int
    source_file: str
    image_path: str | None = None
    record_type: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    used_retrieval: bool
    sources: list[Source] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        qdrant_vector_store(
            settings.qdrant_url,
            settings.qdrant_api_key,
            settings.collection_name,
            settings.embedding_model,
        ).client.get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = process_student_query(
            request.question,
            request.chat_history,
            request.chapter,
        )
        return QueryResponse(
            answer=result["answer"],
            intent=result["intent"],
            used_retrieval=result["used_retrieval"],
            sources=[
                Source(
                    page_number=result.page_start,
                    source_file=result.source_file,
                    image_path=result.page_image,
                    record_type=result.record_type,
                )
                for result in result["sources"]
            ],
            debug=result["debug"],
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
