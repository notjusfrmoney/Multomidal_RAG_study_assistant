import argparse
from pathlib import Path

from src.study_assistant.citations import format_citation
from src.study_assistant.config import settings
from src.study_assistant.generation import generate_answer
from src.study_assistant.ingest import ingest_all
from src.study_assistant.retrieval import qdrant_vector_store, search
from scripts.index import index_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", required=True)
    args = parser.parse_args()
    chapter = Path(args.chapter)
    if not chapter.is_absolute():
        chapter = settings.raw_dir / chapter
    records = ingest_all([chapter])
    print(f"ingestion_records={len(records)}")
    print(f"indexed_records={index_records()}")

    client = qdrant_vector_store(
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.collection_name,
        settings.embedding_model,
    )
    questions = [
        "What is electric flux?",
        "Explain Gauss's law and its physical meaning.",
        "What is the electric field due to a uniformly charged sphere?",
    ]
    for question in questions:
        results = search(client, settings.collection_name, question, settings.embedding_model, settings.top_k)
        text_hits = sum(result.record_type == "text" for result in results)
        page_hits = sum(result.record_type == "page" for result in results)
        print(f"\nQUESTION: {question}")
        print(f"text_hits={text_hits} page_hits={page_hits}")
        for result in results[:3]:
            print(f"- {format_citation(result)} [{result.record_type}] score={result.score:.3f}")
        print(generate_answer(question, results, settings.groq_api_key, settings.generation_model))


if __name__ == "__main__":
    main()
