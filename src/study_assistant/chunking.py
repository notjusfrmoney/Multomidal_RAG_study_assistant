import re
from collections.abc import Iterable


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    words = clean_text(text).split()
    step = chunk_size - overlap
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step) if words[start : start + chunk_size]]


def make_chunk_records(pages: Iterable[tuple[int, str]], document_id: str, source_file: str) -> list[dict]:
    records = []
    for page_number, text in pages:
        for index, chunk in enumerate(chunk_text(text)):
            records.append(
                {
                    "chunk_id": f"{document_id}-p{page_number}-{index}",
                    "document_id": document_id,
                    "record_type": "text",
                    "source_file": source_file,
                    "page_start": page_number,
                    "page_end": page_number,
                    "text": chunk,
                }
            )
    return records
