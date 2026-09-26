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
