from src.study_assistant.chunking import chunk_text, clean_text, make_chunk_records
from src.study_assistant.citations import format_citation
from src.study_assistant.ids import stable_id
from src.study_assistant.store import SearchResult


def test_clean_text_removes_line_break_hyphens():
    assert clean_text("electro-\nstatic   field") == "electrostatic field"


def test_chunk_text_preserves_all_words():
    chunks = chunk_text(" ".join(f"word{i}" for i in range(10)), chunk_size=4, overlap=1)
    assert "word9" in chunks[-1]


def test_metadata_and_ids_are_deterministic():
    records = make_chunk_records([(3, "A short physics explanation.")], "chapter", "chapter.pdf")
    assert records[0]["page_start"] == 3
    assert stable_id("chapter", 3) == stable_id("chapter", 3)


def test_citation_format():
    result = SearchResult("id", "page", 0.9, "evidence", 12, 12, source_file="chapter.pdf")
    assert format_citation(result) == "chapter.pdf, page 12"
