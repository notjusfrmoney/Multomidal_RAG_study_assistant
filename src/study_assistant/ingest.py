import json
from .visual_understanding import describe_pages
from pathlib import Path

from .config import settings


def pdf_files(raw_dir: Path = settings.raw_dir) -> list[Path]:
    return sorted(raw_dir.rglob("*.pdf"))


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install dependencies before ingesting PDFs: pip install -r requirements.txt") from exc
    reader = PdfReader(str(pdf_path))
    return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]


def render_pages(pdf_path: Path, output_dir: Path = settings.pages_dir) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render textbook pages") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    document_id = pdf_path.stem
    rendered = []
    document = fitz.open(pdf_path)
    for index, page in enumerate(document):
        output_path = output_dir / f"{document_id}_p{index + 1}.png"
        if not output_path.exists():
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(output_path)
        rendered.append(output_path)
    return rendered


def ingest_all(pdf_paths: list[Path] | None = None, describe_visuals: bool = True) -> list[dict]:
    from .chunking import make_chunk_records

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pdf_path in pdf_paths or pdf_files():
        document_id = pdf_path.stem
        pages = extract_pages(pdf_path)
        page_images = render_pages(pdf_path)
        visual_descriptions = describe_pages(
            document_id,
            page_images,
            settings.visual_metadata_path,
        ) if describe_visuals else {}
        records.extend(make_chunk_records(pages, document_id, str(pdf_path.relative_to(settings.raw_dir))))
        records.extend(
            {
                "chunk_id": f"{document_id}-page-{page_number}",
                "document_id": document_id,
                "record_type": "page",
                "source_file": str(pdf_path.relative_to(settings.raw_dir)),
                "page_start": page_number,
                "page_end": page_number,
                "text": visual_descriptions.get(str(page_number), text),
                "page_image": str(page_images[page_number - 1]),
            }
            for page_number, text in pages
        )
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        (settings.processed_dir / "records.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    output_path = settings.processed_dir / "records.json"
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    print(f"Created {len(ingest_all())} records")
