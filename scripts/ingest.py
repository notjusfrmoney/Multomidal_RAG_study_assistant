import argparse
from pathlib import Path

from src.study_assistant.config import settings
from src.study_assistant.ingest import ingest_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=str, help="PDF filename or path relative to data/raw")
    args = parser.parse_args()
    pdf_paths = None
    if args.chapter:
        pdf_paths = [Path(args.chapter)]
        if not pdf_paths[0].is_absolute():
            pdf_paths[0] = settings.raw_dir / pdf_paths[0]
        if not pdf_paths[0].exists():
            raise FileNotFoundError(f"PDF not found: {pdf_paths[0]}")
    records = ingest_all(pdf_paths=pdf_paths)
    print(f"Saved {len(records)} records")
