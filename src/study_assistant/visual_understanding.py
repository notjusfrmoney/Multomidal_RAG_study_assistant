import base64
import json
from pathlib import Path

from .config import settings


def _image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def describe_page(image_path: Path, model_name: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing; page visual descriptions cannot be generated")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is required for page visual descriptions") from exc
    response = Groq(api_key=api_key).chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this textbook page for semantic retrieval. "
                            "Mention diagrams, equations, labels, tables, and the main concept. "
                            "Be concise and do not invent details."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=300,
    )
    return response.choices[0].message.content or ""


def describe_pages(document_id: str, image_paths: list[Path], metadata_path: Path) -> dict[str, str]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    for page_number, image_path in enumerate(image_paths, start=1):
        key = f"{document_id}:{page_number}"
        if key not in existing:
            existing[key] = describe_page(image_path, settings.vlm_model, settings.groq_api_key)
            metadata_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return {str(page_number): existing[f"{document_id}:{page_number}"] for page_number in range(1, len(image_paths) + 1)}
