from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(_path):
        return False


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    pages_dir: Path = ROOT / "data" / "pages"
    visual_metadata_path: Path = ROOT / "data" / "processed" / "visual_metadata.json"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "student_physics")
    top_k: int = int(os.getenv("TOP_K", "5"))
    generation_model: str = os.getenv("GENERATION_MODEL", "qwen/qwen3.8-27b")
    vlm_model: str = os.getenv("VLM_MODEL", "qwen/qwen3.8-27b")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")


settings = Settings()
