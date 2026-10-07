import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


# Place for storing all reusable project settings in one place
@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    output_data_dir: Path
    provider: str
    openai_api_key: str | None
    embedding_model: str
    chat_model: str
    local_embedding_model_path: Path
    local_llm_model_path: Path
    retrieval_distance_threshold: float


# Load AppSettings
def _resolve_local_qwen_path(project_root: Path) -> Path:
    configured = os.getenv("LOCAL_LLM_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    project_candidate = project_root / "models" / "llm"
    candidates = [
        project_candidate / "Qwen2.5-1.5B-Instruct",
        project_candidate / "Qwen2.5-3B-Instruct",
        project_candidate / "models--Qwen--Qwen2.5-1.5B-Instruct",
        project_candidate / "models--Qwen--Qwen2.5-3B-Instruct",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        matches = sorted(hf_cache.glob("**/*Qwen*"))
        for match in matches:
            if match.is_dir() and (match / "config.json").exists():
                return match
            if match.is_dir() and (match / "snapshots").exists():
                return match

    return project_candidate / "Qwen2.5-1.5B-Instruct"


def load_settings() -> AppSettings:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    local_embedding_model_path = Path(
        os.getenv(
            "LOCAL_EMBEDDING_MODEL_PATH",
            project_root / "models" / "embeddings" / "all-MiniLM-L6-v2",
        )
    ).expanduser().resolve()
    local_llm_model_path = _resolve_local_qwen_path(project_root)

    return AppSettings(
        project_root=project_root,
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        output_data_dir=project_root / "data" / "output",
        provider=os.getenv("MODEL_PROVIDER", "local"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chat_model=os.getenv(
            "OPENAI_CHAT_MODEL",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        local_embedding_model_path=local_embedding_model_path,
        local_llm_model_path=local_llm_model_path,
        retrieval_distance_threshold=float(
            os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "1.25")
        ),
    )


# Validate Open AI Settings
def validate_openai_settings(settings: AppSettings) -> bool:
    return bool(settings.openai_api_key)


def validate_local_settings(settings: AppSettings) -> bool:
    return settings.provider == "local" and (
        settings.local_embedding_model_path.exists()
        or bool(settings.embedding_model)
    )


def validate_local_llm_settings(settings: AppSettings) -> bool:
    if settings.provider != "local":
        return False
    return settings.local_llm_model_path.exists() or (settings.local_llm_model_path.parent.exists())