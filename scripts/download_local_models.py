from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "models" / "llm"
EMBEDDINGS_DIR = PROJECT_ROOT / "models" / "embeddings"
LLM_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def download_qwen() -> Path:
    target_dir = LLM_DIR / "Qwen2.5-3B-Instruct"
    print(f"Téléchargement du modèle Qwen dans: {target_dir}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        local_files_only=False,
    )
    print(f"Modèle Qwen prêt dans: {target_dir}")
    return target_dir


def download_embedding() -> Path:
    target_dir = EMBEDDINGS_DIR / "all-MiniLM-L6-v2"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de l'embedding dans: {target_dir}")

    model = SentenceTransformer(EMBEDDING_MODEL_ID)
    model.save(str(target_dir))
    print(f"Embedding prêt: {target_dir}")
    return target_dir


if __name__ == "__main__":
    print("Début du téléchargement des modèles locaux...")
    try:
        download_qwen()
    except Exception as exc:
        print(f"Erreur lors du téléchargement de Qwen: {exc}")

    try:
        download_embedding()
    except Exception as exc:
        print(f"Erreur lors du téléchargement de l'embedding: {exc}")

    print("Téléchargement terminé.")
