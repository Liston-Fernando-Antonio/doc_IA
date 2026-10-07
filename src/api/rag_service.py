from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.settings import load_settings
from src.ingestion.pdf_loader import combine_claim_texts, process_pdfs
from src.preprocessing.text_chunker import chunk_cleaned_claims, validate_chunk_files
from src.preprocessing.text_cleaner import clean_combined_claims
from src.rag.chunk_loader import load_all_chunks, save_chunk_manifest
from src.rag.llm_local import generate_answer
from src.rag.vector_store import build_vector_store, search_vector_store


class LocalRAGService:
    def __init__(self, root_dir: str | Path | None = None):
        project_root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]
        self.project_root = project_root
        self.raw_dir = project_root / "data" / "raw"
        self.processed_dir = project_root / "data" / "processed"
        self.settings = load_settings()

    def _ensure_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, question: str, top_k: int = 5) -> dict[str, Any]:
        self._ensure_directories()

        extracted_files = process_pdfs(raw_dir=self.raw_dir, processed_dir=self.processed_dir)
        combined_files = combine_claim_texts(processed_dir=self.processed_dir)
        cleaned_files = clean_combined_claims(processed_dir=self.processed_dir)
        chunk_files = chunk_cleaned_claims(processed_dir=self.processed_dir)
        validation_reports = validate_chunk_files(processed_dir=self.processed_dir)

        chunks = load_all_chunks(processed_dir=self.processed_dir)
        manifest = save_chunk_manifest(processed_dir=self.processed_dir, chunks=chunks)

        if chunks:
            build_vector_store(settings=self.settings, chunks=chunks)

        search_results = []
        answer = "Aucun document n'a été trouvé dans la base locale."
        if chunks:
            search_results = search_vector_store(
                settings=self.settings,
                query=question,
                top_k=max(top_k, 1),
            )
            context_chunks = [result["text"] for result in search_results]
            answer = generate_answer(
                question=question,
                context_chunks=context_chunks,
                settings=self.settings,
            )

        return {
            "question": question,
            "answer": answer,
            "source_count": len(search_results),
            "sources": [
                {
                    "chunk_id": item["chunk_id"],
                    "claim_id": item["metadata"].get("claim_id"),
                    "source_file": item["metadata"].get("source_file"),
                    "distance": round(float(item.get("distance", 0.0)), 4),
                }
                for item in search_results
            ],
            "stats": {
                "pdfs_processed": len(extracted_files),
                "combined_files": len(combined_files),
                "cleaned_files": len(cleaned_files),
                "chunk_files": len(chunk_files),
                "validation_reports": len(validation_reports),
                "manifest_path": str(manifest) if manifest else None,
            },
        }
