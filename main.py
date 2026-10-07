import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.config.settings import load_settings
from src.ingestion.pdf_loader import combine_claim_texts, process_pdfs
from src.preprocessing.text_chunker import chunk_cleaned_claims, validate_chunk_files
from src.preprocessing.text_cleaner import clean_combined_claims
from src.rag.chunk_loader import load_all_chunks, save_chunk_manifest
from src.rag.llm_local import generate_answer
from src.rag.vector_store import build_vector_store, search_vector_store


DEFAULT_QUESTION = "Résume-moi les principaux engagements contenus dans ces documents."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RAG local avec Qwen et MiniLM")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question à poser au modèle local.")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre maximal de chunks à récupérer pour le contexte.")
    parser.add_argument("--skip-ingestion", action="store_true", help="Saute l'étape PDF -> texte brut.")
    parser.add_argument("--skip-indexing", action="store_true", help="Saute la création du vector store.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force la recréation des artefacts existants.")
    return parser.parse_args()


def print_step(title: str) -> None:
    print(f"\n=== {title} ===")


def main():
    """Run the local RAG pipeline with a cleaner CLI and better UX."""
    args = parse_args()
    project_root = Path(__file__).parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    settings = load_settings()

    if args.force_rebuild:
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
            print(f"Nettoyage forcé du dossier {processed_dir}")

    print_step("Pipeline RAG local (Qwen + MiniLM)")
    print(f"Question: {args.question}")
    print(f"Modèle local LLM: {settings.local_llm_model_path}")
    print(f"Modèle local embedding: {settings.local_embedding_model_path}")

    if not args.skip_ingestion:
        print_step("Étape 1/6 - Ingestion PDF")
        output_files = process_pdfs(raw_dir=raw_dir, processed_dir=processed_dir)
        print(f"Fichiers texte extraits: {len(output_files)}")
    else:
        print_step("Étape 1/6 - Ingestion PDF (skip)")

    print_step("Étape 2/6 - Combinaison par claim")
    combined_files = combine_claim_texts(processed_dir=processed_dir)
    print(f"Fichiers combinés: {len(combined_files)}")

    print_step("Étape 3/6 - Nettoyage textuel")
    cleaned_files = clean_combined_claims(processed_dir=processed_dir)
    print(f"Fichiers nettoyés: {len(cleaned_files)}")

    print_step("Étape 4/6 - Découpage en chunks")
    chunk_files = chunk_cleaned_claims(processed_dir=processed_dir)
    print(f"Fichiers chunks: {len(chunk_files)}")

    print_step("Étape 5/6 - Validation des chunks")
    report_files = validate_chunk_files(processed_dir=processed_dir)
    print(f"Rapports de validation: {len(report_files)}")

    print_step("Étape 6/6 - Indexation vectorielle et réponse")
    chunks = load_all_chunks(processed_dir=processed_dir)
    manifest_path = save_chunk_manifest(processed_dir=processed_dir, chunks=chunks)
    print(f"Manifest créé: {manifest_path is not None}")

    if chunks and not args.skip_indexing:
        build_vector_store(settings=settings, chunks=chunks)

    if chunks:
        search_results = search_vector_store(settings=settings, query=args.question, top_k=max(args.top_k, 1))
        if not search_results:
            print("Aucun chunk pertinent trouvé pour cette question.")
            return
        context_chunks = [result["text"] for result in search_results]
        answer = generate_answer(question=args.question, context_chunks=context_chunks, settings=settings)
        print("\n--- Réponse locale Qwen ---")
        print(answer)


if __name__ == "__main__":
    main()