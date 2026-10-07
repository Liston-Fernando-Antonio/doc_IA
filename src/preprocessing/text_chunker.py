import json
from pathlib import Path

REQUIRED_CHUNKS_KEY = {
    "claim_id",
    "chunk_id",
    "chunk_index",
    "source_file",
    "text",
    "text_length"
}

#Create smaller text chunks

def create_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """
    couper le long texte en morceaux plus petits avec chevauchement.
    Args:
        text (str): Le texte à diviser en morceaux.
        chunk_size (int): La taille maximale de chaque morceau.
        overlap (int): Le nombre de caractères qui se chevauchent entre les morceaux.
        cela permet de conserver le contexte entre les morceaux.
    """
    if chunk_size <= 0:
        raise ValueError("La taille du morceau doit être supérieure à zéro.")
    
    if (overlap >= chunk_size):
        raise ValueError("Le chevauchement doit être inférieur à la taille du morceau chunk_size.")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap
    return chunks

#build the chunk record
def build_chunk_records(
        claim_id: str,
        source_file: str | Path,
        chunks: list[str]
        )-> list[dict]:
    """
    Convertir les morceaux de texte en enregistrements de morceaux avec des métadonnées.
    les métadonnées aide le rag pipeline à retrouver le contexte du morceau.
    """
    source_file_str = str(source_file)
    chunk_records = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_records.append(
            {
                "claim_id": claim_id,
                "chunk_id": f"{claim_id}_chunk_{index:03d}",
                "chunk_index": index,
                "source_file": source_file_str,
                "text": chunk,
                "text_length": len(chunk)
            }
        )
    return chunk_records

#appliquer le chunking sur le data nettoyé
def chunk_cleaned_claims(
        processed_dir: Path,
        chunk_size: int = 1000,
        overlap: int = 150
        ) -> list[Path]:
    """
    Creer un RAG-ready chunks from every cleaned text file.

    Inputs files are read from data/processed/cleaned_claims. 
    Chunks JSON files are saved into data/processed/chunks with one JSON file per claim.
    """

    cleaned_claims_dir = processed_dir / "cleaned_claims"
    chunks_output_dir = processed_dir / "chunks"
    chunks_output_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = []

    if not cleaned_claims_dir.exists():
        print(f"Le répertoire {cleaned_claims_dir} n'existe pas. Aucun fichier à traiter.")
        return chunk_files

    cleaned_files = sorted(cleaned_claims_dir.glob("*.txt"))

    if not cleaned_files:
        print(f"Aucun fichier texte nettoyé trouvé dans {cleaned_claims_dir}.")
        return chunk_files

    for cleaned_file in cleaned_files:
        claim_id = cleaned_file.stem
        output_path = chunks_output_dir / f"{claim_id}_chunks.json"

        if output_path.exists():
            print(f"Skipping {output_path} as it already exists.")
            chunk_files.append(output_path)
            continue

        cleaned_text = cleaned_file.read_text(encoding="utf-8")
        chunks = create_text_chunks(
            text=cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap
        )
        chunk_records = build_chunk_records(
            claim_id=claim_id,
            source_file=cleaned_file,
            chunks=chunks
        )

        output_path.write_text(
            json.dumps(chunk_records, indent=2),
            encoding="utf-8"
        )

        chunk_files.append(output_path)
        print(f"Chunks file created: {output_path} with {len(chunk_records)} chunks.")
    return chunk_files

#Validate the chunk records 
def validate_chunk_records(chunk_records: list[dict]) -> dict:
    """
    Valider une liste d'enregistrements de morceaux et retourner un dictionnaire.
    Pour confirmer que chaque enregistrement contient les metadonnées requises et une taille resonable avant de l'insertion.
    """
    missing_metadata_count = 0
    empty_text_count = 0
    chunk_lengths = []

    for chunk_record in chunk_records:
        missing_fields = REQUIRED_CHUNKS_KEY - set(chunk_record)

        if missing_fields:
            missing_metadata_count += 1

        chunk_text = chunk_record.get("text", "")

        if not chunk_text.strip():
            empty_text_count += 1

        chunk_lengths.append(len(chunk_text))

    return {
        "chunks_count": len(chunk_records),
        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
        "avg_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 2) 
        if chunk_lengths 
        else 0,
        "missing_metadata_count": missing_metadata_count,
        "empty_text_count": empty_text_count,
        "is_valid": missing_metadata_count == 0 and empty_text_count == 0,
    }



# Validate chunk files
def validate_chunk_files(processed_dir: Path) -> list[Path]:
    """
    Create a validation report for each chunk JSON file.

    Reports are saved into data/processed/chunk_reports so students can inspect
    chunk quality before moving to embeddings and vector search.
    """
    chunks_dir = processed_dir / "chunks"
    reports_dir = processed_dir / "chunk_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_files = []

    if not chunks_dir.exists():
        print(f"Auncun dossier chunks trouver dans {chunks_dir}")
        return report_files
    
    chunk_files = sorted(chunks_dir.glob("*_chunks.json"))

    if not chunk_files:
        print(f"Aucun fichier json trouvé dans {chunks_dir}")
        return report_files
    
    for chunk_file in chunk_files:
        output_path = reports_dir / f"{chunk_file.stem}_report.json"

        if output_path.exists():
            print(f"Skipping existing chunk validation report: {output_path}")
            report_files.append(output_path)
            continue

        chunk_records = json.loads(chunk_file.read_text(encoding="utf-8"))
        report = validate_chunk_records(chunk_records)
        report["chunk_file"] = chunk_file.name

        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_files.append(output_path)

        print(
            "chunk validation report created: ",
            f"{output_path} (valid={report['is_valid']})"
            )
        
    return report_files