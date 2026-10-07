from __future__ import annotations

import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.settings import AppSettings, validate_local_llm_settings


def normalize_context_chunks(context_chunks: list[str]) -> list[str]:
    seen = set()
    normalized_chunks: list[str] = []
    for chunk in context_chunks:
        if not chunk or not str(chunk).strip():
            continue
        unique = unicodedata.normalize("NFKC", chunk).replace("\r", " ")
        key = "\n".join(line.strip() for line in unique.splitlines() if line.strip())[:1800]
        if key in seen:
            continue
        seen.add(key)
        normalized_chunks.append(unique)
    return normalized_chunks


def resolve_local_llm_model_path(settings: AppSettings) -> Path:
    model_path = settings.local_llm_model_path
    if model_path.exists():
        return model_path

    project_llm_dir = settings.project_root / "models" / "llm"
    if project_llm_dir.exists():
        for candidate in sorted(project_llm_dir.rglob("config.json")):
            if "Qwen" in candidate.parent.name or "Qwen" in str(candidate.parent):
                return candidate.parent

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        for candidate in sorted(hf_cache.rglob("config.json")):
            if "Qwen" in candidate.parent.name or "Qwen" in str(candidate.parent):
                return candidate.parent

    return model_path


def load_local_qwen(settings: AppSettings):
    if not validate_local_llm_settings(settings):
        raise FileNotFoundError(
            "Local Qwen model is not available. Download it first in models/llm."
        )

    model_path = resolve_local_llm_model_path(settings)
    print(f"Loading local Qwen model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    cleaned_context_chunks = normalize_context_chunks(context_chunks)
    if not cleaned_context_chunks:
        return (
            "Tu es un assistant expert en intelligence documentaire. "
            "Réponds en français. Le contexte fourni est vide ou non pertinent. "
            "Indique clairement que l'information n'est pas disponible dans les documents.\n\n"
            f"Question: {question}\n\nRéponse:"
        )

    context = "\n\n---\n\n".join(
        f"[Document extrait]\n{chunk[:1800]}" for chunk in cleaned_context_chunks if chunk.strip()
    )
    return (
        "Tu es un assistant expert en intelligence documentaire et RAG. "
        "Réponds en français, de façon claire et concise. "
        "Utilise uniquement le contexte fourni. "
        "Si l'information n'est pas présente, dis-le explicitement. "
        "Reste factuel, évite les spéculations, et mets en avant les éléments les plus probants.\n\n"
        "Structure recommandée : 1) réponse courte, 2) puis 2 à 4 points clés, 3) mentionne les limites si besoin.\n\n"
        f"Question: {question}\n\nContexte:\n{context}\n\nRéponse:"
    )


def generate_answer(question: str, context_chunks: list[str], settings: AppSettings):
    tokenizer, model = load_local_qwen(settings)
    prompt = build_rag_prompt(question, context_chunks)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.split("\n\nRéponse:", 1)[-1].strip() if "\n\nRéponse:" in answer else answer.strip()
    return unicodedata.normalize("NFKC", answer)
