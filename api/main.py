from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.rag_service import LocalRAGService

app = FastAPI(title="Doc IA API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = LocalRAGService()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "Doc IA local backend"}


@app.post("/api/index")
def index_documents() -> dict[str, Any]:
    try:
        return service.run_pipeline(question="Résume-moi les principaux engagements contenus dans ces documents.", top_k=5)
    except Exception as exc:  # pragma: no cover - API protection
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/query")
def query_documents(payload: QueryRequest) -> dict[str, Any]:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    try:
        return service.run_pipeline(question=payload.question, top_k=payload.top_k)
    except Exception as exc:  # pragma: no cover - API protection
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier PDF fourni.")

    saved: list[str] = []
    for file in files:
        filename = file.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Le fichier {filename} n'est pas un PDF.")

        target = service.raw_dir / filename
        contents = await file.read()
        target.write_bytes(contents)
        saved.append(str(target))

    return {
        "status": "uploaded",
        "saved_files": saved,
        "message": "Fichiers reçus. Vous pouvez maintenant lancer l'indexation.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
