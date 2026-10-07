# Doc IA

Projet local de traitement documentaire avec RAG, modèles open-source et interface web.

## Phase 1 : FastAPI + React/Vite

### Stack
- Python backend : FastAPI
- Frontend : React + Vite
- Vector store : Chroma
- Embeddings : sentence-transformers/all-MiniLM-L6-v2
- Génération locale : Qwen2.5-1.5B-Instruct

### Démarrage backend

```powershell
cd C:\Users\listo\Documents\Doc_IA
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Démarrage frontend

```powershell
cd C:\Users\listo\Documents\Doc_IA\frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

### Accès
- Frontend : http://localhost:5173
- API : http://localhost:8000/health

### Endpoints principaux
- GET /health
- POST /api/upload
- POST /api/query
- POST /api/index

### Notes
- La phase 1 ne contient pas de base SQL relationnelle.
- Le stockage documentaire est géré localement sur disque et via Chroma.
- Les modèles locaux sont chargés depuis `models/llm` et `models/embeddings`.
