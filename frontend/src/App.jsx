import { useState } from 'react';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState('Résume-moi les principaux engagements contenus dans ces documents.');
  const [topK, setTopK] = useState(5);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [status, setStatus] = useState('Prêt');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [stats, setStats] = useState(null);

  const handleFileChange = (event) => {
    setFiles(Array.from(event.target.files));
  };

  const uploadFiles = async () => {
    if (!files.length) {
      setStatus('Veuillez sélectionner au moins un PDF.');
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    setIsUploading(true);
    setStatus('Téléversement des PDF...');

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Erreur pendant le téléversement.');
      }

      setStatus(`Fichiers téléversés : ${payload.saved_files.length}`);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setIsUploading(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) {
      setStatus('La question est vide.');
      return;
    }

    setIsQuerying(true);
    setStatus('Recherche dans le dépôt local...');

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question, top_k: topK }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Erreur pendant la recherche.');
      }

      setAnswer(payload.answer || 'Aucune réponse trouvée.');
      setSources(payload.sources || []);
      setStats(payload.stats || null);
      setStatus(`Réponse générée avec ${payload.source_count} sources.`);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-images">
            <img src="/ALO.PA_BIG.svg" alt="ALSTOM" className="brand-logo brand-logo-large" />
            <img src="/ALO.PA.svg" alt="ALSTOM" className="brand-logo brand-logo-small" />
          </div>
          <div className="brand-copy">
            <span className="eyebrow">Document Intelligence</span>
            <h1>ALSTOM</h1>
          </div>
        </div>
        <p>Analyse locale de documents pour la conformité, la due diligence et les politiques internes.</p>

        <label className="upload-box">
          <span>Choisir des PDFs</span>
          <input type="file" accept=".pdf" multiple onChange={handleFileChange} />
        </label>

        <button className="primary" onClick={uploadFiles} disabled={isUploading || !files.length}>
          {isUploading ? 'Téléversement...' : 'Téléverser les PDFs'}
        </button>

        <div className="field">
          <label htmlFor="topk">Nombre de chunks</label>
          <input
            id="topk"
            type="number"
            min="1"
            max="20"
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value || 1))}
          />
        </div>
      </aside>

      <main className="main-panel">
        <div className="card hero-card">
          <div className="card-header">
            <div>
              <span className="pill">Contrôle documentaire</span>
              <h2>Question métier</h2>
            </div>
          </div>
          <textarea
            rows="5"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Posez une question sur les documents..."
          />
          <div className="actions">
            <button className="primary" onClick={askQuestion} disabled={isQuerying}>
              {isQuerying ? 'Recherche...' : 'Rechercher'}
            </button>
          </div>
        </div>

        <div className="card status-card">
          <h2>Statut</h2>
          <p>{status}</p>
        </div>

        <div className="card answer-card">
          <h2>Réponse</h2>
          <div className="answer-box">{answer || 'Aucune réponse pour le moment.'}</div>
        </div>

        {stats && (
          <div className="card metrics-card">
            <h2>Statistiques</h2>
            <pre>{JSON.stringify(stats, null, 2)}</pre>
          </div>
        )}

        {sources.length > 0 && (
          <div className="card source-card">
            <h2>Sources</h2>
            <ul className="source-list">
              {sources.map((source, index) => (
                <li key={`${source.chunk_id}-${index}`}>
                  <strong>{source.claim_id || 'Claim inconnue'}</strong>
                  <div>{source.source_file || 'Fichier inconnu'}</div>
                  <small>Distance : {source.distance}</small>
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
