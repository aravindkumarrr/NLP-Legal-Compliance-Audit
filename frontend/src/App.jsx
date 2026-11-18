import { useState } from "react";
import { api, recent } from "./api.js";
import Health from "./Health.jsx";
import Uploader from "./Uploader.jsx";
import Report from "./Report.jsx";

export default function App() {
  const [report, setReport] = useState(null);
  const [history, setHistory] = useState(recent.list());
  const [loadingId, setLoadingId] = useState(null);
  const [error, setError] = useState(null);

  const show = (r) => {
    setReport(r);
    setError(null);
    setHistory(
      recent.add({
        document_id: r.document_id,
        filename: r.filename,
        gaps: r.compliance_gap_count,
        contradictions: r.contradiction_count,
        at: Date.now(),
      })
    );
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  async function open(id) {
    setLoadingId(id);
    setError(null);
    try {
      setReport(await api.report(id));
    } catch (e) {
      setError(`Couldn't load that report: ${e.message}`);
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">Legal Compliance Auditor</span>
        <Health />
      </header>

      <div className="layout">
        <aside className="side">
          <h3>Recent audits</h3>
          {history.length === 0 ? (
            <p className="empty">Audits you run in this browser will appear here.</p>
          ) : (
            <ul className="recent">
              {history.map((h) => (
                <li key={h.document_id}>
                  <button className={`open ${report?.document_id === h.document_id ? "active" : ""}`} onClick={() => open(h.document_id)} disabled={loadingId === h.document_id}>
                    <span className="name">{h.filename}</span>
                    <span className="meta">
                      {loadingId === h.document_id ? "Loading…" : `${h.contradictions} contradictions · ${h.gaps} gaps`}
                    </span>
                  </button>
                  <button className="x" aria-label={`Remove ${h.filename} from the list`} onClick={() => setHistory(recent.remove(h.document_id))}>×</button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <main>
          {error && <div className="error" role="alert" style={{ marginTop: 0, marginBottom: 20 }}>{error}</div>}
          <Uploader onReport={show} />
          {report && <Report key={report.document_id} report={report} />}
        </main>
      </div>
    </div>
  );
}
