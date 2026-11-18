import { useRef, useState } from "react";
import { api } from "./api.js";

const STAGES = [
  { key: "upload", label: "Upload and segment", desc: "Extract text and split it into clauses.", run: (f) => api.upload(f), summarize: (r) => `${r.clause_count} clauses` },
  { key: "index", label: "Index clauses", desc: "Embed each clause and store it in Qdrant.", run: (_f, id) => api.index(id), summarize: (r) => `${r.indexed_clauses} indexed` },
  { key: "classify", label: "Classify obligations", desc: "Label each clause as an obligation, permission, prohibition or penalty.", run: (_f, id) => api.classify(id), summarize: (r) => `${r.classified_clauses} classified, ${r.unclassified_clauses} skipped` },
  { key: "graph", label: "Build knowledge graph", desc: "Link classified clauses in Neo4j.", run: (_f, id) => api.graph(id), summarize: (r) => `${r.nodes_created} nodes, ${r.edges_created} edges` },
  { key: "audit", label: "Audit", desc: "Find contradictions and check clauses against the EU reference corpus.", run: (_f, id) => api.audit(id), summarize: (r) => `${r.contradictions.length} contradictions, ${r.compliance_results.length} checks` },
  { key: "report", label: "Compile report", desc: "Assemble the final report.", run: (_f, id) => api.report(id), summarize: (r) => `${r.clause_count} clauses` },
];

const blank = () => STAGES.map(() => ({ status: "idle", result: "" }));

export default function Uploader({ onReport }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [mode, setMode] = useState("steps");
  const [progress, setProgress] = useState(blank());
  const [docId, setDocId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);

  const pick = (f) => {
    if (!f) return;
    setFile(f);
    setProgress(blank());
    setDocId(null);
    setError(null);
    setPreview(null);
  };

  const patch = (i, p) => setProgress((cur) => cur.map((s, j) => (j === i ? { ...s, ...p } : s)));
  const failedAt = progress.findIndex((s) => s.status === "failed");

  async function runSteps(start = 0) {
    setBusy(true);
    setError(null);
    let id = start === 0 ? null : docId;
    try {
      for (let i = start; i < STAGES.length; i++) {
        patch(i, { status: "running", result: "" });
        try {
          const res = await STAGES[i].run(file, id);
          if (i === 0) {
            id = res.document_id;
            setDocId(id);
          }
          patch(i, { status: "done", result: STAGES[i].summarize(res) });
          if (i === STAGES.length - 1) onReport(res);
        } catch (e) {
          patch(i, { status: "failed", result: "" });
          throw e;
        }
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function runOneCall() {
    setBusy(true);
    setError(null);
    setProgress(blank());
    try {
      onReport(await api.runPipeline(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function doPreview() {
    setBusy(true);
    setError(null);
    try {
      setPreview(await api.previewClauses(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const start = () => (mode === "steps" ? runSteps(0) : runOneCall());
  const showStages = mode === "steps" && progress.some((s) => s.status !== "idle");

  return (
    <section className="panel" aria-labelledby="audit-title">
      <h2 id="audit-title">Audit a document</h2>
      <p className="lede">Upload a contract or policy as PDF or DOCX. Each clause is classified, checked against the others for contradictions, and compared with EU regulatory text.</p>

      <div
        className={`drop ${over ? "over" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), inputRef.current?.click())}
        onDragOver={(e) => (e.preventDefault(), setOver(true))}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => (e.preventDefault(), setOver(false), pick(e.dataTransfer.files?.[0]))}
      >
        <strong>{file ? "Choose a different file" : "Drop a file here, or click to browse"}</strong>
        <span>PDF or DOCX</span>
        <input ref={inputRef} type="file" accept=".pdf,.docx" onChange={(e) => pick(e.target.files?.[0])} />
      </div>

      {file && (
        <>
          <div className="picked">
            <span className="fname">{file.name}</span>
            <span className="cid">{(file.size / 1024).toFixed(0)} KB</span>
          </div>
          <div className="row">
            <div className="seg" role="group" aria-label="Run mode">
              <button aria-pressed={mode === "steps"} onClick={() => setMode("steps")} disabled={busy}>Show each stage</button>
              <button aria-pressed={mode === "one"} onClick={() => setMode("one")} disabled={busy}>Run in one call</button>
            </div>
            <button className="btn" onClick={start} disabled={busy}>{busy ? "Running…" : "Run audit"}</button>
            <button className="btn ghost" onClick={doPreview} disabled={busy}>Preview clauses</button>
            {failedAt >= 0 && !busy && docId && (
              <button className="btn ghost" onClick={() => runSteps(failedAt)}>Retry from “{STAGES[failedAt].label}”</button>
            )}
          </div>
        </>
      )}

      {mode === "one" && busy && <p className="note">Running every stage on the server. Large documents can take several minutes, especially with a local model.</p>}

      {showStages && (
        <ul className="stages">
          {STAGES.map((s, i) => (
            <li key={s.key} className={`stage ${progress[i].status}`}>
              <span className="mark" aria-hidden="true" />
              <div>
                <div className="label">{s.label}</div>
                <div className="desc">{s.desc}</div>
              </div>
              <div className="result">
                {progress[i].status === "running" && "Working…"}
                {progress[i].status === "failed" && "Failed"}
                {progress[i].status === "done" && progress[i].result}
              </div>
            </li>
          ))}
        </ul>
      )}

      {error && <div className="error" role="alert">{error}</div>}

      {preview && (
        <div style={{ marginTop: 22 }}>
          <h3>{preview.clause_count} clauses found in {preview.filename}</h3>
          <p className="note" style={{ marginTop: 4 }}>Nothing was saved. Check that the boundaries look right before running the audit.</p>
          <div className="list" style={{ marginTop: 10 }}>
            {preview.clauses.map((c) => (
              <div className="item" key={c.id}>
                <div className="top"><span className="cid">{c.label || c.id}</span></div>
                <div className="text clamp">{c.text}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
