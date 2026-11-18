import { useMemo, useState } from "react";

const CATS = [
  ["Obligation", "var(--obligation)"],
  ["Permission", "var(--permission)"],
  ["Prohibition", "var(--prohibition)"],
  ["Penalty", "var(--penalty)"],
];
const STATUS_LABEL = { compliant: "Compliant", gap: "Gap", unclear: "Unclear", not_applicable: "Not applicable" };

function Text({ children }) {
  const [open, setOpen] = useState(false);
  const long = (children || "").length > 320;
  return (
    <>
      <div className={`text ${long && !open ? "clamp" : ""}`}>{children}</div>
      {long && (
        <button className="link-btn more" onClick={() => setOpen(!open)}>
          {open ? "Show less" : "Show more"}
        </button>
      )}
    </>
  );
}

function Clauses({ clauses }) {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("all");
  const shown = clauses.filter((c) => {
    if (cat !== "all" && (c.deontic_category || "none") !== cat) return false;
    return !q || c.text.toLowerCase().includes(q.toLowerCase()) || (c.deontic_entity || "").toLowerCase().includes(q.toLowerCase());
  });
  return (
    <>
      <div className="filters">
        <input aria-label="Search clauses" placeholder="Search clause text or party" value={q} onChange={(e) => setQ(e.target.value)} />
        <select aria-label="Filter by category" value={cat} onChange={(e) => setCat(e.target.value)}>
          <option value="all">All categories</option>
          {CATS.map(([c]) => <option key={c} value={c}>{c}</option>)}
          <option value="none">Unclassified</option>
        </select>
      </div>
      {shown.length === 0 ? (
        <p className="empty-state">No clauses match these filters.</p>
      ) : (
        <div className="list">
          {shown.map((c) => (
            <div className="item" key={c.id}>
              <div className="top">
                <span className="cid">{c.label || `#${c.order}`}</span>
                <span className={`badge b-${c.deontic_category || "none"}`}>{c.deontic_category || "Unclassified"}</span>
              </div>
              <Text>{c.text}</Text>
              {c.deontic_entity && <div className="entity">Applies to: {c.deontic_entity}</div>}
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function Contradictions({ items, byId }) {
  if (!items.length) return <p className="empty-state">No contradictions were found.</p>;
  return (
    <>
      <div className="list">
        {items.map((f, i) => (
          <div className="item" key={i}>
            <div className="top">
              <span className={`badge b-${f.category_a}`}>{f.category_a}</span>
              <span className="cid">vs</span>
              <span className={`badge b-${f.category_b}`}>{f.category_b}</span>
              <span className="conf">on “{f.entity}” · confidence {Math.round(f.confidence * 100)}%</span>
            </div>
            <div className="pair">
              {[f.clause_id_a, f.clause_id_b].map((id) => (
                <div className="quote" key={id}>
                  <div className="cid">{byId[id]?.label || id}</div>
                  <div className="text">{byId[id]?.text || "Clause text unavailable."}</div>
                </div>
              ))}
            </div>
            {f.explanation && <div className="why">{f.explanation}</div>}
          </div>
        ))}
      </div>
      <p className="note">Contradictions are detected structurally. Two clauses can bind the same party without actually conflicting, so read each pair before acting on it.</p>
    </>
  );
}

function Compliance({ items, byId }) {
  const [status, setStatus] = useState("all");
  const shown = items.filter((r) => status === "all" || r.status === status);
  return (
    <>
      <div className="filters">
        <select aria-label="Filter by status" value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="all">All statuses</option>
          {Object.entries(STATUS_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>
      {shown.length === 0 ? (
        <p className="empty-state">No compliance results with this status.</p>
      ) : (
        <div className="list">
          {shown.map((r) => (
            <div className="item" key={r.clause_id}>
              <div className="top">
                <span className="cid">{byId[r.clause_id]?.label || r.clause_id}</span>
                <span className={`badge b-${r.status}`}>{STATUS_LABEL[r.status] || r.status}</span>
              </div>
              <Text>{byId[r.clause_id]?.text || ""}</Text>
              <div className="why">{r.rationale}</div>
              {r.matched_reference && (
                <div className="quote" style={{ marginTop: 10 }}>
                  <div className="cid">Matched reference</div>
                  <Text>{r.matched_reference}</Text>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export default function Report({ report }) {
  const [tab, setTab] = useState("clauses");
  const byId = useMemo(() => Object.fromEntries(report.clauses.map((c) => [c.id, c])), [report]);
  const mix = useMemo(() => {
    const counts = Object.fromEntries(CATS.map(([c]) => [c, 0]));
    let none = 0;
    report.clauses.forEach((c) => (counts[c.deontic_category] !== undefined ? counts[c.deontic_category]++ : none++));
    return { counts, none };
  }, [report]);
  const total = report.clauses.length || 1;

  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `${report.filename.replace(/\.[^.]+$/, "")}-report.json` });
    a.click();
    URL.revokeObjectURL(url);
  };

  const tabs = [
    ["clauses", "Clauses", report.clauses.length],
    ["contradictions", "Contradictions", report.contradictions.length],
    ["compliance", "Compliance", report.compliance_results.length],
  ];

  return (
    <section className="panel" aria-labelledby="report-title">
      <div className="rhead">
        <div>
          <h2 id="report-title">{report.filename}</h2>
          <div className="sub">Document ID {report.document_id}</div>
        </div>
        <button className="btn ghost" onClick={download}>Download JSON</button>
      </div>

      <div className="stats">
        <div className="stat"><div className="n">{report.clause_count}</div><div className="l">Clauses</div></div>
        <div className={`stat ${report.contradiction_count ? "bad" : ""}`}><div className="n">{report.contradiction_count}</div><div className="l">Contradictions</div></div>
        <div className={`stat ${report.compliance_gap_count ? "warn" : ""}`}><div className="n">{report.compliance_gap_count}</div><div className="l">Compliance gaps</div></div>
        <div className="stat"><div className="n">{report.compliance_not_applicable_count}</div><div className="l">Not applicable</div></div>
      </div>

      <div className="mix">
        <div className="mixbar" role="img" aria-label="Clause mix by deontic category">
          {CATS.map(([c, color]) => mix.counts[c] > 0 && <div key={c} style={{ width: `${(mix.counts[c] / total) * 100}%`, background: color }} title={`${c}: ${mix.counts[c]}`} />)}
          {mix.none > 0 && <div style={{ width: `${(mix.none / total) * 100}%`, background: "var(--line-strong)" }} title={`Unclassified: ${mix.none}`} />}
        </div>
        <div className="legend">
          {CATS.map(([c, color]) => <span key={c}><span className="sw" style={{ background: color }} />{c} {mix.counts[c]}</span>)}
          <span><span className="sw" style={{ background: "var(--line-strong)" }} />Unclassified {mix.none}</span>
        </div>
      </div>

      <div className="tabs" role="tablist">
        {tabs.map(([k, label, n]) => (
          <button key={k} role="tab" className="tab" aria-selected={tab === k} onClick={() => setTab(k)}>
            {label}<span className="count">{n}</span>
          </button>
        ))}
      </div>

      <div role="tabpanel">
        {tab === "clauses" && <Clauses clauses={report.clauses} />}
        {tab === "contradictions" && <Contradictions items={report.contradictions} byId={byId} />}
        {tab === "compliance" && <Compliance items={report.compliance_results} byId={byId} />}
      </div>
    </section>
  );
}
