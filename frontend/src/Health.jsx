import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";

const SERVICES = [
  ["postgres", "Postgres"],
  ["qdrant", "Qdrant"],
  ["neo4j", "Neo4j"],
];

export default function Health() {
  const [state, setState] = useState({ loading: true, data: null, error: null });

  const check = useCallback(async () => {
    setState((s) => ({ ...s, loading: true }));
    try {
      setState({ loading: false, data: await api.health(), error: null });
    } catch (e) {
      setState({ loading: false, data: null, error: e.message });
    }
  }, []);

  useEffect(() => {
    check();
  }, [check]);

  return (
    <div className="health" aria-live="polite">
      {state.error ? (
        <span className="health-item" title={state.error}>
          <span className="dot bad" /> API offline
        </span>
      ) : (
        SERVICES.map(([key, label]) => {
          const v = state.data?.[key];
          const ok = v === "ok";
          return (
            <span className="health-item" key={key} title={v && !ok ? v : undefined}>
              <span className={`dot ${state.data ? (ok ? "ok" : "bad") : ""}`} />
              {label}
            </span>
          );
        })
      )}
      <button className="link-btn" onClick={check} disabled={state.loading}>
        {state.loading ? "Checking…" : "Recheck"}
      </button>
    </div>
  );
}
