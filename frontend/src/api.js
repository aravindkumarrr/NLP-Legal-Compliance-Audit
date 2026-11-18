// Thin client over the FastAPI backend (backend/app/api/documents.py).
const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error("Can't reach the API. Check that the backend is running on port 8000.");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

const fileForm = (file) => {
  const fd = new FormData();
  fd.append("file", file);
  return { method: "POST", body: fd };
};

export const api = {
  health: () => request("/health"),
  previewClauses: (file) => request("/documents/preview-clauses", fileForm(file)),
  upload: (file) => request("/documents/upload", fileForm(file)),
  index: (id) => request(`/documents/${id}/index`, { method: "POST" }),
  classify: (id) => request(`/documents/${id}/classify`, { method: "POST" }),
  graph: (id) => request(`/documents/${id}/graph`, { method: "POST" }),
  audit: (id) => request(`/documents/${id}/audit`, { method: "POST" }),
  report: (id) => request(`/documents/${id}/report`),
  runPipeline: (file) => request("/documents/run-pipeline", fileForm(file)),
};

// The backend has no "list documents" endpoint, so recent runs are remembered
// in this browser only.
const KEY = "lca.recent";
export const recent = {
  list() {
    try {
      return JSON.parse(localStorage.getItem(KEY) || "[]");
    } catch {
      return [];
    }
  },
  add(entry) {
    const next = [entry, ...recent.list().filter((e) => e.document_id !== entry.document_id)].slice(0, 12);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* storage unavailable */
    }
    return next;
  },
  remove(id) {
    const next = recent.list().filter((e) => e.document_id !== id);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
    return next;
  },
};
