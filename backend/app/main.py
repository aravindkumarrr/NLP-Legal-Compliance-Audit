from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.db import Base, engine, get_qdrant_client, get_neo4j_driver
from app.models import db_models  # noqa: F401 -- registers Document/Clause with Base before create_all
from app.api import documents

app = FastAPI(
    title="NLP-Based Legal Compliance Auditing",
    description="Backend for clause segmentation, deontic classification, "
    "knowledge-graph construction, and compliance/contradiction checks.",
    version="0.1.0",
)

# Lets the React frontend call the API directly (e.g. a production build served
# from another origin). In dev the Vite proxy makes this unnecessary.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)


@app.on_event("startup")
def create_tables():
    """
    Simple create-all instead of Alembic migrations -- appropriate for a
    project at this stage; revisit once the schema needs versioned
    migrations across environments.
    """
    Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"status": "ok", "service": "legal-compliance-auditing-backend"}


@app.get("/health")
def health():
    """
    Pings Postgres, Qdrant, and Neo4j individually so a broken connection
    shows up immediately instead of surfacing three pipeline stages later.
    """
    status = {}

    # Postgres
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["postgres"] = "ok"
    except Exception as e:
        status["postgres"] = f"error: {e}"

    # Qdrant
    try:
        get_qdrant_client().get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    # Neo4j
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        status["neo4j"] = "ok"
    except Exception as e:
        status["neo4j"] = f"error: {e}"

    return status
