from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

from app.core.config import settings

# --- Postgres (SQLAlchemy) ---
engine = create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a Postgres session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Qdrant (vector DB) ---
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


# --- Neo4j (graph DB) ---
_neo4j_driver = None


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _neo4j_driver
