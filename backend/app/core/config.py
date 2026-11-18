from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# LLM PROVIDER SWITCH -- edit this one line to change which backend
# deontic classification (stage 3) and compliance judgment (stage 7b) use.
#   "ollama" -> local, free, slow (serialized requests to a local model)
#   "groq"   -> hosted, fast, needs GROQ_API_KEY set in .env
# This is a literal code constant on purpose, not left to an environment
# variable that could silently differ between machines -- change it here
# and every environment that runs this code picks it up.
# ============================================================================
ACTIVE_LLM_PROVIDER: str = "groq"  # "ollama" | "groq"


class Settings(BaseSettings):
    """
    Central config, loaded from environment variables / a .env file.
    Defaults match docker-compose.yml so `docker compose up` + default .env just works.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    postgres_user: str = "lca_user"
    postgres_password: str = "lca_password"
    postgres_db: str = "lca_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "clauses"
    eurlex_qdrant_collection: str = "eurlex57k_reference"

    # Cross-encoder re-ranker (stage 6)
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    dense_top_k: int = 10
    graph_hops: int = 1

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "lca_password"

    # Embedding model (CPU-friendly)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM backend for deontic classification (stage 3) and compliance
    # judgments (stage 7b). Defaults to the hardcoded ACTIVE_LLM_PROVIDER
    # constant above (not an env var) -- flip that constant to switch
    # backends everywhere this runs. Still overridable via LLM_PROVIDER
    # in .env for one-off local testing without touching code.
    llm_provider: str = ACTIVE_LLM_PROVIDER  # "ollama" | "groq"

    # Ollama (local LLM)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Groq (hosted LLM, OpenAI-compatible API). The key is intentionally
    # NOT a hardcoded constant like the provider switch above -- it's a
    # credential, not a mode selector, and belongs in .env (gitignored),
    # never committed to source. See .env.example.
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"  # fast + cheap alt: "openai/gpt-oss-20b"

    # How many clauses to classify/assess concurrently. Keep at 1 for
    # Ollama -- a local single-instance server serializes requests anyway,
    # so concurrency just adds overhead. Raise for Groq, which is a hosted
    # API built for concurrent traffic (stay under your account's rate limit).
    llm_max_workers: int = 3

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
