"""
FastAPI application entrypoint. Loads environment config, ensures the
Qdrant collection exists, and mounts the API router.
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402  (must follow load_dotenv)

from api.routes import router  # noqa: E402
from ingestion.pipeline import ensure_collection  # noqa: E402

app = FastAPI(title="RAG Chat System")


@app.on_event("startup")
async def startup() -> None:
    """Create the Qdrant collection on startup if it does not already exist."""
    ensure_collection()


app.include_router(router)
