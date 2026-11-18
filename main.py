"""
FastAPI application entrypoint. Loads environment config, ensures the
Qdrant collection exists, and mounts the API router.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402  (must follow load_dotenv)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

from api.routes import router  # noqa: E402
from ingestion.pipeline import ensure_collection  # noqa: E402

app = FastAPI(title="RAG Chat System")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, _rate_limit_exceeded_handler
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    """Create the Qdrant collection on startup if it does not already exist."""
    ensure_collection()


app.include_router(router)

# Serve React build — only if the build folder exists
# frontend/ lives one level up, alongside llm-chat-system/
BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(BUILD_DIR):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(BUILD_DIR, "assets")),
        name="assets"
    )

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))
