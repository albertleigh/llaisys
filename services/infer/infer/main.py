"""FastAPI application entry-point with lifespan model loading."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import Settings, settings
from .engine import InferenceEngine
from .api.chat import router as chat_router

# Portal UI build output (ui/portal/dist/)
_PORTAL_DIST = Path(__file__).resolve().parents[3] / "ui" / "portal" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    cfg: Settings = app.state.settings
    model_path = Path(cfg.model_path)
    if not model_path.is_absolute():
        # Try resolving from CWD first (handles user-provided relative paths like ../../models/...)
        cwd_resolved = Path.cwd() / model_path
        if cwd_resolved.exists():
            model_path = cwd_resolved.resolve()
        else:
            # Fall back to project root (for default config value like "models/...")
            model_path = (Path(__file__).resolve().parents[3] / model_path).resolve()

    print(f"[infer] Loading model from {model_path}  (device={cfg.device})")
    engine = InferenceEngine(
        model_path=model_path,
        device=cfg.device,
        max_ctx_len=cfg.max_ctx_len,
    )
    app.state.engine = engine
    print(f"[infer] Model ready — serving at http://{cfg.host}:{cfg.port}")
    yield
    # Cleanup: Python GC will call model.__del__
    print("[infer] Shutting down.")


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or settings

    app = FastAPI(
        title="llaisys-infer",
        version="0.1.0",
        description="OpenAI-compatible chat-completion API backed by llaisys",
        lifespan=lifespan,
    )
    app.state.settings = cfg

    # CORS — allow the portal UI (and any other origin during dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ── Serve portal UI static files (production build) ──────────────
    if _PORTAL_DIST.is_dir():
        # Serve index.html for the root and any non-API paths (SPA fallback)
        @app.get("/")
        async def portal_index():
            return FileResponse(_PORTAL_DIST / "index.html")

        # Mount static assets (JS, CSS, etc.) — must come after API routes
        app.mount("/assets", StaticFiles(directory=_PORTAL_DIST / "assets"), name="portal-assets")

        # SPA catch-all: serve index.html for any unmatched path so
        # client-side routing works (must be the last route)
        @app.get("/{path:path}")
        async def portal_spa_fallback(path: str):
            # If a real file exists in dist/, serve it (e.g. favicon, vite.svg)
            file = _PORTAL_DIST / path
            if file.is_file():
                return FileResponse(file)
            return FileResponse(_PORTAL_DIST / "index.html")

    return app


# Default app instance for `uvicorn infer.main:app`
app = create_app()


def cli():
    """CLI entry-point: ``llaisys-infer``."""
    parser = argparse.ArgumentParser(description="llaisys inference server")
    parser.add_argument(
        "--model", default=settings.model_path,
        help="Path to model directory",
    )
    parser.add_argument(
        "--device", default=settings.device, choices=["cpu", "nvidia"],
        help="Inference device",
    )
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--max-ctx-len", type=int, default=settings.max_ctx_len)
    args = parser.parse_args()

    cfg = Settings(
        model_path=args.model,
        device=args.device,
        host=args.host,
        port=args.port,
        max_ctx_len=args.max_ctx_len,
    )
    application = create_app(cfg)
    uvicorn.run(application, host=cfg.host, port=cfg.port)
