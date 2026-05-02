"""Local development runner.

    python run.py

Serves the FastAPI app at http://127.0.0.1:8000 and the static frontend
at http://127.0.0.1:8000/ui/ via a simple mount.
"""
from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

from api.main import app

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
