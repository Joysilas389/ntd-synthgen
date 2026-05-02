"""Vercel serverless entrypoint.

Vercel discovers Python files under /api and treats them as serverless
functions. We re-export the FastAPI app from `api.main` so a single
function handles all routes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make project root importable so `from api.main import app` works in
# Vercel's ephemeral container.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Vercel's filesystem is read-only except for /tmp; route writes there.
os.environ.setdefault("NTD_DATA_ROOT", "/tmp/ntd-data")
os.environ.setdefault("NTD_MODEL_ROOT", "/tmp/ntd-models")

from api.main import app  # noqa: E402

# Vercel's Python runtime invokes a callable named `app` (ASGI) or `handler`.
handler = app
