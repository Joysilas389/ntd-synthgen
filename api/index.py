"""Vercel serverless entrypoint.

Vercel discovers Python files under /api and treats them as serverless
functions. Vercel routes /api/* requests to this file, but the URL
path it forwards still includes the /api prefix.

We mount the FastAPI app under /api here so the routes line up with
what the frontend calls (e.g. GET /api/modules → app's /modules route).
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

from starlette.applications import Starlette  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from api.main import app as inner_app  # noqa: E402

# Wrap the FastAPI app so it's reachable at /api/* on Vercel.
# Locally (via run.py) the app is served at / directly, so this wrapper
# is only used by Vercel's serverless function.
app = Starlette(routes=[Mount("/api", app=inner_app)])

# Vercel's Python runtime invokes a callable named `app` (ASGI) or `handler`.
handler = app
