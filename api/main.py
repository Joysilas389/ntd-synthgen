"""FastAPI application for the NTD Synthetic Data Engine.

Endpoints
---------

  GET  /                     — health probe
  GET  /modules              — list available disease modules
  GET  /modules/{name}       — module detail (schema + description)
  POST /upload-data          — upload a CSV; returns a dataset id
  POST /train-model          — fit a synthesizer on uploaded or seed data
  POST /generate-data        — sample synthetic rows from a fitted model
  GET  /download-data/{id}   — download a generated CSV
  GET  /metrics/{model_id}   — full validation report (real vs synthetic)
  GET  /samples/{module}     — fetch a few real seed rows for the UI

Design notes
------------
The app uses a simple file-backed state on disk:

    data/uploads/        uploaded CSVs (by dataset id)
    data/generated/      generated CSVs (by dataset id)
    models/              pickled fitted synthesizers (by model id)

This keeps the API stateless across requests within one filesystem, which
is fine for local dev and persistent deployments. For Vercel's
read-only/ephemeral filesystem, see vercel.json — we route to /tmp.
"""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from core.logging_config import get_logger
from core.preprocessing import prepare
from core.registry import ModuleRegistry
from core.synthesizer import GaussianCopulaSynthesizer
from core.validator import full_report

from .schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    ModuleInfo,
    TrainRequest,
    TrainResponse,
)

log = get_logger("api")

VERSION = "1.0.0"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB hard cap


# ---------------------------------------------------------------------------
# Storage paths (Vercel serverless writes are only allowed under /tmp)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow override via env var so Vercel can point us at /tmp.
DATA_ROOT = Path(os.environ.get("NTD_DATA_ROOT", PROJECT_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("NTD_MODEL_ROOT", PROJECT_ROOT / "models"))

UPLOAD_DIR = DATA_ROOT / "uploads"
GENERATED_DIR = DATA_ROOT / "generated"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"  # bundled, read-only
MODULES_DIR = PROJECT_ROOT / "modules"

for d in (UPLOAD_DIR, GENERATED_DIR, MODEL_ROOT):
    d.mkdir(parents=True, exist_ok=True)

registry = ModuleRegistry(MODULES_DIR)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NTD Synthetic Data Engine",
    version=VERSION,
    description=(
        "Modular synthetic data generation for Neglected Tropical Diseases "
        "in low-resource settings. Designed for Sub-Saharan Africa public "
        "health informatics."
    ),
)

# CORS — frontend lives at the same Vercel deployment in production but we
# enable wide CORS for local development convenience.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _read_csv_safely(content: bytes, name: str) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File '{name}' exceeds 5 MB limit.")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")
    if df.empty:
        raise HTTPException(400, "Uploaded CSV is empty.")
    return df


def _resolve_dataset_path(dataset_id: str | None, module_name: str) -> Path:
    """Return either the user-uploaded CSV or the bundled seed for the module."""
    if dataset_id:
        candidate = UPLOAD_DIR / f"{dataset_id}.csv"
        if not candidate.exists():
            raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
        return candidate
    seed = SAMPLES_DIR / f"{module_name}_seed.csv"
    if not seed.exists():
        raise HTTPException(404, f"No seed data bundled for module '{module_name}'.")
    return seed


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=VERSION,
        modules_loaded=len(registry.discover()),
    )


@app.get("/modules", response_model=list[ModuleInfo])
def list_modules() -> list[ModuleInfo]:
    return [ModuleInfo(**s) for s in registry.list_summaries()]


@app.get("/modules/{name}")
def get_module(name: str) -> dict[str, Any]:
    try:
        m = registry.get(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return {
        "name": m.name,
        "title": m.title,
        "description": m.description,
        "schema": m.schema,
        "categorical_columns": m.categorical_columns,
        "feature_columns": m.feature_columns,
        "target": m.target_column,
    }


@app.get("/samples/{module}")
def get_sample(module: str, limit: int = 20) -> dict[str, Any]:
    """Return the first `limit` rows of the bundled seed dataset."""
    path = SAMPLES_DIR / f"{module}_seed.csv"
    if not path.exists():
        raise HTTPException(404, f"No seed data bundled for '{module}'.")
    df = pd.read_csv(path).head(max(1, min(limit, 100)))
    return {
        "module": module,
        "rows": len(df),
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    }


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")
    content = await file.read()
    df = _read_csv_safely(content, file.filename)

    dataset_id = _new_id("ds")
    out_path = UPLOAD_DIR / f"{dataset_id}.csv"
    df.to_csv(out_path, index=False)

    log.info("Uploaded dataset %s | shape=%s | file=%s", dataset_id, df.shape, file.filename)
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": df.shape[0],
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


@app.post("/train-model", response_model=TrainResponse)
def train_model(req: TrainRequest) -> TrainResponse:
    try:
        module = registry.get(req.module)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    csv_path = _resolve_dataset_path(req.dataset, req.module)
    df_raw = pd.read_csv(csv_path)
    df, validation = prepare(df_raw, module.schema)

    if validation["missing"]:
        log.warning("Training data missing columns: %s", validation["missing"])

    cat_cols = [c for c in module.categorical_columns if c in df.columns]
    synth = GaussianCopulaSynthesizer(seed=req.seed)
    try:
        synth.fit(df, categorical_cols=cat_cols)
    except ValueError as exc:
        raise HTTPException(400, f"Training failed: {exc}")

    model_id = _new_id("mdl")
    model_path = MODEL_ROOT / f"{model_id}.pkl"
    synth.save(model_path)

    # Persist a sidecar so we can find the training data later for metrics.
    sidecar = MODEL_ROOT / f"{model_id}.meta.csv"
    df.to_csv(sidecar, index=False)
    sidecar_module = MODEL_ROOT / f"{model_id}.module.txt"
    sidecar_module.write_text(req.module)

    log.info("Trained model %s on module %s", model_id, req.module)
    return TrainResponse(
        model_id=model_id,
        module=req.module,
        rows_trained=df.shape[0],
        columns=list(df.columns),
        categorical=cat_cols,
    )


@app.post("/generate-data", response_model=GenerateResponse)
def generate_data(req: GenerateRequest) -> GenerateResponse:
    model_path = MODEL_ROOT / f"{req.model_id}.pkl"
    if not model_path.exists():
        raise HTTPException(404, f"Model '{req.model_id}' not found. Train one first.")

    synth = GaussianCopulaSynthesizer.load(model_path)
    try:
        df = synth.sample(req.n_rows, conditions=req.conditions)
    except ValueError as exc:
        raise HTTPException(400, f"Generation failed: {exc}")

    # Apply module-specific constraints if available.
    module_file = MODEL_ROOT / f"{req.model_id}.module.txt"
    if module_file.exists():
        module_name = module_file.read_text().strip()
        try:
            module = registry.get(module_name)
            if module.constraints:
                df = module.constraints(df)
        except KeyError:
            pass

    dataset_id = _new_id("gen")
    out_path = GENERATED_DIR / f"{dataset_id}.csv"
    df.to_csv(out_path, index=False)

    return GenerateResponse(
        dataset_id=dataset_id,
        rows=df.shape[0],
        columns=list(df.columns),
        preview=df.head(10).to_dict(orient="records"),
    )


@app.get("/download-data/{dataset_id}")
def download_data(dataset_id: str) -> FileResponse:
    path = GENERATED_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    return FileResponse(
        path=path,
        filename=f"ntd_synthetic_{dataset_id}.csv",
        media_type="text/csv",
    )


@app.get("/metrics/{model_id}", response_model=MetricsResponse)
def metrics(model_id: str, n_synth: int = 500) -> MetricsResponse:
    """Run a fresh validation comparing real seed/training data vs synthetic.

    Generates `n_synth` rows from the model and compares them against the
    training data captured during /train-model.
    """
    sidecar = MODEL_ROOT / f"{model_id}.meta.csv"
    model_path = MODEL_ROOT / f"{model_id}.pkl"
    module_file = MODEL_ROOT / f"{model_id}.module.txt"
    if not (sidecar.exists() and model_path.exists()):
        raise HTTPException(404, f"Model '{model_id}' not found or missing training data.")

    real = pd.read_csv(sidecar)
    synth = GaussianCopulaSynthesizer.load(model_path).sample(n_synth)

    # Apply constraints to make comparisons epidemiologically fair.
    if module_file.exists():
        try:
            module = registry.get(module_file.read_text().strip())
            if module.constraints:
                synth = module.constraints(synth)
            target = module.target_column
        except KeyError:
            target = None
    else:
        target = None

    return MetricsResponse(report=full_report(real, synth, target_col=target))


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exc_handler(request, exc):
    # Don't shadow HTTPException's default formatting.
    if isinstance(exc, HTTPException):
        raise exc
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
