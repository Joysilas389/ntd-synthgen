"""Request and response schemas for the API layer."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    modules_loaded: int


class ModuleInfo(BaseModel):
    name: str
    title: str
    description: str
    features: list[str]
    target: str | None


class TrainRequest(BaseModel):
    module: str = Field(..., description="Module name, e.g. 'water_borne'")
    dataset: str | None = Field(
        default=None,
        description="Dataset id from /upload-data, or None to use the bundled seed.",
    )
    seed: int | None = 42


class TrainResponse(BaseModel):
    model_id: str
    module: str
    rows_trained: int
    columns: list[str]
    categorical: list[str]


class GenerateRequest(BaseModel):
    model_id: str
    n_rows: int = Field(default=200, ge=1, le=10_000)
    conditions: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: list[str]
    preview: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    report: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
