"""End-to-end API tests with FastAPI's TestClient."""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Redirect storage roots to a temp dir BEFORE importing the app.
    tmp = tmp_path_factory.mktemp("ntdtest")
    os.environ["NTD_DATA_ROOT"] = str(tmp / "data")
    os.environ["NTD_MODEL_ROOT"] = str(tmp / "models")
    # Ensure the bundled samples exist (regenerate just in case).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.seed_data import generate_all
    generate_all(Path(__file__).resolve().parent.parent / "data" / "samples")

    # Import after env is set.
    from api.main import app
    return TestClient(app)


def test_root_health(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["modules_loaded"] >= 3


def test_list_modules(client):
    res = client.get("/modules")
    assert res.status_code == 200
    names = {m["name"] for m in res.json()}
    assert {"water_borne", "vector_borne", "skin_ntd"} <= names


def test_get_module_detail(client):
    res = client.get("/modules/water_borne")
    assert res.status_code == 200
    body = res.json()
    assert body["target"] == "disease_cases"
    assert "region" in body["categorical_columns"]


def test_train_with_seed_then_generate_then_metrics(client):
    # Train
    train_res = client.post("/train-model", json={"module": "water_borne"})
    assert train_res.status_code == 200, train_res.text
    model_id = train_res.json()["model_id"]

    # Generate
    gen_res = client.post("/generate-data", json={"model_id": model_id, "n_rows": 40})
    assert gen_res.status_code == 200, gen_res.text
    body = gen_res.json()
    assert body["rows"] == 40
    ds_id = body["dataset_id"]

    # Download
    dl_res = client.get(f"/download-data/{ds_id}")
    assert dl_res.status_code == 200
    assert "text/csv" in dl_res.headers["content-type"]

    # Metrics
    m_res = client.get(f"/metrics/{model_id}?n_synth=200")
    assert m_res.status_code == 200, m_res.text
    rep = m_res.json()["report"]
    assert "tstr" in rep and "ks" in rep


def test_upload_invalid_extension_rejected(client):
    files = {"file": ("evil.exe", b"\x00\x01", "application/octet-stream")}
    res = client.post("/upload-data", files=files)
    assert res.status_code == 400


def test_upload_csv_then_train(client):
    df = pd.DataFrame({
        "region": ["Lagos", "Accra"] * 50,
        "rainfall_mm": list(range(100)),
        "temperature_c": [25 + (i % 5) for i in range(100)],
        "population_density": [200 + i for i in range(100)],
        "sanitation_index": [(i % 10) / 10 for i in range(100)],
        "water_source": ["piped", "well"] * 50,
        "open_defecation_rate": [(i % 10) / 20 for i in range(100)],
        "disease_cases": [i % 30 for i in range(100)],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("test.csv", buf.getvalue().encode(), "text/csv")}
    up = client.post("/upload-data", files=files)
    assert up.status_code == 200, up.text
    ds = up.json()["dataset_id"]

    train = client.post("/train-model", json={"module": "water_borne", "dataset": ds})
    assert train.status_code == 200, train.text


def test_generate_unknown_model(client):
    res = client.post("/generate-data", json={"model_id": "nonexistent", "n_rows": 10})
    assert res.status_code == 404
