"""
Main benchmark runner.

Executes every (backend × dataset × n_rows × seed) combination, evaluates each,
and writes a tidy CSV to study/results/benchmark.csv.

Usage:
    python -m study.run_benchmark
    python -m study.run_benchmark --quick    # smaller grid for fast iteration
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.seed_data import water_borne, vector_borne, skin_ntd  # noqa: E402
from study.backends import all_backends  # noqa: E402
from study.evaluation import evaluate  # noqa: E402


DATASETS = {
    "water_borne": (water_borne, "disease_cases"),
    "vector_borne": (vector_borne, "disease_cases"),
    "skin_ntd": (skin_ntd, "disease_cases"),
}


def run_one(
    backend_factory,
    dataset_name: str,
    n_rows: int,
    seed: int,
    target: str,
    real: pd.DataFrame,
    tmp_path: Path,
) -> dict:
    """Run a single benchmark cell and return a flat result row."""
    backend = backend_factory()
    row = {
        "backend": backend.name,
        "dataset": dataset_name,
        "n_rows": n_rows,
        "seed": seed,
    }
    try:
        backend.fit(real)
        synth = backend.sample(len(real))
        metrics = evaluate(real, synth, target=target)
        try:
            disk = backend.disk_bytes(tmp_path)
        except Exception:
            disk = float("nan")
        row.update(metrics)
        row["fit_seconds"] = backend.fit_seconds
        row["sample_seconds"] = backend.sample_seconds
        row["disk_bytes"] = disk
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Smaller grid, ~3 min")
    parser.add_argument("--out", default=str(ROOT / "study" / "results" / "benchmark.csv"))
    args = parser.parse_args()

    if args.quick:
        n_rows_grid = [300]
        seeds = [42]
    else:
        n_rows_grid = [300, 1000]
        seeds = [42, 7, 123]

    # Build the factory list — fresh instance per cell so state doesn't leak.
    factories = [
        # We rebuild per cell to capture fit/sample timings cleanly.
        # Each lambda returns a NEW instance each call.
    ]
    from study.backends import OursBackend, SDVCopulaBackend, CTGANBackend, TVAEBackend
    factories = [
        lambda: OursBackend(),
        lambda: SDVCopulaBackend(),
        lambda: CTGANBackend(epochs=100),
        lambda: TVAEBackend(epochs=100),
    ]

    rows = []
    total_cells = len(DATASETS) * len(n_rows_grid) * len(seeds) * len(factories)
    cell_idx = 0
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for dataset_name, (gen_fn, target) in DATASETS.items():
            for n_rows in n_rows_grid:
                for seed in seeds:
                    real = gen_fn(n=n_rows, seed=seed)
                    for factory in factories:
                        cell_idx += 1
                        # Peek at the backend name without consuming it.
                        peek = factory()
                        bname = peek.name
                        del peek
                        elapsed = time.perf_counter() - t_start
                        print(
                            f"[{cell_idx}/{total_cells}] {bname} | {dataset_name} | "
                            f"n={n_rows} seed={seed} | elapsed={elapsed:.0f}s",
                            flush=True,
                        )
                        row = run_one(
                            factory, dataset_name, n_rows, seed, target, real, tmp_path
                        )
                        rows.append(row)
                        # Incremental save in case something dies mid-run.
                        df = pd.DataFrame(rows)
                        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n=== Benchmark complete: {len(df)} rows ===")
    print(f"Total time: {time.perf_counter() - t_start:.1f}s")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
