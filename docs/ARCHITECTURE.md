# Architecture

## Overview

NTD-SynthGen is a four-layer system:

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Bootstrap 5 + custom editorial CSS)          │
│  index → upload → train → generate → results → download │
└──────────────────────────┬──────────────────────────────┘
                           │ fetch JSON
┌──────────────────────────▼──────────────────────────────┐
│  API layer (FastAPI)                                    │
│  /modules, /upload-data, /train-model, /generate-data,  │
│  /download-data, /metrics                               │
└──────────────────────────┬──────────────────────────────┘
                           │ Python calls
┌──────────────────────────▼──────────────────────────────┐
│  Core engine                                            │
│  synthesizer · validator · registry · preprocessing     │
└──────────────────────────┬──────────────────────────────┘
                           │ filesystem
┌──────────────────────────▼──────────────────────────────┐
│  Storage                                                │
│  modules/ (read-only) · data/ · models/                 │
└─────────────────────────────────────────────────────────┘
```

## Synthesizer pipeline

Training (`GaussianCopulaSynthesizer.fit`):

1. Schema-aware split. Columns are partitioned into continuous and categorical based on dtype and module config.
2. Marginal estimation. Continuous columns get a Gaussian KDE marginal; categorical columns get an empirical category-frequency table.
3. Probability Integral Transform. Every value is mapped to a uniform via its column's marginal CDF. For categorical columns, an ordering is assigned and category midpoints are used.
4. Inverse normal CDF. Uniforms become standard normals.
5. Covariance estimation. Sample covariance on the standard-normal-space matrix, with a small ridge added so the matrix is positive definite.

Sampling (`GaussianCopulaSynthesizer.sample`):

1. Draw from MVN with the fitted covariance.
2. Apply the standard normal CDF to get uniforms.
3. Apply each column's inverse marginal CDF to recover original-scale values.
4. Apply post-sampling clipping rules from the module config.
5. For conditional sampling, oversample then filter (rejection sampling).

## Module discovery

`core/registry.py` walks `modules/` at startup. Any subdirectory containing `schema.json` is registered as a `DiseaseModule`. The optional `config.py` is dynamically imported via `importlib.util.spec_from_file_location` and its `MODULE_CONFIG` dict is attached to the module.

Adding a new disease family is purely additive. Drop a folder in, restart the server, and it shows up in `/api/modules`.

## Validator pipeline

`core/validator.full_report(real_df, synthetic_df, target=None)` returns a dict with these keys:

- `ks`: `{column: {statistic, p_value}}` for each numeric column.
- `wasserstein`: `{column: distance}` for each numeric column.
- `tvd`: `{column: distance}` for each categorical column.
- `correlation_distance`: Frobenius norm of the difference between real and synthetic Pearson correlation matrices.
- `summary`: per-column means and stds, side by side.
- `tstr`: `{r2_real, r2_synthetic, ratio}` if a target is supplied. The model is `RandomForestRegressor(n_estimators=100, random_state=42)`.

## Persistence

Trained models are saved as `.npz` archives containing the covariance matrix, marginal parameters, column names, and dtypes. Sidecar `.json` files store the original training data path and module name so that `/api/metrics` can re-load the real reference data later.

Generated datasets are saved as CSVs keyed by a UUID and exposed via `/api/download-data/{id}`.

## Environment overrides

The `NTD_DATA_ROOT` and `NTD_MODEL_ROOT` environment variables override the default `data/` and `models/` paths. This is what `api/index.py` uses to redirect writes to `/tmp` on Vercel.
