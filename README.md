# NTD-SynthGen

**A modular synthetic data generation engine for Neglected Tropical Diseases (NTDs) in Sub-Saharan Africa.**

NTD-SynthGen generates synthetic tabular datasets for three NTD families: water-borne, vector-borne, and skin. Four interchangeable backends (Gaussian Copula, Bayesian Network, CTGAN, TVAE) sit behind a single `BaseSynthesizer` interface. The benchmark suite that ships with the project tells you which backend to use for your data and your deployment target.

It is built for the case that comes up over and over in African NTD research: you cannot get patient-level surveillance data because it is restricted, locked behind data-sharing agreements that take months to negotiate, or simply never digitised. Synthetic data is a workaround; this is a tool that makes that workaround usable in low-resource settings.

---

## Which backend should you use?

We benchmarked all four backends on three NTD modules across three seeds (36 trials, 600 rows each). On NTD-scale tabular data the picture looks like this:

| Backend         | TSTR R² ratio    | TSTR macro-F1 ratio | Wasserstein-1 | Fit time |
|-----------------|------------------|---------------------|---------------|----------|
| Gaussian Copula | **0.92 – 1.05**  | **1.00 – 1.05**     | **1.7 – 5.4** | 0.004 s  |
| Bayesian Net    | -0.19 – 0.17     | 0.60 – 0.75         | 2.6 – 16.2    | 0.34 s   |
| CTGAN           | -1.77 – -0.64    | 0.55 – 0.59         | 6.1 – 18.8    | 4.36 s   |
| TVAE            | -0.51 – 0.15     | 0.54 – 0.74         | 5.9 – 23.1    | 1.50 s   |

The headline: on data of this size (hundreds to a few thousand rows, 6 to 10 columns), the deep neural backends underperform the Gaussian Copula on every metric, while costing 1000× more compute and forcing an 800 MB PyTorch dependency. CTGAN's TSTR R² is *negative*. A regressor trained on its synthetic output predicts disease cases worse than predicting the mean. So the Gaussian Copula is the default. The other backends are still there if you want to verify on your own data.

Reproduce the headline benchmark with:

```bash
pip install -r requirements-bench.txt
python -m bench.run_benchmark
```

Results land in `bench/results/`: three CSV tables, a JSON dump, three publication-ready figures.

For the larger study (size effects at n=300 vs n=1000, our copula vs SDV's copula vs CTGAN vs TVAE, 72-cell sweep, four figures), see `study/STUDY_FINDINGS.md` and run:

```bash
pip install -r study/requirements-study.txt
python -m study.run_benchmark    # ~4 minutes
python -m study.aggregate
python -m study.make_figures
```

---

## Why this exists

NTD researchers in Sub-Saharan Africa run into the same wall over and over: the surveillance records exist, but the data is siloed inside ministries, locked behind ethics-board barriers, or simply not digitised. Meanwhile the methods (modelling, dashboarding, intervention-targeting) need data to develop and test against.

Synthetic data closes that gap. A model that captures the joint distribution of a real surveillance dataset can produce as many additional rows as you want, preserving the statistical structure without exposing any individual record. The synthetic copy is for prototyping, teaching, and benchmarking. Real data is for final validation, once you can get hold of it.

What the project tries to be:

- Backend-agnostic. One `BaseSynthesizer` interface, four reference backends, room to drop in your own.
- Empirically grounded. Backend selection is justified by a reproducible benchmark, not by reflex toward the most recent deep-learning paper.
- Disease-aware. Three pluggable disease modules with schemas, constraints, and seed generators that make epidemiological sense.
- Deployable anywhere. The production stack is pure Python, CPU-only, around 50 MB. It runs locally, on Vercel, or on any VPS. Heavy backends are opt-in.
- Honest. Fidelity metrics (KS, Wasserstein, TVD, correlation distance) and TSTR utility scores are computed on every model and shown to the user.

---

## Quick start

```bash
# clone / unzip the project, then:
cd ntd-synthgen
pip install -r requirements.txt
python run.py
```

Then open `http://localhost:8000/ui/` in a browser.

The API is at `http://localhost:8000/api/` and serves interactive docs at `http://localhost:8000/docs`.

---

## Architecture

```
ntd-synthgen/
├── core/                     # Pure-Python statistical engine
│   ├── base_synthesizer.py   # Abstract BaseSynthesizer interface
│   ├── synthesizer.py        # Gaussian Copula (PIT → MVN → inverse PIT)
│   ├── bayesnet_synthesizer.py  # Bayesian Network (pgmpy, optional)
│   ├── neural_synthesizer.py # CTGAN + TVAE adapters (optional, SDV)
│   ├── validator.py          # KS, Wasserstein, TVD, correlation, TSTR
│   ├── registry.py           # Disease-module discovery
│   ├── preprocessing.py      # Schema validation, type coercion
│   └── seed_data.py          # Synthetic ground-truth generators
├── modules/                  # Pluggable disease modules
│   ├── water_borne/          # Schistosomiasis, STH, dracunculiasis
│   ├── vector_borne/         # LF, onchocerciasis, leishmaniasis, HAT
│   └── skin_ntd/             # Buruli ulcer, leprosy, yaws, scabies
├── api/                      # FastAPI service
│   ├── main.py               # Endpoints
│   ├── schemas.py            # Pydantic request/response models
│   └── index.py              # Vercel ASGI entrypoint
├── frontend/                 # Bootstrap 5 + custom editorial CSS
│   ├── index.html
│   ├── pages/                # upload, train, generate, results, download
│   ├── css/styles.css
│   └── js/                   # api.js (client), layout.js (chrome)
├── bench/                    # Cross-backend benchmark suite
│   ├── run_benchmark.py      # Runs all backends × all modules × N seeds
│   ├── tasks.py              # Case regression + outbreak classification
│   └── results/              # CSVs, JSON dump, figures (after running)
├── tests/                    # pytest, 41 tests (16 contract + 25 unit/integration)
├── data/                     # uploads/ and generated/ (created at runtime)
├── models/                   # Trained synthesizer artifacts
├── requirements.txt          # Production deps (~50 MB, no PyTorch)
├── requirements-bench.txt    # Optional benchmark deps (CTGAN/TVAE/pgmpy)
├── run.py                    # Local dev runner
└── vercel.json               # Vercel deployment config
```

### Why Gaussian Copula is the default backend

Two reasons, in this order:

1. **It wins the empirical benchmark.** On NTD-scale data, it beats CTGAN, TVAE, and a Bayesian Network on every fidelity metric and on both downstream utility tasks. See the table at the top of this README, or run `python -m bench.run_benchmark` to reproduce.

2. **It deploys anywhere.** The neural backends require PyTorch (~800 MB), which exceeds Vercel's 250 MB serverless lambda limit. The Bayesian Network requires pgmpy (~100 MB). The Gaussian Copula needs only NumPy, SciPy, and pandas, all of which are already in the FastAPI dependency closure.

The Gaussian Copula approach used here:

1. Estimates a marginal distribution per column (KDE for continuous, empirical CDF for categorical).
2. Maps each column to a uniform via its empirical CDF (the **probability integral transform**, or PIT).
3. Maps uniforms to standard normals via the inverse normal CDF.
4. Estimates a covariance matrix on the standard-normal-space data.
5. Samples by drawing from the multivariate normal, then inverting steps 3 and 2.

This preserves both marginals and pairwise dependencies. On the test data it produces TSTR ratios at or above 1.0 for both regression and outbreak classification, which means a downstream model trained on synthetic data is *as useful as or more useful than* one trained on the real data of the same size. The likely reason is that the copula smooths out small-sample noise. The synthesizer interface (`fit`, `sample`, `save`, `load`) is shared by all four backends, so any of them can be swapped in with a one-line change.

### Disease modules

Each module is a folder with three files:

- **`schema.json`**: column names, types, allowed categories, prediction target.
- **`config.py`**: domain constraints (for instance `rainfall_mm ∈ [0, 4000]`, `sanitation_index ∈ [0, 1]`) and post-sampling clipping rules.
- **`description.md`**: human-readable summary of the module's epidemiological scope.

The `core/registry.py` module auto-discovers any folder under `modules/` and exposes it through `/api/modules`. Adding a fourth disease family (say, `helminth_coinfection`) is a matter of creating a new folder. No code changes elsewhere.

The seed data generators in `core/seed_data.py` produce epidemiologically plausible ground truth using Sub-Saharan regional names (Ghana, Nigeria, Kenya), Poisson case counts driven by covariates, and realistic ranges for environmental variables. These are useful both for testing and for users who want to explore the system before uploading their own data.

---

## API reference

| Method | Path                          | Purpose                                                            |
|--------|-------------------------------|--------------------------------------------------------------------|
| GET    | `/api/`                       | Health check                                                       |
| GET    | `/api/modules`                | List all disease modules                                           |
| GET    | `/api/modules/{name}`         | Module detail (schema, constraints, description)                   |
| GET    | `/api/samples/{module}`       | Download a 300-row seed dataset for the module                     |
| POST   | `/api/upload-data`            | Upload a CSV (≤5 MB) and validate against a module schema          |
| POST   | `/api/train-model`            | Fit a Gaussian Copula synthesizer on uploaded or seed data         |
| POST   | `/api/generate-data`          | Sample N rows from a trained model, optionally conditional         |
| GET    | `/api/download-data/{id}`     | Download a generated CSV                                           |
| GET    | `/api/metrics/{model_id}`     | Compute KS, Wasserstein, TVD, correlation distance, and TSTR       |

Full interactive docs at `/docs` when the server is running.

---

## Running the tests

```bash
pip install pytest httpx
pytest tests/ -v
```

Expected: **41 passed** (25 unit/integration tests + 16 backend-contract tests). The contract tests automatically skip neural backends if SDV isn't installed.

The test suite covers:

- Preprocessing pipeline (column normalisation, type coercion, schema validation).
- Synthesizer correctness for every backend (shape, categorical preservation, conditional sampling, save/load round-trip, error cases).
- Validator metrics (KS, Wasserstein, TVD, correlation distance, summary stats, TSTR utility, full report aggregation).
- Backend-contract conformance: every concrete synthesizer implements the same `BaseSynthesizer` surface.
- End-to-end API flow (train → generate → download → metrics) using FastAPI's `TestClient` against temporary directories.

---

## Deployment to Vercel

The repository ships with a `vercel.json` that routes `/api/*` to `api/index.py` (an ASGI adapter for FastAPI) and serves the static frontend.

```bash
npm i -g vercel
vercel deploy
```

The `api/index.py` entrypoint sets `NTD_DATA_ROOT` and `NTD_MODEL_ROOT` to `/tmp/...` so all writes land in Vercel's writable ephemeral storage. For persistent storage, replace those with an S3 or Supabase-backed adapter.

---

## Fidelity metrics: what the dashboard shows

After a model is trained and a synthetic dataset is generated, the **Results** page reports:

- **Per-column Kolmogorov–Smirnov D and p-value** for every numeric column. Lower D and higher p means the synthetic marginal is statistically indistinguishable from the real one.
- **Wasserstein-1 distance** for each numeric column. A more interpretable distance, in the column's own units.
- **Total Variation Distance** for each categorical column. Bounded in [0, 1], where 0 means identical category proportions.
- **Correlation-matrix Frobenius distance** between real and synthetic Pearson correlation matrices. Captures whether pairwise structure is preserved.
- **TSTR R² ratio (Train-on-Synthetic, Test-on-Real).** The strictest utility check. A predictive model is trained on the synthetic data and evaluated on real held-out data. The score is reported as a fraction of the score you get when training on real data. A ratio above 0.8 means the synthetic data is substitutable for downstream modelling.

A representative run on the water-borne seed dataset (500 rows, seed 42): **TSTR R² ratio = 0.98** (synthetic R² = 0.728 vs real-on-real baseline 0.740), **KS p-value for `rainfall_mm` = 0.99** (D = 0.028), **Wasserstein for `sanitation_index` = 0.008**, **normalised correlation distance = 0.035**, **TVD for `region` = 0.10**, **TVD for `water_source` = 0.04**.

---

## Limitations

A few things worth being upfront about.

The Gaussian Copula assumes linear dependence after the marginals are transformed into standard-normal space. Strongly non-linear interactions (threshold effects, interaction-only covariates) are partially captured but not perfectly. If TSTR drops below 0.7 on your data, that is a signal to try the CTGAN backend instead.

Conditional sampling uses rejection. For rare categorical conditions, the realised sample size shrinks. The system warns you when fewer rows are returned than requested.

Synthetic data is not a substitute for real data in confirmatory analysis. It is a tool for prototyping, teaching, and benchmarking. Any clinical or policy claim still needs to be validated on the real dataset once you can get hold of it.

The seed generators are not real surveillance data. They are statistically plausible fixtures meant to make the system testable and explorable before you upload your own data. Use them as fixtures, not as evidence.

---

## License

MIT.

---

## Citation

If this engine contributes to published research, please cite the accompanying paper (in preparation).
