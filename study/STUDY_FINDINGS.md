# Benchmark Study: Generative Model Choice for NTD Surveillance Synthesis

## What this study tests

For NTD-scale tabular datasets (300–1000 rows, 6–10 columns) typical of Sub-Saharan surveillance, does a deep tabular generative model outperform a classical Gaussian Copula on fidelity and downstream utility?

## Setup

| | |
|---|---|
| Datasets | water_borne, vector_borne, skin_ntd (the three deployed modules) |
| Sizes | n = 300, 1000 |
| Seeds | 42, 7, 123 |
| Backends | Ours (custom Gaussian Copula), SDV Gaussian Copula, CTGAN (100 epochs), TVAE (100 epochs) |
| Total cells | 4 × 3 × 2 × 3 = **72** |
| Hardware | CPU only (Vercel-equivalent; no GPU) |
| Total wall time | ~4 minutes |

## Headline result

Mean across all 18 evaluation cells per backend:

| Backend | KS↓ | W₁↓ | TVD↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ | Fit (s)↓ | Disk (KB)↓ |
|---|---|---|---|---|---|---|---|---|
| **Ours (Gaussian Copula)** | **0.034** | **0.050** | **0.048** | **0.036** | **0.966** | **1.052** | **0.00** | **36** |
| SDV Gaussian Copula | 0.088 | 0.127 | 0.053 | 0.042 | 0.487 | 0.625 | 0.36 | 88 |
| CTGAN (100 ep) | 0.301 | 0.695 | 0.079 | 0.151 | -0.691 | 0.245 | 6.75 | 909 |
| TVAE (100 ep) | 0.324 | 0.505 | 0.672 | 0.117 | 0.063 | 0.546 | 2.26 | 338 |

Across every fidelity metric and both utility tasks, the simpler model wins. CTGAN's R² ratio is negative, which means a regressor trained on its synthetic data performs *worse than predicting the mean* when evaluated on real test data. This is a well-known failure mode of GAN-based tabular generators on small datasets.

## Effect of training-set size (Figure 3)

| Backend | n=300 R² ratio | n=1000 R² ratio | improvement |
|---|---|---|---|
| Ours (Gaussian Copula) | 0.962 | 0.971 | flat (already saturated) |
| SDV Gaussian Copula | 0.280 | 0.694 | substantial |
| CTGAN | -0.980 | -0.403 | improving but still negative |
| TVAE | -0.148 | 0.274 | improving but still poor |

The deep models do improve with more data, but at n=1000 they are still substantially worse than our copula at n=300. Extrapolation suggests CTGAN/TVAE would need roughly an order of magnitude more rows to match the copula. NTD surveillance datasets simply don't have that volume.

## Why the deep models lose

This is not a controversial finding once examined carefully. Tabular GANs face three structural problems on small data:

1. **Mode collapse on rare categories.** With 6–8 regions and a few hundred rows per dataset, some region × land-use combinations have only 10–20 examples. CTGAN's discriminator overfits these quickly, and the generator collapses onto the dominant modes. This shows up in our Wasserstein and TVD metrics.
2. **Spurious correlations.** Deep models capture pairwise correlations indirectly through their hidden layers. With limited samples, they pick up sample-noise correlations and reproduce them as if they were real structure. This shows up in our correlation distance metric.
3. **TVAE's variance overshoot.** Inspection of Figure 4 shows TVAE consistently produces narrower-than-real distributions on continuous columns. This is a known consequence of the KL-divergence term in the VAE objective for low-data regimes.

The Gaussian Copula sidesteps all three: it parameterises marginals non-parametrically (KDE), captures dependencies through a single covariance matrix on PIT-transformed data, and has no neural network capable of overfitting noise.

## What this means for the paper

The contribution is **not** the copula method. It is:

> *A systematic empirical demonstration that, for the data regime characteristic of NTD surveillance in Sub-Saharan Africa (hundreds to low thousands of rows, mixed-type schemas, strong covariate-driven case counts), a properly-implemented Gaussian Copula beats deep tabular generators on both fidelity and downstream utility, at roughly 1000× lower training cost and 25× lower disk footprint.*

The implication for practitioners is concrete: do not reach for CTGAN/TVAE on NTD-scale data. Use a copula. We provide a deployable open-source system that does this.

## Pre-empting reviewer attacks

| Reviewer concern | Response |
|---|---|
| "100 epochs is too few for CTGAN, train longer." | Defensible, but training longer increases the cost gap, not closes it. We can run a 500-epoch sensitivity analysis as a supplementary experiment if requested. The data-scarcity claim is the key: at this dataset size, more epochs amplifies overfitting. |
| "The seed datasets are synthetic, you've designed a benchmark that favours copulas." | Important caveat to disclose in the paper. The seed generators use **independent covariate sampling and a single Poisson link function**; they are *not* designed to embed copula-friendly structure. The covariate generation includes mixed types, regional categorical effects, and non-linear case-count dependencies (multiplicative interactions in the Poisson rate). Still, validating on real NTD surveillance data is the natural follow-up and we will state this as a limitation. |
| "TSTR R² < 0 means the model failed; you can't average failed cells." | Fair. We report the median in supplementary; medians and means tell the same story. We also report the binary "did the synthetic data produce a usable downstream model" rate, where a model is "usable" if R² ratio > 0.5: **Ours = 18/18, SDV = 9/18, CTGAN = 0/18, TVAE = 4/18**. For the outbreak F1 task: Ours = 18/18, SDV = 12/18, CTGAN = 4/18, TVAE = 7/18. |
| "Why not include Bayesian Networks / TabDDPM / Synthetic Minority Oversampling?" | Scope. We chose the two most-cited deep tabular generators (CTGAN, TVAE) and the canonical statistical baseline (SDV Copula). Adding more methods is in future work. |
| "Ghana/Nigeria/Kenya synthetic data is not 'Sub-Saharan Africa'." | Agreed. The system is designed to ingest real surveillance data; the seed data is for development and benchmarking. Real-data validation requires data-sharing agreements with ministries of health that take 6–18 months. We frame this as Phase 1 of a programme. |

## Reproducibility

```bash
cd ntd-synthgen
pip install -r study/requirements-study.txt
python -m study.run_benchmark            # ~4 min
python -m study.aggregate                # tables
python -m study.make_figures             # figures
```

All randomness is seeded; results in `study/results/benchmark.csv` are deterministic up to the OS-level scheduling that affects fit-time measurements.
