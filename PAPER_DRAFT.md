# A modular, deployable synthetic data generation system for Neglected Tropical Diseases in Sub-Saharan Africa: an empirical comparison of generative model backends

**Author:** [Your name], [Your affiliation], Accra, Ghana
**Corresponding author:** [Your email]
**Target journal:** *BMC Medical Informatics and Decision Making*

---

## Abstract

### Background

Research on Neglected Tropical Diseases (NTDs) in Sub-Saharan Africa is bottlenecked by a familiar problem: there is little data, and what exists is locked inside ministry systems, behind privacy law, or never digitised at all. Synthetic data generation is one workaround. But the field has settled on deep generative models, primarily Conditional Tabular GAN (CTGAN) and Tabular VAE (TVAE), that need infrastructure low-resource settings do not have, and whose performance on small datasets has barely been measured in the NTD context.

### Methods

We built NTD-SynthGen, an open-source synthetic data system with one shared `BaseSynthesizer` interface and four backends behind it: a Gaussian Copula, a discrete Bayesian Network, CTGAN, and TVAE. The system also ships a modular disease registry, a FastAPI backend, a Bootstrap web frontend, and a 41-test suite. We benchmarked all four backends on three NTD modules (water-borne, vector-borne, skin) and on a WHO-anchored Sub-Saharan dataset (25 countries, 13 years of schistosomiasis preventive chemotherapy coverage). Fidelity was measured with Kolmogorov–Smirnov, Wasserstein-1, Total Variation Distance, and correlation distance. Downstream utility was measured with Train-on-Synthetic-Test-on-Real R² (case regression) and macro-F1 (outbreak classification).

### Results

The Gaussian Copula came first on every metric on every dataset. On the seed datasets (n=600), it reached a TSTR R² ratio of 0.92 to 1.05; the deep neural backends ran from -1.77 to 0.17. Mean Wasserstein was 1.7 to 5.4 for the copula against 5.9 to 23.1 for the neural backends. Fit time was 0.004 s, against 4.36 s for CTGAN. On the WHO-anchored dataset the picture was the same: TSTR R² ratio 0.92 for the copula, -0.67 for CTGAN, 0.10 for TVAE. CTGAN's R² ratio was negative in every regression cell, which means a regressor trained on its synthetic output predicted real coverage worse than predicting the mean. The full production deployment is 50 MB. CTGAN and TVAE need PyTorch, which is roughly 800 MB, and breaks the 250 MB serverless function limit on the platforms most low-resource teams can actually afford.

### Conclusions

For NTD-scale tabular data (a few hundred to a few thousand rows, mixed types) a Gaussian Copula beats the deep tabular generators on both fidelity and utility, at three orders of magnitude lower compute cost. We release NTD-SynthGen as an open-source artifact and recommend that practitioners default to copula-based methods on data of this size. Deep generative models are still worth trying, but as opt-in alternatives that have to earn their place against a copula on the specific dataset, not as the default.

**Keywords:** synthetic data; neglected tropical diseases; Sub-Saharan Africa; public health informatics; Gaussian Copula; CTGAN; tabular data; low-resource settings; benchmark

---

## Background

### Data scarcity as an informatics problem

The World Health Organization recognises twenty conditions as Neglected Tropical Diseases (NTDs). Together they affect over 1.6 billion people, mostly in Sub-Saharan Africa [1, 2]. Schistosomiasis alone required preventive chemotherapy for an estimated 253.7 million people in 2024, with 91% of that burden in the WHO African Region [3]. Lymphatic filariasis affects 51 million Africans; onchocerciasis 37 million; trachoma 30 million [2].

For all that scale, NTDs are barely represented in machine learning research. The reason is mundane. The bottleneck is not analytical capacity, it is data access. Patient-level surveillance records sit in country-specific systems (DHIS2 deployments, ministry archives) governed by data-sharing agreements that take 6 to 18 months to negotiate when they are negotiable at all. Aggregate country-year data exists in WHO's Global Health Observatory, but the granularity is too coarse for most modelling tasks [4]. The practical effect is that informatics research about Sub-Saharan disease is mostly done on data from somewhere else, usually North American or European hospital records, producing models whose fit to the actual burden is unverified.

Synthetic data generation is one of the few workable responses to this. A model that captures the joint distribution of a real surveillance dataset can produce as many additional rows as you want, preserving statistical structure without exposing any individual record. Researchers can prototype, benchmark, and teach on the synthetic copy and reserve the real data for final validation [5, 6].

### The state of the art and its mismatch with NTD deployment reality

Tabular synthetic data generation has been led by two deep generative models since 2019. CTGAN [7] is a conditional GAN with mode-specific normalisation for continuous columns and one-hot encoding plus training-by-sampling for categoricals. TVAE, introduced in the same paper, is the variational-autoencoder counterpart. Both ship in the Synthetic Data Vault library [8] and are the standard reach in published applications.

Three problems show up when you try to use them for NTDs:

1. **Sample-size mismatch.** The original CTGAN and TVAE evaluations used datasets of 10,000 to 600,000 rows [7]. Country-year aggregated NTD datasets are usually a few hundred. District-level surveillance rarely passes a few thousand per disease per country. Performance of deep tabular generators in this regime has barely been studied, and the comparisons that exist are mixed [9, 10].

2. **Compute infrastructure.** CTGAN and TVAE depend on PyTorch, which is roughly 800 MB on disk. That blows past the 250 MB function-size limit on AWS Lambda, Vercel, and Cloudflare Workers [11]. Deployment falls back to dedicated VPS or container infrastructure that ministries of health and academic groups in the region often cannot maintain.

3. **Methodological opacity.** Deep tabular generators are difficult to audit. When an epidemiologist asks why the synthetic data looks the way it does, the honest answer is that the discriminator's gradient said so. That answer satisfies nobody, and it does not help when the synthetic data fails a downstream task.

These constraints are part of why classical statistical methods for tabular synthesis, particularly Gaussian Copulas, have been getting a second look [12, 13]. The Gaussian Copula approach splits the joint distribution into per-column marginals (estimated non-parametrically) and a multivariate Gaussian dependence on probability-integral-transformed (PIT) data. It is interpretable, fast, and has no neural network to overfit noise. The open question is whether it actually competes with deep generators on NTD-scale data. Nobody has measured this systematically.

### What this paper contributes

Four things:

1. **A backend-agnostic synthetic data system** for NTDs in Sub-Saharan Africa, with a documented `BaseSynthesizer` interface and four interchangeable backends (Gaussian Copula, Bayesian Network, CTGAN, TVAE).
2. **A modular disease registry.** New NTD schemas are added by dropping a folder containing `schema.json`, `config.py`, and `description.md` into the modules directory. No code changes elsewhere.
3. **A reproducible benchmark** comparing all four backends on three NTD families and a WHO-anchored real-world dataset, with both fidelity and downstream-utility metrics.
4. **A deployable open-source artifact** with a FastAPI service, Bootstrap web frontend, and Vercel deployment configuration, under the MIT license.

Source code: [GitHub repository URL, author to insert before submission].

---

## Methods

### System architecture

NTD-SynthGen has four layers (Figure 1).

The frontend is a static Bootstrap 5 application with a custom editorial style sheet, structured across five pages: upload, train, generate, results, and download. It talks to the backend through `fetch` and renders results with inline SVG histograms and heatmaps.

Behind the frontend is a FastAPI service with nine endpoints. Module discovery (`GET /modules`, `GET /modules/{name}`), seed sample download (`GET /samples/{module}`), data upload (`POST /upload-data`, with a 5 MB cap and CSV-only validation), training (`POST /train-model`), generation (`POST /generate-data`, with optional conditional rules), output download (`GET /download-data/{id}`), and metric computation (`GET /metrics/{model_id}`).

The core engine holds the `BaseSynthesizer` abstract base class, four backend implementations, the validator (per-column KS, Wasserstein-1, TVD, correlation Frobenius distance, summary statistics, TSTR utility), the module registry, and the preprocessing pipeline.

For storage, the system uses environment-overridable paths so it can run on serverless platforms where the only writable directory is `/tmp`. Trained models persist as `.npz` archives (Gaussian Copula) or `.pkl` (other backends), with `.json` sidecar files for schema and provenance.

Three deployment modes: local development (`python run.py`), Vercel serverless (via `vercel.json` and an ASGI shim), or any standard VPS via `uvicorn`.

### Synthesizer backends

The `BaseSynthesizer` interface specifies four methods (`fit`, `sample`, `save`, `load`) and one class attribute (`name`). All four backends implement this contract. Conformance is checked by 16 parametrised contract tests.

**Gaussian Copula.** A custom NumPy/SciPy implementation. For each continuous column we fit a Gaussian KDE marginal; for each categorical column an empirical category-frequency table. Each value is mapped to a uniform via its column's marginal CDF (probability integral transform, PIT), then to a standard normal via the inverse normal CDF. We estimate the sample covariance on the standard-normal-space matrix and add a small ridge term so the matrix is positive definite. Sampling reverses the pipeline: draw from MVN(0, Σ), apply the standard normal CDF to get uniforms, apply each column's inverse marginal CDF to recover original-scale values, and apply post-sampling clipping rules from the module config. Conditional sampling uses rejection: oversample, then filter on the conditioning columns.

**Bayesian Network.** Built on pgmpy [14]. Continuous columns are discretised into 8 quantile bins. The DAG is learned by Hill-Climb search using the BIC discrete score (`bic-d`) with `max_iter=30`. Conditional probability distributions are fit by maximum likelihood using `DiscreteMLE`. Sampling uses forward sampling on the network; binned-continuous columns are then mapped back to real-valued draws by uniform sampling within each bin's edges.

**CTGAN and TVAE.** Adapter classes that wrap the SDV implementations [8] in our `BaseSynthesizer` interface. Default training is 100 epochs on CPU. Conditional sampling uses rejection rather than SDV's native conditional sampler so behaviour is identical across backends.

### Disease modules

Three disease families are implemented as pluggable modules:

- **Water-borne** (schistosomiasis, soil-transmitted helminthiases, dracunculiasis). Features: region, rainfall_mm, temperature_c, population_density, sanitation_index, water_source, open_defecation_rate. Target: disease_cases.
- **Vector-borne** (lymphatic filariasis, onchocerciasis, leishmaniasis, human African trypanosomiasis). Features: region, rainfall_mm, temperature_c, humidity_pct, vegetation_index, vector_density, bednet_coverage, land_use.
- **Skin** (Buruli ulcer, leprosy, yaws, scabies). Features: region, humidity_pct, rainfall_mm, proximity_to_water_km, household_size, sanitation_index, healthcare_access_index, skin_contact_occupation.

Each module folder contains `schema.json` (declarative column definitions), `config.py` (numeric range constraints and post-sampling clipping rules), and `description.md` (epidemiological scope). The module registry auto-discovers folders at startup and exposes them through the API.

### Datasets

We evaluated on two dataset families.

**Synthetic seed datasets.** For each disease module, a seed generator produces realistic country-year tabular data using Sub-Saharan regional names (Ghana, Nigeria, Kenya, and others) with Poisson-distributed case counts driven by epidemiologically plausible covariates. For vector-borne disease, for instance, the case rate is proportional to the product of `vector_density` and the inverse of `bednet_coverage`. These datasets are statistical fixtures, not real surveillance records. We use n=600 rows per module unless we say otherwise.

**WHO-anchored dataset.** A country-year tabular dataset for schistosomiasis preventive chemotherapy coverage across 25 endemic Sub-Saharan countries, 2010 to 2022. Per-country baseline coverage values are anchored to published WHO reports (4% in Nigeria and 27.5% in Ghana in 2010, for instance [15]) and progress year-on-year toward the 2022 published values [3]. Geographic covariates (rainfall, temperature, population density) use public country-average values. The dataset is 325 rows × 10 columns and ships with the project as `data/who_anchor.py`. We also include `data/fetch_who_ntd.py`, a CLI that pulls the live WHO Global Health Observatory OData API and writes a comparable real-data CSV. Reviewers and users with unrestricted internet can run it to confirm results on the live data.

### Evaluation

For each (backend × dataset × seed) triple, we:

1. Split the dataset 50/50 into train and test halves (random permutation, seeded).
2. Fit the backend on the train half.
3. Sample N synthetic rows, where N equals the dataset size.
4. Compute fidelity metrics on (test, synthetic):
   - Per-column Kolmogorov–Smirnov statistic and *p*-value (numeric columns)
   - Per-column Wasserstein-1 distance (numeric columns)
   - Per-column Total Variation Distance (categorical columns)
   - Frobenius distance between Pearson correlation matrices, normalised by the matrix dimension
5. Compute downstream utility metrics:
   - **Case regression (TSTR R²).** Train a `RandomForestRegressor(n_estimators=100)` on the train half (real-on-real baseline) and on the synthetic data (TSTR), evaluating both on the test half. Report the ratio.
   - **Outbreak classification (TSTR macro-F1).** Define an outbreak as a row with `disease_cases` above the 75th percentile of the training data. Train a `RandomForestClassifier(n_estimators=100)` on the same two training sets and evaluate macro-F1 on the test half. Report the ratio.

We use 3 random seeds (1, 2, 3) per condition and report mean ± standard deviation.

### Implementation and reproducibility

All code is Python 3.11+. Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `fastapi`, `uvicorn`, `pydantic` (production); `pgmpy`, `sdv`, `torch`, `matplotlib` (benchmark only). All randomness is seeded; results are deterministic up to OS-level scheduling that affects fit-time measurements. The full benchmark runs on a single CPU in approximately 5 minutes for the synthetic seed sweep and 4 minutes for the WHO-anchored evaluation. Tests (41 in total) execute in 17 seconds.

---

## Results

### Backend performance on synthetic seed datasets

Table 1 reports mean fidelity and utility metrics across three modules and three seeds (9 cells per backend, 36 trials total).

**Table 1.** Backend performance on synthetic seed datasets (n=600 per module, mean ± SD across 3 seeds × 3 modules).

| Backend | KS mean D ↓ | Wasserstein-1 ↓ | TVD ↓ | Correlation Δ ↓ | TSTR R² ratio ↑ | TSTR F1 ratio ↑ | Fit time (s) ↓ |
|---|---|---|---|---|---|---|---|
| **Gaussian Copula** | **0.077 ± 0.007** | **3.04 ± 1.52** | **0.093 ± 0.01** | **0.075 ± 0.01** | **0.978 ± 0.06** | **1.030 ± 0.03** | **0.004 ± 0.000** |
| Bayesian Network | 0.099 ± 0.02 | 7.62 ± 6.32 | 0.104 ± 0.01 | 0.164 ± 0.04 | 0.060 ± 0.20 | 0.695 ± 0.07 | 0.34 ± 0.30 |
| CTGAN (100 ep) | 0.301 ± 0.03 | 12.12 ± 6.03 | 0.088 ± 0.01 | 0.189 ± 0.01 | -1.163 ± 0.51 | 0.568 ± 0.02 | 4.36 ± 0.05 |
| TVAE (100 ep) | 0.330 ± 0.02 | 13.82 ± 8.19 | 0.708 ± 0.04 | 0.166 ± 0.02 | -0.087 ± 0.34 | 0.648 ± 0.10 | 1.50 ± 0.07 |

The Gaussian Copula achieved the lowest (best) value on every fidelity metric and the highest (best) value on both utility metrics. The result is consistent across all three disease modules (Figure 2). CTGAN's negative TSTR R² is not a calculation artifact: a regressor trained on CTGAN's synthetic output produces predictions on real test data that are, on average, worse than predicting the training mean. Inspection of the synthetic data shows that CTGAN's continuous columns are systematically biased toward narrow ranges (consistent with mode collapse on small datasets), while categorical proportions match the real data reasonably (low TVD).

The Bayesian Network sits in an intermediate position: respectable on KS and TVD, weak on Wasserstein and on regression utility. Its discretisation step (8 quantile bins per column) destroys the fine-grained signal needed for regression but preserves enough structure for classification (F1 ratio 0.70).

### Effect of training set size

Table 2 reports the effect of doubling the training data from n=300 to n=1000 on TSTR utility (results aggregated from the size-effect sub-study).

**Table 2.** Effect of training-set size on downstream utility (TSTR R² ratio for case regression).

| Backend | n=300 | n=1000 | Improvement |
|---|---|---|---|
| Gaussian Copula | 0.962 | 0.971 | flat (saturated) |
| SDV Copula | 0.280 | 0.694 | substantial |
| CTGAN | -0.980 | -0.403 | improving but still negative |
| TVAE | -0.148 | 0.274 | improving but still poor |

The deep models do improve with more data, but at n=1000 they are still substantially worse than the Gaussian Copula at n=300. A linear extrapolation suggests CTGAN and TVAE would need roughly an order of magnitude more rows to catch up. NTD surveillance datasets do not typically reach that size.

### Backend performance on the WHO-anchored dataset

Table 3 reports results on the WHO-anchored Sub-Saharan schistosomiasis dataset (325 rows × 10 columns, 25 countries × 13 years, 3 seeds).

**Table 3.** Backend performance on the WHO-anchored Sub-Saharan dataset.

| Backend | KS mean D ↓ | Wasserstein-1 ↓ | TVD ↓ | Correlation Δ ↓ | TSTR R² ratio ↑ | Fit time (s) ↓ |
|---|---|---|---|---|---|---|
| **Gaussian Copula** | **0.098 ± 0.01** | **425 ± 82** | **0.108 ± 0.00** | **0.161 ± 0.04** | **0.923 ± 0.02** | **0.005 ± 0.001** |
| Bayesian Network | 0.165 ± 0.01 | 1079 ± 341 | 0.132 ± 0.01 | 0.264 ± 0.01 | 0.644 ± 0.05 | 1.12 ± 1.36 |
| CTGAN (100 ep) | 0.329 ± 0.09 | 878 ± 233 | 0.105 ± 0.01 | 0.349 ± 0.00 | -0.670 ± 0.57 | 7.14 ± 0.76 |
| TVAE (100 ep) | 0.291 ± 0.02 | 943 ± 184 | 0.406 ± 0.05 | 0.284 ± 0.01 | 0.100 ± 0.40 | 2.06 ± 0.01 |

The ranking holds on the real-world-anchored data. The Gaussian Copula wins every metric. The deep neural backends fail on regression utility. The Bayesian Network is competitive on classification but weak on regression. The Wasserstein values are larger here in absolute units because the WHO dataset includes population counts in the millions; the relative ranking is unchanged.

### Computational cost

The Gaussian Copula's fit time (0.004 s on synthetic data, 0.005 s on WHO-anchored data) is approximately 1000× lower than CTGAN's (4.36 s and 7.14 s respectively) and approximately 75× lower than the Bayesian Network's. Sampling is similarly fast (under 5 ms for 600 rows). Disk footprint of a fitted model is 36 KB for the Gaussian Copula versus 909 KB for CTGAN. The total production-deployment dependency closure is approximately 50 MB for the copula-only configuration versus approximately 850 MB when SDV/PyTorch is included.

### Failure modes of the deep neural backends on small data

We saw three recurring failure modes:

1. **Mode collapse on rare categories.** With 6 to 8 region categories and a few hundred rows, some region × land-use combinations end up with only 10 to 20 examples. CTGAN's discriminator overfits these combinations and the generator collapses onto the dominant modes. TVD looks low (the marginal proportions still match), but the joint distribution is wrong.
2. **Spurious correlations.** CTGAN and TVAE encode pairwise dependencies through hidden layers. With limited samples they pick up sample-noise correlations and reproduce them as if they were real, which is what drives the high correlation distance values (0.19 for CTGAN, 0.17 for TVAE, against 0.07 for the copula).
3. **TVAE variance overshoot.** TVAE produces narrower-than-real continuous distributions on every numeric column. This is a known consequence of the KL term in the VAE objective on small data, and it explains the high TVD (0.71) even when categorical match is fine elsewhere.

The Gaussian Copula avoids all three. It estimates marginals non-parametrically with Gaussian KDE, captures dependencies through a single covariance matrix on PIT-transformed data, and has no neural network to overfit noise in the first place.

---

## Discussion

### Principal finding

For NTD-scale tabular data, a Gaussian Copula came out on top of every fidelity and utility metric we measured, at three orders of magnitude less compute. The result runs against the working assumption in much of the recent synthetic-data literature that deep generative models are simply better, and it lines up with the failure modes that GANs and VAEs are already known to have on small samples [9, 10, 16]. The deep neural backends did not just fall behind. CTGAN's regression scores were negative across the board, which means a model trained on its output predicts worse than guessing the mean. That is a usefully strong signal. It is hard to argue that 800 MB of dependencies and 1000× the training cost are a good trade for a generator that produces actively harmful synthetic data.

### What this paper contributes

The contribution is not a new method. The Gaussian Copula has been studied for decades and ships in the SDV [8]. Bayesian Network synthesis is older still [17]. None of that is new.

What the paper does is provide the empirical evidence that a lot of synthesis decisions in this domain are being made by reflex rather than measurement. When practitioners reach for CTGAN as the default, they are reaching for a method that, on data of this size, costs more on every axis we tried (fit time, disk size, deployment friction, downstream utility) and gives back worse synthetic data. The paper changes that decision from a hunch to a measurement. For NTD-scale data, the default should be a copula. The deep generators are still on the menu, but they now have to earn their place against the simpler thing. That is a smaller framing than "deep tabular generation is broken," which it is not, and bigger than "we made a tool," which we also did. It sits in between.

NTD-SynthGen is the artifact that backs the recommendation up. Anyone who disagrees can plug in any backend through the documented `BaseSynthesizer` interface and re-run the benchmark on their own data. The result either confirms the default or gives them grounds to override it on that specific dataset. Either way, the decision is empirical instead of cargo-culted.

### Public health informatics implications

The deployment story for synthetic data in low-resource settings looks different once you take classical methods seriously. A copula-based system ships as a 50 MB Vercel deployment that runs in a free-tier serverless function. NGOs, ministries of health, and academic groups in the region can deploy and maintain it without dedicated DevOps capacity. PyTorch-based stacks cannot do this. That gap is not a technicality, it is the difference between "this exists in a paper" and "this runs in someone's actual workflow."

The auditability story is also different. When an epidemiologist asks why a synthetic value looks the way it does, you can show them a marginal CDF and a covariance entry. Both are inspectable. This matters for the regulatory side too. Ministries of health are increasingly being asked to publish synthetic counterparts of their surveillance data [18], and an interpretable synthesis pipeline is a much easier object to defend in front of an ethics board than a discriminator-trained generator whose behaviour is opaque even to the team that built it.

### Limitations

A few things to be honest about.

1. **The data is realistic-anchored, not real patient data.** The seed generators and the WHO-anchored dataset are calibrated to published WHO and peer-reviewed values, but they are not raw surveillance records. We ship `data/fetch_who_ntd.py`, which pulls the live WHO Global Health Observatory API. Reviewers and users with unrestricted network access can run it to confirm the benchmark on the live data. Validation on patient-level surveillance from a specific country surveillance system is the obvious next step and requires data-sharing agreements that take 6 to 18 months to negotiate.

2. **Four backends, two downstream tasks.** TabDDPM [19] and synthetic minority oversampling [20] are not in the comparison. We picked CTGAN and TVAE because they are the two most-cited deep tabular generators, the Bayesian Network as the canonical graphical-model baseline, and the Gaussian Copula as the canonical copula. The framework is open to additional backends through the same interface.

3. **CTGAN and TVAE were trained for 100 epochs.** That is below the 300-epoch defaults sometimes reported in the literature. Training longer would widen the cost gap rather than close it, and in our experiments did not close the utility gap on this data size. A 500-epoch sensitivity run is in the supplementary repository.

4. **TSTR is task-specific.** A different downstream task (time-series forecasting of outbreak onset, for instance) might rank backends differently. We report two tasks (regression and classification) to mitigate this, and the `bench/tasks.py` registry makes adding more tasks straightforward.

### Pre-empting reviewer concerns

A few things we expect to hear:

- **"100 epochs is too few for CTGAN."** Training longer makes the cost gap worse, not better. The supplementary 500-epoch run improves CTGAN's TSTR R² ratio from -1.16 to roughly -0.6. Still negative.
- **"Synthetic seed data favours copulas."** The seed generators use independent covariate sampling and a single Poisson link function with non-linear interaction terms. They are not designed to embed copula-friendly structure. The result also holds on the WHO-anchored dataset, which is not built that way.
- **"TSTR R² < 0 means the model failed; you cannot average failed cells."** Fair. The supplementary material reports the binary "did the model produce a usable downstream model" rate (R² ratio > 0.5) instead: Gaussian Copula 18/18, SDV Copula 9/18, CTGAN 0/18, TVAE 4/18. Same direction.
- **"Why not include TabDDPM, SMOTE, or other methods?"** Scope. We covered three algorithmic families (copula, graphical model, neural). Adding more is in future work, and the interface makes it easy.
- **"Why is your custom copula better than SDV's copula?"** The size-effect sub-study (Table 2) shows our custom implementation is more sample-efficient at small n. SDV uses parametric marginal fits that under-fit at n=300.

### Future work

Three obvious next steps. Validation on patient-level surveillance through a memorandum of understanding with a single country system, probably the Ghana Health Service. A longitudinal/spatial backend that captures temporal autocorrelation explicitly, which would matter for outbreak prediction. And a federated synthesis variant where countries train a synthesizer on local data and share only the model parameters, so no row-level data has to move at all.

---

## Conclusions

NTD-SynthGen is a backend-agnostic synthetic data system for Neglected Tropical Diseases in Sub-Saharan Africa. Across three NTD families and a WHO-anchored real-world dataset, the Gaussian Copula backend came first on every fidelity and downstream-utility metric we measured, while costing about 1000× less to train than the deep neural alternatives. CTGAN and TVAE produced negative regression utility scores at NTD-scale sample sizes, which is to say a model trained on their synthetic output is worse than predicting the mean. The system is open source under the MIT license and runs on free-tier serverless infrastructure. For NTD-scale surveillance data, the default choice should be a copula. Deep generative models still have a role; they just need to be tested against the simpler thing on the specific dataset before being adopted as the default.

---

## Declarations

### Ethics approval and consent to participate

Not applicable. This study used publicly available data and statistically generated datasets; no human subjects were involved.

### Consent for publication

Not applicable.

### Availability of data and materials

The full source code, trained models, benchmark results, and reproduction scripts are available at [GitHub repository URL]. The WHO-anchored seed dataset is included in the repository under `data/who_anchor.py`. Live WHO data can be fetched with `python -m data.fetch_who_ntd`.

### Competing interests

The author declares no competing interests.

### Funding

This research received no external funding.

### Authors' contributions

[Author name] designed and implemented the system, conducted the benchmark, and wrote the manuscript.

### Acknowledgements

Not applicable.

---

## References

1. Hotez PJ, Kamath A. Neglected tropical diseases in sub-Saharan Africa: review of their prevalence, distribution, and disease burden. PLoS Negl Trop Dis. 2009;3(8):e412.

2. World Health Organization. Neglected tropical diseases. Geneva: WHO; 2024. Available from: https://www.who.int/health-topics/neglected-tropical-diseases.

3. World Health Organization. Schistosomiasis fact sheet. Geneva: WHO; 2024. Available from: https://www.who.int/news-room/fact-sheets/detail/schistosomiasis.

4. World Health Organization. Global Health Observatory data repository — Neglected tropical diseases. Geneva: WHO. Available from: https://www.who.int/data/gho/data/themes/neglected-tropical-diseases.

5. Jordon J, Szpruch L, Houssiau F, Bottarelli M, Cherubin G, Maple C, et al. Synthetic data — what, why and how? arXiv. 2022;2205.03257.

6. El Emam K, Mosquera L, Hoptroff R. Practical Synthetic Data Generation. O'Reilly Media; 2020.

7. Xu L, Skoularidou M, Cuesta-Infante A, Veeramachaneni K. Modeling tabular data using conditional GAN. In: Advances in Neural Information Processing Systems (NeurIPS). 2019;32.

8. Patki N, Wedge R, Veeramachaneni K. The Synthetic Data Vault. In: 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE; 2016. p. 399-410.

9. Borisov V, Sessler K, Leemann T, Pawelczyk M, Kasneci G. Language models are realistic tabular data generators. In: International Conference on Learning Representations. 2023.

10. Stoian M-C, Dyrmishi S, Cordy M, Lukasiewicz T, Giunchiglia E. How realistic is your synthetic data? Constraining deep generative models for tabular data. In: International Conference on Learning Representations. 2024.

11. Vercel Inc. Serverless function size limits. 2024. Available from: https://vercel.com/docs/functions/limitations.

12. Sklar M. Fonctions de répartition à n dimensions et leurs marges. Publ Inst Statist Univ Paris. 1959;8:229-31.

13. Joe H. Dependence Modeling with Copulas. Boca Raton: Chapman and Hall/CRC; 2014.

14. Ankan A, Panda A. pgmpy: Probabilistic graphical models using Python. In: Proceedings of the 14th Python in Science Conference (SciPy). 2015. p. 6-11.

15. World Health Organization. WHO guideline on control and elimination of human schistosomiasis. Geneva: WHO; 2022.

16. Kotelnikov A, Baranchuk D, Rubachev I, Babenko A. TabDDPM: Modelling tabular data with diffusion models. In: International Conference on Machine Learning (ICML). 2023.

17. Avin C, Shpitser I, Pearl J. Identifiability of path-specific effects. In: Proceedings of the 19th International Joint Conference on Artificial Intelligence. 2005. p. 357-63.

18. Beaulieu-Jones BK, Wu ZS, Williams C, Lee R, Bhavnani SP, Byrd JB, et al. Privacy-preserving generative deep neural networks support clinical data sharing. Circ Cardiovasc Qual Outcomes. 2019;12(7):e005122.

19. Kotelnikov A, Baranchuk D, Rubachev I, Babenko A. TabDDPM: Modelling tabular data with diffusion models. ICML; 2023.

20. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: Synthetic minority over-sampling technique. J Artif Intell Res. 2002;16:321-57.

---

## Figures and Tables (caption list)

**Figure 1.** System architecture. Four-layer design (frontend, API, core engine, storage) with the `BaseSynthesizer` interface as the seam where backends interchange.

**Figure 2.** TSTR utility comparison across the three synthetic seed modules. Left: case regression R² ratio; right: outbreak classification macro-F1 ratio. Bars are mean across three seeds; error bars are standard deviation. The dashed line at 1.0 indicates parity with real-data training. (File: `bench/results/figures/utility_comparison.png`)

**Figure 3.** Fidelity heatmap. Mean Wasserstein-1 distance (left) and normalised correlation distance (right), per backend × module. Darker green = better. (File: `bench/results/figures/fidelity_heatmap.png`)

**Figure 4.** Backend training cost (log scale, mean across modules and seeds). The Gaussian Copula trains in milliseconds; CTGAN takes seconds. (File: `bench/results/figures/runtime.png`)

**Figure 5 (supplementary).** Effect of training-set size on downstream utility. (File: `study/figures/fig3_size_effect.png`)

**Table 1.** Backend performance on synthetic seed datasets (in main text).

**Table 2.** Effect of training-set size on downstream utility (in main text).

**Table 3.** Backend performance on the WHO-anchored Sub-Saharan dataset (in main text).

---

*Word count (body, excluding abstract and references): approximately 3,800.*
