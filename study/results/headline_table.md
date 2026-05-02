# Headline benchmark results

Mean across all 18 evaluation cells (3 datasets × 2 sizes × 3 seeds).
Lower is better for fidelity columns; higher is better for utility ratios.

| Backend | KS↓ | W₁↓ | TVD↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ | Fit (s)↓ | Sample (s)↓ | Disk (KB)↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Ours (Gaussian Copula)** | 0.034 | 0.050 | 0.048 | 0.036 | 0.966 | 1.052 | 0.00 | 0.00 | 36 |
| **SDV Gaussian Copula** | 0.088 | 0.127 | 0.053 | 0.042 | 0.487 | 0.625 | 0.36 | 0.04 | 88 |
| **CTGAN** | 0.301 | 0.695 | 0.079 | 0.151 | -0.691 | 0.245 | 6.75 | 0.05 | 909 |
| **TVAE** | 0.324 | 0.505 | 0.672 | 0.117 | 0.063 | 0.546 | 2.26 | 0.05 | 338 |

KS = Kolmogorov-Smirnov statistic (mean across numeric columns). 
W₁ = standardised Wasserstein-1 distance. 
TVD = total variation distance (categorical columns). 
CorrΔ = normalised Frobenius distance between Pearson correlation matrices. 
R² ratio = Train-on-Synthetic-Test-on-Real R² ÷ real-on-real R². 
F1 ratio = same idea for outbreak classification (cases > 75th percentile).
# Effect of training-set size

Does the picture change between n=300 and n=1000?

| Backend | n | KS↓ | W₁↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ | Fit (s) |
|---|---|---|---|---|---|---|---|
| Ours (Gaussian Copula) | 300 | 0.043 | 0.063 | 0.042 | 0.962 | 1.109 | 0.00 |
| Ours (Gaussian Copula) | 1000 | 0.026 | 0.036 | 0.030 | 0.971 | 0.995 | 0.01 |
| SDV Gaussian Copula | 300 | 0.104 | 0.159 | 0.050 | 0.280 | 0.405 | 0.33 |
| SDV Gaussian Copula | 1000 | 0.072 | 0.095 | 0.033 | 0.694 | 0.844 | 0.39 |
| CTGAN | 300 | 0.339 | 0.779 | 0.156 | -0.980 | 0.372 | 4.48 |
| CTGAN | 1000 | 0.263 | 0.612 | 0.145 | -0.403 | 0.117 | 9.02 |
| TVAE | 300 | 0.331 | 0.515 | 0.115 | -0.148 | 0.542 | 1.48 |
| TVAE | 1000 | 0.316 | 0.496 | 0.118 | 0.274 | 0.550 | 3.03 |
# Per-dataset breakdown (averaged over both sizes and seeds)


## skin_ntd

| Backend | KS↓ | W₁↓ | TVD↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ |
|---|---|---|---|---|---|---|
| Ours (Gaussian Copula) | 0.035 | 0.050 | 0.048 | 0.037 | 0.973 | 1.053 |
| SDV Gaussian Copula | 0.074 | 0.114 | 0.059 | 0.041 | 0.674 | 0.770 |
| CTGAN | 0.278 | 0.658 | 0.091 | 0.142 | -0.995 | 0.167 |
| TVAE | 0.298 | 0.478 | 0.698 | 0.123 | 0.338 | 0.654 |

## vector_borne

| Backend | KS↓ | W₁↓ | TVD↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ |
|---|---|---|---|---|---|---|
| Ours (Gaussian Copula) | 0.034 | 0.048 | 0.046 | 0.036 | 0.989 | 1.085 |
| SDV Gaussian Copula | 0.089 | 0.126 | 0.047 | 0.046 | 0.320 | 0.301 |
| CTGAN | 0.282 | 0.650 | 0.081 | 0.145 | -0.467 | 0.320 |
| TVAE | 0.310 | 0.489 | 0.673 | 0.096 | 0.076 | 0.864 |

## water_borne

| Backend | KS↓ | W₁↓ | TVD↓ | CorrΔ↓ | R² ratio↑ | F1 ratio↑ |
|---|---|---|---|---|---|---|
| Ours (Gaussian Copula) | 0.035 | 0.051 | 0.049 | 0.034 | 0.937 | 1.017 |
| SDV Gaussian Copula | 0.101 | 0.141 | 0.053 | 0.039 | 0.467 | 0.803 |
| CTGAN | 0.343 | 0.778 | 0.064 | 0.165 | -0.612 | 0.247 |
| TVAE | 0.363 | 0.550 | 0.646 | 0.131 | -0.225 | 0.099 |