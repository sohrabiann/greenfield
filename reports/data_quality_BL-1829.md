# Data Quality Report

**Generated**: 2025-12-04 13:48:37

**Dataset Shape**: 12 rows × 84 features

## Missing Values Analysis

- **Total cells**: 1,008
- **Missing cells**: 12 (1.19%)
- **Features with missing values**: 12

### Top Features with Missing Values

| Feature | Missing Count | Missing % |
|---------|---------------|-----------|
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_mean | 1 | 8.33% |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_std | 1 | 8.33% |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_min | 1 | 8.33% |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_max | 1 | 8.33% |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_last | 1 | 8.33% |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_slope | 1 | 8.33% |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_mean | 1 | 8.33% |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_std | 1 | 8.33% |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_min | 1 | 8.33% |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_max | 1 | 8.33% |

## Outlier Detection

- **Method**: iqr (factor=3.0)
- **Total outliers detected**: 58
- **Features with outliers**: 37

### Top Features with Outliers

| Feature | Count | % | Min Outlier | Max Outlier |
|---------|-------|---|-------------|-------------|
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_mean | 4 | 33.3% | 382.30 | 482.27 |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_min | 4 | 33.3% | -0.03 | 482.27 |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_mean | 3 | 25.0% | -0.02 | 482.13 |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_last | 3 | 25.0% | -0.02 | 482.27 |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_slope | 3 | 25.0% | -0.06 | 0.01 |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_mean | 2 | 16.7% | 482.13 | 482.27 |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_min | 2 | 16.7% | 482.13 | 482.27 |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_max | 2 | 16.7% | 482.13 | 482.27 |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_last | 2 | 16.7% | 482.13 | 482.27 |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_count | 2 | 16.7% | 0.00 | 426.00 |

## Feature Correlation Analysis

- **Correlation threshold**: 0.95
- **High correlation pairs found**: 153

### Highly Correlated Feature Pairs (Top 10)

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| w72h_CH-DRA_PROC_AIR:TT18179A.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179A.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179B.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179A.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179C.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179A.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179A.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179B.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179B.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179B.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179C.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179B.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179A.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179C.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179B.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179C.MEAS_count | 1.000 |
| w72h_CH-DRA_PROC_AIR:TT18179C.MEAS_count | w168h_CH-DRA_PROC_AIR:TT18179C.MEAS_count | 1.000 |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_max | w24h_CH-DRA_PROC_AIR:IT18179.MEAS_last | 1.000 |

## Summary Statistics

**Top 10 features by standard deviation (most variable):**

| Feature | Mean | Std | Min | Max | Missing % |
|---------|------|-----|-----|-----|-----------|
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_count | 929.00 | 271.78 | 66.00 | 1008.00 | 0.0% |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_min | 274.88 | 242.63 | -0.03 | 482.13 | 0.0% |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_min | 386.68 | 191.21 | -0.03 | 482.27 | 8.3% |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_max | 436.23 | 137.70 | -0.02 | 500.17 | 0.0% |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_last | 432.85 | 136.40 | -0.02 | 482.27 | 0.0% |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_mean | 427.37 | 135.49 | -0.02 | 482.13 | 0.0% |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_count | 395.50 | 124.56 | 0.00 | 432.00 | 0.0% |
| w72h_CH-DRA_PROC_AIR:IT18179.MEAS_std | 23.39 | 56.89 | 0.00 | 180.09 | 8.3% |
| w168h_CH-DRA_PROC_AIR:IT18179.MEAS_std | 22.75 | 42.66 | 0.00 | 140.88 | 0.0% |
| w24h_CH-DRA_PROC_AIR:IT18179.MEAS_count | 132.00 | 41.57 | 0.00 | 144.00 | 0.0% |

## Data Types
- **float64**: 72 features
- **int64**: 12 features

## Recommendations
- **Missing values**: Significant missing data detected. Use forward-fill + median imputation.
- **Outliers**: 58 outliers detected. Consider IQR clipping.
- **Correlation**: 153 highly correlated pairs. Consider removing redundant features.