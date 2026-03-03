# Multivariate Realized GARCH Models with Matrix-Logarithm Correlation Parametrization

This repository implements the complete methodology from Archakov & Hansen (2024) for modeling multivariate realized covariance matrices using the Matrix Realized GARCH (MRG) model with matrix-logarithm correlation parametrization.

## Overview

The project analyzes a 9-asset equity panel (3 Energy, 3 Health Care, 3 IT stocks) over 2002-2020 using high-frequency realized measures and multivariate GARCH models.

### Key Features

- **Data Construction**: TAQ intraday data cleaning and realized kernel covariance estimation
- **Univariate Realized GARCH**: Stage 1 volatility filtering for all 9 assets
- **Multivariate Correlation Models**: Implementation of 9 model specifications:
  - CCC+ (Constant Conditional Correlation): Equi/Block/Full
  - DCC+ (Dynamic Conditional Correlation): Equi/Block/Full  
  - MRG (Matrix Realized GARCH): Equi/Block/Full
- **Archakov-Hansen Parametrization**: Matrix logarithm-based correlation modeling
- **Out-of-Sample Evaluation**: Annual rolling re-estimation with Model Confidence Set
- **Portfolio Applications**: Global minimum-variance portfolio construction and evaluation

## Experiments

1. **exp_1**: Data Construction and Realized Measures Computation
2. **exp_2**: Univariate Realized GARCH Estimation (Stage 1)
3. **exp_3**: Multivariate Correlation Model Estimation (Stage 2)
4. **exp_4**: Out-of-Sample Predictive Log-Likelihood Evaluation
5. **exp_5**: GMV Portfolio Performance Evaluation
6. **exp_6**: Gaussianity Diagnostics for Transformed Correlations

## Installation

```bash
pip install -r requirements.txt
```

## Data Requirements

- **MASSIVE API**: Requires MASSIVE_TOKEN environment variable for accessing TAQ and CRSP data
- **9 Tickers**: CVX, MRO, OXY (Energy); JNJ, LLY, MRK (Health Care); AAPL, MU, ORCL (IT)
- **Time Period**: 2002-01-02 to 2020-12-31 (4,744 trading days)
- **In-Sample**: 2002-01-02 to 2011-12-30 (2,496 days)
- **Out-of-Sample**: 2012-01-03 to 2020-12-31 (2,248 days)

## Usage

```python
from src.exp_1_data_construction import build_daily_panel
from src.exp_2_realized_garch import estimate_univariate_realized_garch
from src.exp_3_correlation_models import estimate_mrg_models

# Build data panel
panel = build_daily_panel(tickers=['CVX', 'MRO', 'OXY', 'JNJ', 'LLY', 'MRK', 'AAPL', 'MU', 'ORCL'])

# Estimate models
results = estimate_univariate_realized_garch(panel)
```

## Running Experiments

```bash
# Run all experiments
python run_experiments.py

# Run specific experiment
python -m pytest tests/test_exp_1.py -v
```

## Results

All experimental results, metrics, and visualizations are saved in the `results/` directory:

- `results/RESULTS.md`: Summary of all metrics and findings
- `results/figures/`: Q-Q plots, correlation dynamics, portfolio performance charts
- `results/tables/`: Parameter estimates, log-likelihood comparisons, MCS results

## Technical Details

### Archakov-Hansen Correlation Parametrization

The key innovation is the matrix-logarithm correlation parametrization:

```
γ(C) = vecl(log C)  (forward map)
C(γ) = D^{-1/2} * exp(L) * D^{-1/2}  (inverse map, Corollary 1)
```

where L is the symmetric matrix with vecl(L) = γ and zero diagonal, and D = diag(exp(L)).

### Model Specifications

- **r=1 (Equi)**: Single correlation factor (equicorrelation)
- **r=6 (Block)**: Six factors (3 within-sector + 3 cross-sector correlations)
- **r=36 (Full)**: Full correlation dynamics (36 unique correlation pairs)

## References

Archakov, I., & Hansen, P. R. (2024). A Multivariate Realized GARCH Model. *Journal of Business & Economic Statistics*.

Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2011). Multivariate realised kernels: Consistent positive semi-definite estimators of the covariation of equity prices with noise and non-synchronous trading. *Journal of Econometrics*, 162(2), 149-169.

Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453-497.

## License

MIT License
