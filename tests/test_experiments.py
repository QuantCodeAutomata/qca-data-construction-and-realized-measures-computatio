"""
Tests for experiment implementations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.exp_1_data_construction import (
    parzen_kernel, build_daily_panel, split_sample,
    compute_summary_statistics, compute_correlation_matrix
)
from src.exp_2_realized_garch import (
    leverage_function, measurement_function,
    filter_realized_garch, estimate_realized_garch_single
)
from src.exp_3_correlation_models import (
    estimate_ccc_equi, estimate_ccc_block, estimate_ccc_full
)
from src.utils import is_positive_definite


def test_parzen_kernel():
    """Test Parzen kernel function."""
    # Test regions
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    k = parzen_kernel(x)
    
    # Check bounds
    assert np.all(k >= 0)
    assert np.all(k <= 1)
    
    # Check specific values
    assert k[0] == 1.0  # k(0) = 1
    assert k[-1] == 0.0  # k(1.5) = 0
    
    # Check continuity at 0.5
    x_around_half = np.array([0.499, 0.501])
    k_around_half = parzen_kernel(x_around_half)
    assert np.abs(k_around_half[0] - k_around_half[1]) < 0.01


def test_build_daily_panel():
    """Test panel construction."""
    # Build small panel for testing
    panel = build_daily_panel(use_intraday=False)
    
    # Check structure
    assert 'dates' in panel
    assert 'returns' in panel
    assert 'realized_var' in panel
    assert 'realized_corr' in panel
    assert 'y_t' in panel
    assert 'tickers' in panel
    assert 'metadata' in panel
    
    # Check dimensions
    T = len(panel['dates'])
    n = len(panel['tickers'])
    d = n * (n - 1) // 2
    
    assert panel['returns'].shape == (T, n)
    assert panel['realized_var'].shape == (T, n)
    assert panel['realized_corr'].shape == (T, n, n)
    assert panel['y_t'].shape == (T, d)
    
    # Check realized variances are positive (after first day which may be zero)
    assert np.all(panel['realized_var'][1:] > 0)
    
    # Check realized correlations are valid (skip first few days due to initialization)
    for t in range(max(1, T-10), T):  # Check last 10
        Y_t = panel['realized_corr'][t]
        if np.any(Y_t != 0):  # Skip if not initialized
            assert np.allclose(np.diag(Y_t), 1.0, atol=1e-6)
            assert is_positive_definite(Y_t)


def test_split_sample():
    """Test sample splitting."""
    panel = build_daily_panel(use_intraday=False)
    
    insample, outsample = split_sample(panel)
    
    # Check sizes add up
    assert (insample['metadata']['n_days'] + 
            outsample['metadata']['n_days'] == 
            panel['metadata']['n_days'])
    
    # Check in-sample is before out-of-sample
    assert insample['dates'][-1] < outsample['dates'][0]


def test_summary_statistics():
    """Test summary statistics computation."""
    panel = build_daily_panel(use_intraday=False)
    
    stats = compute_summary_statistics(panel)
    
    # Check structure
    assert 'Ticker' in stats.columns
    assert 'Mean' in stats.columns
    assert 'Std' in stats.columns
    assert 'Skewness' in stats.columns
    
    # Check number of rows
    assert len(stats) == len(panel['tickers'])
    
    # Check std is positive
    assert np.all(stats['Std'] > 0)


def test_correlation_matrix():
    """Test correlation matrix computation."""
    panel = build_daily_panel(use_intraday=False)
    
    corr = compute_correlation_matrix(panel)
    
    # Check shape
    n = len(panel['tickers'])
    assert corr.shape == (n, n)
    
    # Check diagonal is 1
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-10)
    
    # Check symmetric
    assert np.allclose(corr.values, corr.values.T, atol=1e-10)
    
    # Check correlation bounds
    assert np.all(corr.values >= -1.0)
    assert np.all(corr.values <= 1.0)


def test_leverage_function():
    """Test leverage function."""
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    tau_1 = -0.1
    tau_2 = 0.05
    
    tau_z = leverage_function(z, tau_1, tau_2)
    
    # Check dimension
    assert len(tau_z) == len(z)
    
    # Check formula: tau_1*z + tau_2*(z^2 - 1)
    expected = tau_1 * z + tau_2 * (z**2 - 1)
    assert np.allclose(tau_z, expected)


def test_measurement_function():
    """Test measurement equation leverage function."""
    z = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    delta_1 = -0.05
    delta_2 = 0.02
    
    delta_z = measurement_function(z, delta_1, delta_2)
    
    # Check formula
    expected = delta_1 * z + delta_2 * (z**2 - 1)
    assert np.allclose(delta_z, expected)


def test_filter_realized_garch():
    """Test Realized GARCH filtering."""
    T = 100
    
    # Simulate data
    np.random.seed(42)
    returns = np.random.randn(T) * 0.01
    realized_var = np.abs(np.random.randn(T) * 0.0002) + 0.0001
    
    # Parameters
    params = np.array([
        0.0001,  # mu
        -0.5,    # omega
        0.90,    # beta
        -0.1,    # tau_1
        0.05,    # tau_2
        0.05,    # alpha
        -0.3,    # xi
        0.95,    # phi
        -0.05,   # delta_1
        0.02     # delta_2
    ])
    
    # Filter
    log_h, z, v = filter_realized_garch(params, returns, realized_var)
    
    # Check dimensions
    assert len(log_h) == T
    assert len(z) == T
    assert len(v) == T
    
    # Check no extreme values
    assert np.all(np.isfinite(log_h))
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(v))
    
    # Check standardized residuals have approximately unit variance
    assert np.abs(np.std(z) - 1.0) < 0.5


def test_estimate_realized_garch_single():
    """Test univariate Realized GARCH estimation."""
    T = 200
    
    # Simulate realistic data
    np.random.seed(123)
    
    # True parameters
    true_h = np.zeros(T)
    true_h[0] = 0.0002
    z_true = np.random.randn(T)
    
    for t in range(T-1):
        true_h[t+1] = 0.0001 + 0.92 * true_h[t] + 0.02 * z_true[t]**2
    
    returns = np.sqrt(true_h) * z_true
    realized_var = true_h * np.exp(np.random.randn(T) * 0.3)
    realized_var = np.maximum(realized_var, 1e-8)
    
    # Estimate
    results = estimate_realized_garch_single(returns, realized_var, n_starts=2)
    
    # Check results structure
    assert 'params' in results
    assert 'log_h' in results
    assert 'z' in results
    assert 'loglik' in results
    assert 'persistence' in results
    
    # Check parameter dimensions
    assert len(results['params']) == 10
    
    # Check persistence in reasonable range
    assert 0.5 < results['persistence'] < 1.0
    
    # Check filtered variances are positive
    assert np.all(results['h'] > 0)


def test_estimate_ccc_equi():
    """Test CCC+ Equicorrelation estimation."""
    T = 100
    n = 9
    
    # Simulate standardized residuals with correlation
    np.random.seed(42)
    rho = 0.6
    
    # Create equicorrelation matrix
    C_true = np.full((n, n), rho)
    np.fill_diagonal(C_true, 1.0)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(C_true)
    
    # Generate correlated residuals
    z_indep = np.random.randn(T, n)
    z_matrix = z_indep @ L.T
    
    # Estimate
    results = estimate_ccc_equi(z_matrix)
    
    # Check results
    assert 'model' in results
    assert 'C' in results
    assert 'rho' in results
    assert 'loglik' in results
    
    # Check correlation matrix properties
    C = results['C']
    assert C.shape == (n, n)
    assert np.allclose(np.diag(C), 1.0)
    assert is_positive_definite(C)
    
    # Check estimated rho is close to true
    assert np.abs(results['rho'] - rho) < 0.2


def test_estimate_ccc_block():
    """Test CCC+ Block estimation."""
    T = 100
    n = 9
    
    # Simulate standardized residuals
    np.random.seed(42)
    z_matrix = np.random.randn(T, n)
    
    # Estimate
    results = estimate_ccc_block(z_matrix, n)
    
    # Check results
    assert 'model' in results
    assert 'C' in results
    assert 'rhos' in results
    assert 'categories' in results
    
    # Check correlation matrix
    C = results['C']
    assert C.shape == (n, n)
    assert np.allclose(np.diag(C), 1.0)
    assert is_positive_definite(C)
    
    # Check number of correlation parameters
    assert len(results['rhos']) == 6
    assert len(results['categories']) == 6


def test_estimate_ccc_full():
    """Test CCC+ Full estimation."""
    T = 100
    n = 9
    
    # Simulate standardized residuals
    np.random.seed(42)
    z_matrix = np.random.randn(T, n)
    
    # Estimate
    results = estimate_ccc_full(z_matrix)
    
    # Check results
    assert 'model' in results
    assert 'C' in results
    assert 'loglik' in results
    
    # Check correlation matrix
    C = results['C']
    assert C.shape == (n, n)
    assert np.allclose(np.diag(C), 1.0)
    assert is_positive_definite(C)


def test_correlation_model_likelihood_ordering():
    """Test that more flexible models have higher likelihoods."""
    T = 100
    n = 9
    
    # Simulate standardized residuals
    np.random.seed(42)
    z_matrix = np.random.randn(T, n)
    
    # Estimate all three CCC models
    equi_results = estimate_ccc_equi(z_matrix)
    block_results = estimate_ccc_block(z_matrix, n)
    full_results = estimate_ccc_full(z_matrix)
    
    # Full should have highest likelihood
    assert full_results['loglik'] >= block_results['loglik']
    assert block_results['loglik'] >= equi_results['loglik']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
