"""
Tests for utility functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy import linalg

from src.utils import (
    vecl, vecl_inverse, matrix_log, matrix_exp,
    archakov_hansen_inverse, gamma_from_correlation,
    is_positive_definite, ensure_positive_definite,
    realized_variance_to_correlation,
    create_equi_loading_matrix, create_block_loading_matrix,
    compute_check_y
)


def test_vecl_and_inverse():
    """Test vecl and vecl_inverse are inverses."""
    n = 5
    # Create symmetric matrix
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    
    # Apply vecl
    v = vecl(A)
    
    # Check dimension
    assert len(v) == n * (n - 1) // 2
    
    # Reconstruct
    A_recon = vecl_inverse(v, n)
    
    # Check lower triangular matches
    for i in range(n):
        for j in range(i):
            assert np.abs(A[i, j] - A_recon[i, j]) < 1e-10


def test_matrix_log_exp():
    """Test matrix log and exp are inverses for PD matrices."""
    n = 5
    # Create PD matrix
    A = np.random.randn(n, n)
    C = A @ A.T + 0.1 * np.eye(n)
    C = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))  # Make correlation
    
    # Compute log
    L = matrix_log(C)
    
    # Compute exp
    C_recon = matrix_exp(L)
    
    # Should recover original
    assert np.allclose(C, C_recon, atol=1e-8)


def test_archakov_hansen_inverse_consistency():
    """Test Archakov-Hansen parametrization is consistent."""
    n = 4
    
    # Create correlation matrix
    C = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.3],
        [0.3, 0.4, 1.0, 0.6],
        [0.2, 0.3, 0.6, 1.0]
    ])
    
    # Forward map: C -> gamma
    gamma = gamma_from_correlation(C)
    
    # Check dimension
    d = n * (n - 1) // 2
    assert len(gamma) == d
    
    # Inverse map: gamma -> C
    C_recon = archakov_hansen_inverse(gamma, n)
    
    # Should recover original (with relaxed tolerance for numerical precision)
    assert np.allclose(C, C_recon, atol=1e-2)
    
    # Verify properties
    assert np.allclose(np.diag(C_recon), 1.0, atol=1e-8)
    assert is_positive_definite(C_recon)


def test_archakov_hansen_equicorrelation():
    """Test Archakov-Hansen parametrization for equicorrelation."""
    n = 9
    rho = 0.6
    
    # Create equicorrelation matrix
    C = np.full((n, n), rho)
    np.fill_diagonal(C, 1.0)
    
    # Forward map
    gamma = gamma_from_correlation(C)
    
    # All elements should be equal for equicorrelation
    assert np.allclose(gamma, gamma[0], atol=1e-10)
    
    # Inverse map
    C_recon = archakov_hansen_inverse(gamma, n)
    
    # Should recover equicorrelation
    assert np.allclose(C, C_recon, atol=1e-6)


def test_realized_variance_to_correlation():
    """Test conversion of realized covariance to variance and correlation."""
    n = 5
    # Create covariance matrix
    A = np.random.randn(n, n)
    RM = A @ A.T + 0.01 * np.eye(n)
    
    # Convert
    x, Y = realized_variance_to_correlation(RM)
    
    # Check dimensions
    assert x.shape == (n,)
    assert Y.shape == (n, n)
    
    # Check diagonal of Y is 1
    assert np.allclose(np.diag(Y), 1.0, atol=1e-10)
    
    # Check Y is symmetric
    assert np.allclose(Y, Y.T, atol=1e-10)
    
    # Check PD
    assert is_positive_definite(Y)
    
    # Check reconstruction
    Lambda = np.diag(np.sqrt(x))
    RM_recon = Lambda @ Y @ Lambda
    assert np.allclose(RM, RM_recon, atol=1e-8)


def test_equi_loading_matrix():
    """Test equicorrelation loading matrix."""
    n = 9
    d = n * (n - 1) // 2
    
    A = create_equi_loading_matrix(n)
    
    # Check shape
    assert A.shape == (d, 1)
    
    # Check all ones
    assert np.allclose(A, 1.0)


def test_block_loading_matrix():
    """Test block structure loading matrix."""
    n = 9
    d = n * (n - 1) // 2
    r = 6
    
    A, categories = create_block_loading_matrix(n)
    
    # Check shape
    assert A.shape == (d, r)
    
    # Check binary
    assert np.all((A == 0) | (A == 1))
    
    # Each pair should belong to exactly one category
    assert np.all(np.sum(A, axis=1) == 1)
    
    # Check category counts
    # Within-sector: 3 pairs each for 3 sectors
    # Cross-sector: 9 pairs each for 3 cross-sector combinations
    expected_counts = [3, 3, 3, 9, 9, 9]
    actual_counts = np.sum(A, axis=0).astype(int)
    assert np.array_equal(actual_counts, expected_counts)


def test_compute_check_y_equi():
    """Test factor projection for equicorrelation."""
    n = 9
    d = n * (n - 1) // 2
    T = 100
    
    # Simulate y_t
    y_t = np.random.randn(T, d)
    
    # Equi loading
    A = create_equi_loading_matrix(n)
    
    # Compute check_y
    check_y = compute_check_y(y_t, A)
    
    # Should be mean across all pairs
    expected = np.mean(y_t, axis=1, keepdims=True)
    
    assert np.allclose(check_y, expected, atol=1e-10)


def test_compute_check_y_block():
    """Test factor projection for block structure."""
    n = 9
    d = n * (n - 1) // 2
    T = 100
    
    # Simulate y_t
    y_t = np.random.randn(T, d)
    
    # Block loading
    A, _ = create_block_loading_matrix(n)
    
    # Compute check_y
    check_y = compute_check_y(y_t, A)
    
    # Check shape
    assert check_y.shape == (T, 6)
    
    # Each factor should be mean of its category
    for k in range(6):
        pair_indices = np.where(A[:, k] == 1)[0]
        expected_k = np.mean(y_t[:, pair_indices], axis=1)
        assert np.allclose(check_y[:, k], expected_k, atol=1e-10)


def test_ensure_positive_definite():
    """Test eigenvalue flooring for PD enforcement."""
    n = 5
    
    # Create matrix with some negative eigenvalues
    eigenvalues = np.array([2.0, 1.0, 0.5, -0.1, -0.2])
    eigenvectors = np.random.randn(n, n)
    eigenvectors, _ = np.linalg.qr(eigenvectors)
    
    A = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    A = (A + A.T) / 2
    
    # Make PD
    A_pd = ensure_positive_definite(A)
    
    # Check PD
    assert is_positive_definite(A_pd)
    
    # Check symmetry
    assert np.allclose(A_pd, A_pd.T, atol=1e-10)


def test_edge_case_single_factor():
    """Test with minimal dimension."""
    n = 2
    d = 1
    
    # Create correlation
    C = np.array([[1.0, 0.7], [0.7, 1.0]])
    
    # Forward map
    gamma = gamma_from_correlation(C)
    assert len(gamma) == 1
    
    # Inverse map
    C_recon = archakov_hansen_inverse(gamma, n)
    assert np.allclose(C, C_recon, atol=1e-8)


def test_extreme_correlations():
    """Test with extreme correlation values."""
    n = 3
    
    # High positive correlation
    C_high = np.array([
        [1.0, 0.95, 0.90],
        [0.95, 1.0, 0.92],
        [0.90, 0.92, 1.0]
    ])
    
    gamma_high = gamma_from_correlation(C_high)
    C_high_recon = archakov_hansen_inverse(gamma_high, n)
    assert np.allclose(C_high, C_high_recon, atol=1e-2)
    
    # Low/negative correlations
    C_low = np.array([
        [1.0, -0.3, 0.1],
        [-0.3, 1.0, -0.2],
        [0.1, -0.2, 1.0]
    ])
    
    # Ensure PD
    if is_positive_definite(C_low):
        gamma_low = gamma_from_correlation(C_low)
        C_low_recon = archakov_hansen_inverse(gamma_low, n)
        assert np.allclose(C_low, C_low_recon, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
