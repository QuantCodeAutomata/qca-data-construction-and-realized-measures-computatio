"""
Utility functions for multivariate realized GARCH models.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy import linalg
import pandas as pd


def vecl(matrix: np.ndarray) -> np.ndarray:
    """
    Extract strictly lower-triangular elements in column-major order.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Symmetric matrix of shape (n, n)
        
    Returns:
    --------
    np.ndarray
        Vector of length d = n(n-1)/2 containing lower-triangular elements
        in column-major order: (2,1), (3,1), (3,2), (4,1), ..., (n,n-1)
    """
    n = matrix.shape[0]
    assert matrix.shape == (n, n), "Matrix must be square"
    
    result = []
    for j in range(n):
        for i in range(j + 1, n):
            result.append(matrix[i, j])
    
    return np.array(result)


def vecl_inverse(vec: np.ndarray, n: int) -> np.ndarray:
    """
    Reconstruct symmetric matrix from vecl representation.
    
    Parameters:
    -----------
    vec : np.ndarray
        Vector of length d = n(n-1)/2
    n : int
        Dimension of output matrix
        
    Returns:
    --------
    np.ndarray
        Symmetric matrix of shape (n, n) with zero diagonal
    """
    d = len(vec)
    assert d == n * (n - 1) // 2, f"Vector length {d} incompatible with n={n}"
    
    matrix = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j + 1, n):
            matrix[i, j] = vec[idx]
            matrix[j, i] = vec[idx]
            idx += 1
    
    return matrix


def matrix_log(C: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    """
    Compute matrix logarithm via eigendecomposition.
    
    Parameters:
    -----------
    C : np.ndarray
        Positive definite symmetric matrix
    symmetrize : bool
        Whether to symmetrize the result (default: True)
        
    Returns:
    --------
    np.ndarray
        Matrix logarithm log(C)
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(C)
    
    # Check for numerical issues
    if np.any(eigenvalues <= 0):
        min_eig = np.min(eigenvalues)
        if min_eig < -1e-10:
            raise ValueError(f"Matrix is not positive definite: min eigenvalue = {min_eig}")
        # Small negative eigenvalues due to numerical error - floor to small positive
        eigenvalues = np.maximum(eigenvalues, 1e-12)
    
    # Compute log of eigenvalues
    log_eigenvalues = np.log(eigenvalues)
    
    # Reconstruct: L = V * log(D) * V'
    L = eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.T
    
    # Symmetrize to eliminate numerical asymmetry
    if symmetrize:
        L = (L + L.T) / 2
    
    return L


def matrix_exp(L: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    """
    Compute matrix exponential via eigendecomposition.
    
    Parameters:
    -----------
    L : np.ndarray
        Symmetric matrix
    symmetrize : bool
        Whether to symmetrize the input (default: True)
        
    Returns:
    --------
    np.ndarray
        Matrix exponential exp(L)
    """
    if symmetrize:
        L = (L + L.T) / 2
    
    # Eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(L)
    
    # Compute exp of eigenvalues
    exp_eigenvalues = np.exp(eigenvalues)
    
    # Reconstruct: M = V * exp(D) * V'
    M = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
    
    return M


def archakov_hansen_inverse(gamma: np.ndarray, n: int, 
                            max_iter: int = 5, tol: float = 1e-10) -> np.ndarray:
    """
    Compute correlation matrix from vecl(log(C)) using Archakov-Hansen (2021) Corollary 1.
    
    Algorithm:
    1. Reconstruct symmetric L with zero diagonal from vecl^{-1}(gamma)
    2. Compute M = exp(L)
    3. Return C = D^{-1/2} * M * D^{-1/2} where D = diag(M)
    4. Optionally refine via fixed-point iteration
    
    Parameters:
    -----------
    gamma : np.ndarray
        Vector of length d = n(n-1)/2 representing vecl(log(C))
    n : int
        Dimension of correlation matrix
    max_iter : int
        Maximum iterations for refinement (default: 5)
    tol : float
        Tolerance for diagonal = 1 check (default: 1e-10)
        
    Returns:
    --------
    np.ndarray
        Correlation matrix C of shape (n, n) with unit diagonal
    """
    # Step 1: Reconstruct symmetric L with zero diagonal
    L = vecl_inverse(gamma, n)
    
    # Verify diagonal is zero
    assert np.allclose(np.diag(L), 0), "L must have zero diagonal"
    
    # Step 2-3: Iterative refinement for high accuracy
    for iteration in range(max_iter):
        # Compute M = exp(L)
        M = matrix_exp(L, symmetrize=True)
        
        # Extract diagonal
        d_M = np.diag(M)
        
        # Check for numerical issues
        if np.any(d_M <= 0):
            raise ValueError(f"Diagonal of exp(L) has non-positive elements: min = {np.min(d_M)}")
        
        # Compute C = D^{-1/2} * M * D^{-1/2}
        D_sqrt_inv = np.diag(1.0 / np.sqrt(d_M))
        C = D_sqrt_inv @ M @ D_sqrt_inv
        
        # Check diagonal
        diag_C = np.diag(C)
        diag_error = np.max(np.abs(diag_C - 1.0))
        
        if diag_error < tol:
            break
        
        # Refine L by adjusting diagonal of log
        if iteration < max_iter - 1:
            # L_corrected = L - diag(log(diag(M)))
            L = L - np.diag(np.log(d_M))
    
    # Final symmetrization
    C = (C + C.T) / 2
    
    # Force exact unit diagonal
    C = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    
    # Validate
    assert np.allclose(np.diag(C), 1.0, atol=1e-8), \
        f"Diagonal of C not unity: max error = {np.max(np.abs(np.diag(C) - 1.0))}"
    
    # Check positive definiteness
    eigenvalues = linalg.eigvalsh(C)
    if np.any(eigenvalues < -1e-10):
        raise ValueError(f"C is not positive definite: min eigenvalue = {np.min(eigenvalues)}")
    
    return C


def gamma_from_correlation(C: np.ndarray) -> np.ndarray:
    """
    Compute gamma = vecl(log(C)) from correlation matrix C.
    
    Parameters:
    -----------
    C : np.ndarray
        Correlation matrix (n, n) with unit diagonal
        
    Returns:
    --------
    np.ndarray
        Vector gamma of length d = n(n-1)/2
    """
    # Compute matrix logarithm
    L = matrix_log(C, symmetrize=True)
    
    # Extract vecl
    gamma = vecl(L)
    
    return gamma


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if matrix is positive definite.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Symmetric matrix
    tol : float
        Tolerance for smallest eigenvalue
        
    Returns:
    --------
    bool
        True if positive definite
    """
    try:
        eigenvalues = linalg.eigvalsh(matrix)
        return np.all(eigenvalues > -tol)
    except:
        return False


def ensure_positive_definite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Ensure matrix is positive definite via eigenvalue flooring.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Symmetric matrix
    epsilon : float
        Minimum eigenvalue as fraction of largest eigenvalue
        
    Returns:
    --------
    np.ndarray
        Positive definite matrix
    """
    eigenvalues, eigenvectors = linalg.eigh(matrix)
    
    # Floor negative eigenvalues
    max_eig = np.max(eigenvalues)
    min_eig_threshold = epsilon * max_eig
    eigenvalues_fixed = np.maximum(eigenvalues, min_eig_threshold)
    
    # Reconstruct
    matrix_fixed = eigenvectors @ np.diag(eigenvalues_fixed) @ eigenvectors.T
    
    # Symmetrize
    matrix_fixed = (matrix_fixed + matrix_fixed.T) / 2
    
    return matrix_fixed


def realized_variance_to_correlation(RM: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert realized covariance matrix to realized variance vector and correlation matrix.
    
    Parameters:
    -----------
    RM : np.ndarray
        Realized covariance matrix (n, n)
        
    Returns:
    --------
    x : np.ndarray
        Realized variance vector (diagonal of RM)
    Y : np.ndarray
        Realized correlation matrix
    """
    n = RM.shape[0]
    
    # Extract realized variances
    x = np.diag(RM)
    
    # Check for non-positive variances
    if np.any(x <= 0):
        raise ValueError(f"Non-positive realized variances detected: min = {np.min(x)}")
    
    # Compute correlation matrix
    Lambda_sqrt_inv = np.diag(1.0 / np.sqrt(x))
    Y = Lambda_sqrt_inv @ RM @ Lambda_sqrt_inv
    
    # Ensure exact unit diagonal and symmetry
    Y = (Y + Y.T) / 2
    np.fill_diagonal(Y, 1.0)
    
    # Ensure positive definiteness
    if not is_positive_definite(Y):
        Y = ensure_positive_definite(Y)
    
    return x, Y


def create_pair_ordering(n: int) -> List[Tuple[int, int]]:
    """
    Create the fixed ordering of pairs (p, q) for p > q.
    
    Parameters:
    -----------
    n : int
        Number of assets
        
    Returns:
    --------
    List[Tuple[int, int]]
        List of (p, q) pairs in column-major order
    """
    pairs = []
    for j in range(n):
        for i in range(j + 1, n):
            pairs.append((i, j))
    return pairs


def get_sector_blocks(n: int = 9) -> Dict[str, List[int]]:
    """
    Get the predefined sector block structure for 9 assets.
    
    Energy: indices 0, 1, 2 (CVX, MRO, OXY)
    Health Care: indices 3, 4, 5 (JNJ, LLY, MRK)
    IT: indices 6, 7, 8 (AAPL, MU, ORCL)
    
    Returns:
    --------
    Dict[str, List[int]]
        Dictionary mapping sector names to asset indices
    """
    assert n == 9, "Sector blocks are predefined for n=9 assets"
    
    return {
        'Energy': [0, 1, 2],
        'HealthCare': [3, 4, 5],
        'IT': [6, 7, 8]
    }


def create_block_loading_matrix(n: int = 9) -> Tuple[np.ndarray, List[str]]:
    """
    Create the factor loading matrix A for Block structure (r=6).
    
    Six pair categories:
    1. Within Energy (3 pairs)
    2. Within Health Care (3 pairs)
    3. Within IT (3 pairs)
    4. Energy-Health cross (9 pairs)
    5. Energy-IT cross (9 pairs)
    6. Health-IT cross (9 pairs)
    
    Parameters:
    -----------
    n : int
        Number of assets (must be 9)
        
    Returns:
    --------
    A : np.ndarray
        Binary matrix of shape (d, 6) where d = n(n-1)/2 = 36
    categories : List[str]
        Names of the 6 categories
    """
    assert n == 9, "Block structure is predefined for n=9 assets"
    
    d = n * (n - 1) // 2  # 36
    r = 6
    A = np.zeros((d, r))
    
    sectors = get_sector_blocks(n)
    energy_idx = sectors['Energy']
    health_idx = sectors['HealthCare']
    it_idx = sectors['IT']
    
    pairs = create_pair_ordering(n)
    
    categories = [
        'Within-Energy',
        'Within-Health',
        'Within-IT',
        'Energy-Health',
        'Energy-IT',
        'Health-IT'
    ]
    
    for pair_idx, (i, j) in enumerate(pairs):
        # Within Energy
        if i in energy_idx and j in energy_idx:
            A[pair_idx, 0] = 1
        # Within Health
        elif i in health_idx and j in health_idx:
            A[pair_idx, 1] = 1
        # Within IT
        elif i in it_idx and j in it_idx:
            A[pair_idx, 2] = 1
        # Energy-Health cross
        elif (i in energy_idx and j in health_idx) or (i in health_idx and j in energy_idx):
            A[pair_idx, 3] = 1
        # Energy-IT cross
        elif (i in energy_idx and j in it_idx) or (i in it_idx and j in energy_idx):
            A[pair_idx, 4] = 1
        # Health-IT cross
        elif (i in health_idx and j in it_idx) or (i in it_idx and j in health_idx):
            A[pair_idx, 5] = 1
    
    return A, categories


def create_equi_loading_matrix(n: int = 9) -> np.ndarray:
    """
    Create the factor loading matrix A for Equi structure (r=1).
    
    Parameters:
    -----------
    n : int
        Number of assets
        
    Returns:
    --------
    np.ndarray
        Column vector of ones with shape (d, 1) where d = n(n-1)/2
    """
    d = n * (n - 1) // 2
    return np.ones((d, 1))


def compute_check_y(y_t: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute factor projections check_y_t = A' * y_t / diag(A' * A).
    
    For Equi: check_y_t = mean(y_t)
    For Block: check_y_{k,t} = mean of y_{j,t} for j in category k
    For Full: check_y_t = y_t
    
    Parameters:
    -----------
    y_t : np.ndarray
        Transformed realized correlations, shape (d,) or (T, d)
    A : np.ndarray
        Factor loading matrix, shape (d, r)
        
    Returns:
    --------
    np.ndarray
        Factor projections, shape (r,) or (T, r)
    """
    # Compute A' * A diagonal
    AtA_diag = np.sum(A, axis=0)  # For binary A, this is the count per factor
    
    if y_t.ndim == 1:
        # Single observation
        check_y = A.T @ y_t / AtA_diag
    else:
        # Multiple observations (T, d)
        check_y = (y_t @ A) / AtA_diag[None, :]
    
    return check_y
