"""
Experiment 2: Univariate Realized GARCH Estimation

Estimate the Realized GARCH model independently for each asset.

Model specification:
- Return equation: r_{i,t} = mu_i + sqrt(h_{i,t}) * z_{i,t}
- Log-GARCH equation: log(h_{i,t}) = omega_i + beta_i*log(h_{i,t-1}) + tau_i(z_{i,t-1}) + alpha_i*log(x_{i,t-1})
- Measurement equation: log(x_{i,t}) = xi_i + phi_i*log(h_{i,t}) + delta_i(z_{i,t}) + v_{i,t}

where tau_i(z) = tau_{i,1}*z + tau_{i,2}*(z^2-1)
      delta_i(z) = delta_{i,1}*z + delta_{i,2}*(z^2-1)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy.optimize import minimize
from scipy import linalg
import warnings


def leverage_function(z: np.ndarray, tau_1: float, tau_2: float) -> np.ndarray:
    """
    Leverage function tau(z) = tau_1 * z + tau_2 * (z^2 - 1).
    
    Parameters:
    -----------
    z : np.ndarray
        Standardized residuals
    tau_1 : float
        Linear coefficient
    tau_2 : float
        Quadratic coefficient
        
    Returns:
    --------
    np.ndarray
        Leverage effect contribution
    """
    return tau_1 * z + tau_2 * (z**2 - 1)


def measurement_function(z: np.ndarray, delta_1: float, delta_2: float) -> np.ndarray:
    """
    Measurement equation leverage function delta(z) = delta_1 * z + delta_2 * (z^2 - 1).
    
    Parameters:
    -----------
    z : np.ndarray
        Standardized residuals
    delta_1 : float
        Linear coefficient
    delta_2 : float
        Quadratic coefficient
        
    Returns:
    --------
    np.ndarray
        Measurement leverage contribution
    """
    return delta_1 * z + delta_2 * (z**2 - 1)


def filter_realized_garch(params: np.ndarray, returns: np.ndarray, 
                         realized_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter conditional variances and compute standardized residuals.
    
    Parameters:
    -----------
    params : np.ndarray
        Parameters [mu, omega, beta, tau_1, tau_2, alpha, xi, phi, delta_1, delta_2]
    returns : np.ndarray
        Return series (T,)
    realized_var : np.ndarray
        Realized variance series (T,)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (log_h, z, v) - log conditional variance, standardized residuals, measurement errors
    """
    T = len(returns)
    
    mu, omega, beta, tau_1, tau_2, alpha, xi, phi, delta_1, delta_2 = params
    
    # Initialize
    log_h = np.zeros(T)
    z = np.zeros(T)
    v = np.zeros(T)
    
    # Initialize log_h[0] with unconditional mean of log(realized_var)
    log_rv_mean = np.mean(np.log(np.maximum(realized_var, 1e-8)))
    log_h[0] = log_rv_mean
    
    for t in range(T):
        # Compute z_t
        h_t = np.exp(log_h[t])
        z[t] = (returns[t] - mu) / np.sqrt(h_t)
        
        # Compute measurement error v_t
        log_rv_t = np.log(np.maximum(realized_var[t], 1e-8))
        v[t] = log_rv_t - xi - phi * log_h[t] - measurement_function(z[t], delta_1, delta_2)
        
        # Update log_h for next period
        if t < T - 1:
            log_h[t+1] = (omega + beta * log_h[t] + 
                         leverage_function(z[t], tau_1, tau_2) + 
                         alpha * log_rv_t)
    
    return log_h, z, v


def profiled_likelihood_realized_garch(params: np.ndarray, returns: np.ndarray,
                                      realized_var: np.ndarray) -> float:
    """
    Compute profiled Gaussian QML likelihood.
    
    The variance parameter sigma_v^2 is profiled out analytically.
    
    Parameters:
    -----------
    params : np.ndarray
        Parameters [mu, omega, beta, tau_1, tau_2, alpha, xi, phi, delta_1, delta_2]
    returns : np.ndarray
        Return series (T,)
    realized_var : np.ndarray
        Realized variance series (T,)
        
    Returns:
    --------
    float
        Negative profiled log-likelihood
    """
    T = len(returns)
    
    try:
        # Filter
        log_h, z, v = filter_realized_garch(params, returns, realized_var)
        
        # Check for invalid values
        if np.any(~np.isfinite(log_h)) or np.any(~np.isfinite(z)) or np.any(~np.isfinite(v)):
            return 1e10
        
        if np.any(np.abs(z) > 50):  # Extreme standardized residuals
            return 1e10
        
        # Profiled sigma_v^2
        sigma_v_sq = np.mean(v**2)
        
        if sigma_v_sq <= 0:
            return 1e10
        
        # Log-likelihood (profiled)
        loglik_return = -0.5 * np.sum(log_h + z**2)
        loglik_measurement = -0.5 * T * np.log(sigma_v_sq)
        loglik = loglik_return + loglik_measurement
        
        # Return negative for minimization
        return -loglik
        
    except Exception as e:
        return 1e10


def estimate_realized_garch_single(returns: np.ndarray, realized_var: np.ndarray,
                                  n_starts: int = 5) -> Dict:
    """
    Estimate univariate Realized GARCH model for a single asset.
    
    Parameters:
    -----------
    returns : np.ndarray
        Return series (T,)
    realized_var : np.ndarray
        Realized variance series (T,)
    n_starts : int
        Number of random starting values
        
    Returns:
    --------
    Dict
        Estimation results with keys: params, log_h, z, loglik, sigma_v_sq
    """
    T = len(returns)
    
    # Initial parameter values
    mu_init = np.mean(returns)
    
    log_rv = np.log(np.maximum(realized_var, 1e-8))
    log_rv_mean = np.mean(log_rv)
    
    # Starting values
    best_result = None
    best_loglik = -np.inf
    
    for start_idx in range(n_starts):
        if start_idx == 0:
            # Deterministic starting values
            params_init = np.array([
                mu_init,           # mu
                log_rv_mean * 0.1, # omega
                0.9,               # beta
                -0.1,              # tau_1 (leverage)
                0.05,              # tau_2
                0.05,              # alpha
                log_rv_mean * 0.1, # xi
                0.95,              # phi
                -0.05,             # delta_1
                0.02               # delta_2
            ])
        else:
            # Random perturbations
            params_init = np.array([
                mu_init + np.random.randn() * 0.001,
                log_rv_mean * 0.1 * (1 + 0.2 * np.random.randn()),
                0.8 + 0.15 * np.random.rand(),
                -0.2 * np.random.rand(),
                0.1 * np.random.rand(),
                0.1 * np.random.rand(),
                log_rv_mean * 0.1 * (1 + 0.2 * np.random.randn()),
                0.9 + 0.08 * np.random.rand(),
                -0.1 * np.random.rand(),
                0.05 * np.random.rand()
            ])
        
        # Parameter bounds
        bounds = [
            (None, None),      # mu
            (None, None),      # omega
            (0.01, 0.999),     # beta
            (None, None),      # tau_1
            (None, None),      # tau_2
            (0.0, 0.5),        # alpha
            (None, None),      # xi
            (0.5, 1.5),        # phi
            (None, None),      # delta_1
            (None, None)       # delta_2
        ]
        
        # Optimize
        try:
            result = minimize(
                profiled_likelihood_realized_garch,
                params_init,
                args=(returns, realized_var),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                loglik = -result.fun
                
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_result = result
        
        except Exception as e:
            warnings.warn(f"Optimization failed at start {start_idx}: {e}")
            continue
    
    if best_result is None:
        raise ValueError("All optimizations failed")
    
    # Extract results
    params = best_result.x
    log_h, z, v = filter_realized_garch(params, returns, realized_var)
    sigma_v_sq = np.mean(v**2)
    
    # Compute persistence
    beta = params[2]
    alpha = params[5]
    phi = params[7]
    persistence = beta + alpha * phi
    
    # Compute BIC
    n_params = 10
    bic = -2 * best_loglik + n_params * np.log(T)
    
    results = {
        'params': params,
        'param_names': ['mu', 'omega', 'beta', 'tau_1', 'tau_2', 
                       'alpha', 'xi', 'phi', 'delta_1', 'delta_2'],
        'log_h': log_h,
        'z': z,
        'v': v,
        'h': np.exp(log_h),
        'loglik': best_loglik,
        'sigma_v_sq': sigma_v_sq,
        'persistence': persistence,
        'bic': bic,
        'n_obs': T
    }
    
    return results


def estimate_univariate_realized_garch(panel: Dict, n_starts: int = 5) -> Dict:
    """
    Estimate Realized GARCH for all assets in the panel.
    
    Parameters:
    -----------
    panel : Dict
        Data panel with returns and realized_var
    n_starts : int
        Number of random starting values per asset
        
    Returns:
    --------
    Dict
        Results dictionary mapping asset index -> estimation results
    """
    returns = panel['returns']
    realized_var = panel['realized_var']
    tickers = panel['tickers']
    n_assets = len(tickers)
    
    print(f"Estimating univariate Realized GARCH for {n_assets} assets...")
    
    results = {}
    
    for i in range(n_assets):
        print(f"  Asset {i+1}/{n_assets} ({tickers[i]})...")
        
        ret_i = returns[:, i]
        rv_i = realized_var[:, i]
        
        try:
            res_i = estimate_realized_garch_single(ret_i, rv_i, n_starts=n_starts)
            results[i] = res_i
            
            print(f"    Persistence: {res_i['persistence']:.4f}, LogLik: {res_i['loglik']:.2f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            raise
    
    print("Univariate Realized GARCH estimation complete.")
    
    return results


def create_parameter_table(results: Dict, tickers: List[str]) -> pd.DataFrame:
    """
    Create parameter estimate table (Table 2/3 style).
    
    Parameters:
    -----------
    results : Dict
        Estimation results from estimate_univariate_realized_garch
    tickers : List[str]
        Asset tickers
        
    Returns:
    --------
    pd.DataFrame
        Parameter estimates table
    """
    rows = []
    
    for i, ticker in enumerate(tickers):
        res = results[i]
        params = res['params']
        
        row = {
            'Ticker': ticker,
            'mu': params[0],
            'omega': params[1],
            'beta': params[2],
            'tau_1': params[3],
            'tau_2': params[4],
            'alpha': params[5],
            'xi': params[6],
            'phi': params[7],
            'delta_1': params[8],
            'delta_2': params[9],
            'sigma_v_sq': res['sigma_v_sq'],
            'persistence': res['persistence'],
            'loglik': res['loglik'],
            'BIC': res['bic']
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df


def extract_standardized_residuals(results: Dict, n_assets: int, T: int) -> np.ndarray:
    """
    Extract standardized residuals matrix for all assets.
    
    Parameters:
    -----------
    results : Dict
        Estimation results
    n_assets : int
        Number of assets
    T : int
        Number of observations
        
    Returns:
    --------
    np.ndarray
        Standardized residuals (T, n_assets)
    """
    z_matrix = np.zeros((T, n_assets))
    
    for i in range(n_assets):
        z_matrix[:, i] = results[i]['z']
    
    return z_matrix


def extract_conditional_variances(results: Dict, n_assets: int, T: int) -> np.ndarray:
    """
    Extract conditional variances matrix for all assets.
    
    Parameters:
    -----------
    results : Dict
        Estimation results
    n_assets : int
        Number of assets
    T : int
        Number of observations
        
    Returns:
    --------
    np.ndarray
        Conditional variances (T, n_assets)
    """
    h_matrix = np.zeros((T, n_assets))
    
    for i in range(n_assets):
        h_matrix[:, i] = results[i]['h']
    
    return h_matrix
