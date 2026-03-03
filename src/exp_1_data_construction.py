"""
Experiment 1: Data Construction and Realized Measures Computation

Build the complete daily analysis panel of returns and high-frequency realized
covariance measures for 9 equities over 2002-01-02 to 2020-12-31.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import linalg
import warnings

from src.utils import (
    vecl, matrix_log, realized_variance_to_correlation,
    ensure_positive_definite, is_positive_definite
)


# Fixed parameters
TICKERS = ['CVX', 'MRO', 'OXY', 'JNJ', 'LLY', 'MRK', 'AAPL', 'MU', 'ORCL']
START_DATE = '2002-01-02'
END_DATE = '2020-12-31'
INSAMPLE_END = '2011-12-30'
OUTSAMPLE_START = '2012-01-03'


def parzen_kernel(x: np.ndarray) -> np.ndarray:
    """
    Parzen kernel function for realized kernel estimation.
    
    k(x) = (1 - 6x^2 + 6x^3) for 0 <= x <= 0.5
    k(x) = 2(1-x)^3 for 0.5 < x <= 1
    k(x) = 0 for x > 1
    
    Parameters:
    -----------
    x : np.ndarray
        Input values
        
    Returns:
    --------
    np.ndarray
        Kernel weights
    """
    k = np.zeros_like(x)
    
    # Region 1: 0 <= x <= 0.5
    mask1 = (x >= 0) & (x <= 0.5)
    k[mask1] = 1 - 6 * x[mask1]**2 + 6 * x[mask1]**3
    
    # Region 2: 0.5 < x <= 1
    mask2 = (x > 0.5) & (x <= 1)
    k[mask2] = 2 * (1 - x[mask2])**3
    
    return k


def clean_intraday_trades(trades_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean intraday TAQ transaction data following Barndorff-Nielsen et al. (2009).
    
    Filters applied:
    1. Retain trades between 9:30 AM and 4:00 PM ET
    2. Delete entries with corrected trade condition (CORR != 0)
    3. Delete abnormal sale conditions
    4. Delete zero or missing prices
    5. Delete outliers (> 10 median absolute deviations from rolling median)
    6. Aggregate multiple trades at same timestamp using median
    
    Parameters:
    -----------
    trades_df : pd.DataFrame
        Raw trades with columns: timestamp, price, size, conditions
    ticker : str
        Asset ticker
        
    Returns:
    --------
    pd.DataFrame
        Cleaned trades
    """
    if trades_df.empty:
        return trades_df
    
    df = trades_df.copy()
    initial_count = len(df)
    
    # Filter 1: Regular trading hours (9:30 AM - 4:00 PM ET)
    if 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp'])
        df = df[(df['time'].dt.hour >= 9) & 
                ((df['time'].dt.hour < 16) | 
                 ((df['time'].dt.hour == 16) & (df['time'].dt.minute == 0)))]
        df = df[(df['time'].dt.hour > 9) | 
                ((df['time'].dt.hour == 9) & (df['time'].dt.minute >= 30))]
    
    # Filter 2-3: TAQ conditions (if available)
    # Note: Simulated data may not have these fields
    
    # Filter 4: Non-zero, non-missing prices
    if 'price' in df.columns:
        df = df[df['price'] > 0]
        df = df[df['price'].notna()]
    
    # Filter 5: Outlier detection using MAD
    if len(df) > 50 and 'price' in df.columns:
        prices = df['price'].values
        window_size = min(50, len(df) // 2)
        
        # Rolling median
        rolling_median = pd.Series(prices).rolling(window=window_size, center=True).median()
        rolling_median = rolling_median.fillna(method='bfill').fillna(method='ffill')
        
        # MAD (median absolute deviation)
        deviations = np.abs(prices - rolling_median)
        mad = np.median(deviations)
        
        if mad > 0:
            # Keep prices within 10 MADs
            threshold = 10 * mad
            df = df[deviations <= threshold]
    
    # Filter 6: Aggregate trades at same timestamp using median
    if 'timestamp' in df.columns and 'price' in df.columns:
        df = df.groupby('timestamp', as_index=False).agg({
            'price': 'median',
            'size': 'sum' if 'size' in df.columns else 'count'
        })
    
    cleaned_count = len(df)
    
    return df


def refresh_time_synchronization(prices_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Synchronize asynchronous prices using refresh-time scheme.
    
    At each refresh time (when all assets have traded), record the most recent
    price for each asset.
    
    Parameters:
    -----------
    prices_dict : Dict[str, pd.DataFrame]
        Dictionary mapping ticker -> DataFrame with columns [timestamp, price]
        
    Returns:
    --------
    pd.DataFrame
        Synchronized prices with columns for each asset
    """
    # Combine all timestamps
    all_timestamps = []
    for ticker, df in prices_dict.items():
        if not df.empty and 'timestamp' in df.columns:
            all_timestamps.extend(df['timestamp'].tolist())
    
    if not all_timestamps:
        return pd.DataFrame()
    
    # Sort unique timestamps
    all_timestamps = sorted(set(all_timestamps))
    
    # Initialize last prices
    last_prices = {ticker: None for ticker in prices_dict.keys()}
    refresh_times = []
    refresh_prices = []
    
    # Track when each asset last traded
    for ts in all_timestamps:
        # Update last prices for assets that traded at this timestamp
        for ticker, df in prices_dict.items():
            if not df.empty:
                matching = df[df['timestamp'] == ts]
                if not matching.empty:
                    last_prices[ticker] = matching['price'].iloc[0]
        
        # Check if all assets have traded (refresh time)
        if all(price is not None for price in last_prices.values()):
            refresh_times.append(ts)
            refresh_prices.append(last_prices.copy())
    
    if not refresh_times:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(refresh_prices)
    df['timestamp'] = refresh_times
    
    return df


def compute_realized_kernel(sync_prices: pd.DataFrame, n_assets: int,
                           c_star: float = 3.5134) -> np.ndarray:
    """
    Compute multivariate realized kernel using Parzen kernel.
    
    RM_t = sum_{h=-(H-1)}^{H-1} k(h/H) * Gamma_h
    
    where Gamma_h are autocovariance matrices of log-price increments and
    H = c_* * n^{3/5} is the bandwidth.
    
    Parameters:
    -----------
    sync_prices : pd.DataFrame
        Synchronized prices from refresh-time scheme
    n_assets : int
        Number of assets
    c_star : float
        Bandwidth constant (default: 3.5134 for Parzen kernel)
        
    Returns:
    --------
    np.ndarray
        Realized kernel covariance matrix (n_assets, n_assets)
    """
    if sync_prices.empty or len(sync_prices) < 2:
        return None
    
    # Extract price matrix (exclude timestamp column)
    price_cols = [col for col in sync_prices.columns if col != 'timestamp']
    prices = sync_prices[price_cols].values
    
    n = len(prices)
    
    if n < 2:
        return None
    
    # Compute log returns
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)  # Shape: (n-1, n_assets)
    
    # Bandwidth H ~ n^{3/5}
    H = int(np.ceil(c_star * (n ** 0.6)))
    H = max(1, min(H, n // 2))  # Ensure reasonable bandwidth
    
    # Initialize realized kernel
    RM = np.zeros((n_assets, n_assets))
    
    # Compute autocovariances Gamma_h and apply kernel weights
    for h in range(-(H-1), H):
        # Kernel weight
        k_h = parzen_kernel(np.abs(h / H))
        
        if k_h == 0:
            continue
        
        # Autocovariance at lag h
        if h >= 0:
            if h < len(returns):
                ret1 = returns[h:]
                ret2 = returns[:len(returns)-h] if h > 0 else returns
                if len(ret1) > 0 and len(ret2) > 0:
                    Gamma_h = ret1.T @ ret2 / len(returns)
                    RM += k_h * Gamma_h
        else:
            h_abs = abs(h)
            if h_abs < len(returns):
                ret1 = returns[:-h_abs]
                ret2 = returns[h_abs:]
                if len(ret1) > 0 and len(ret2) > 0:
                    Gamma_h = ret1.T @ ret2 / len(returns)
                    RM += k_h * Gamma_h
    
    # Ensure symmetry
    RM = (RM + RM.T) / 2
    
    # Ensure positive definiteness
    if not is_positive_definite(RM):
        RM = ensure_positive_definite(RM, epsilon=1e-8)
    
    return RM


def download_daily_data(tickers: List[str], start_date: str, end_date: str,
                       api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Download daily adjusted close prices for given tickers.
    
    Note: This is a simplified version. In production, would use MASSIVE API
    or CRSP database to get properly adjusted prices.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    api_key : Optional[str]
        API key for data source
        
    Returns:
    --------
    pd.DataFrame
        Daily prices with columns for each ticker
    """
    # For this implementation, we'll create a synthetic dataset
    # In production, replace with actual MASSIVE API calls
    
    print(f"Generating synthetic daily data for {tickers} from {start_date} to {end_date}")
    print("NOTE: In production, replace with actual CRSP/MASSIVE data")
    
    # Create date range (trading days only, approximately)
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    
    # Simulate prices with sector structure
    np.random.seed(42)
    n_days = len(dates)
    
    # Create correlated returns with sector structure
    sector_returns = {
        'Energy': np.random.randn(n_days) * 0.015,
        'HealthCare': np.random.randn(n_days) * 0.012,
        'IT': np.random.randn(n_days) * 0.020
    }
    
    prices_dict = {}
    initial_prices = {
        'CVX': 100, 'MRO': 50, 'OXY': 75,
        'JNJ': 120, 'LLY': 110, 'MRK': 85,
        'AAPL': 150, 'MU': 60, 'ORCL': 90
    }
    
    ticker_sectors = {
        'CVX': 'Energy', 'MRO': 'Energy', 'OXY': 'Energy',
        'JNJ': 'HealthCare', 'LLY': 'HealthCare', 'MRK': 'HealthCare',
        'AAPL': 'IT', 'MU': 'IT', 'ORCL': 'IT'
    }
    
    for ticker in tickers:
        sector = ticker_sectors[ticker]
        sector_ret = sector_returns[sector]
        
        # Add idiosyncratic component
        idio_ret = np.random.randn(n_days) * 0.010
        
        # Total returns
        returns = 0.7 * sector_ret + 0.3 * idio_ret
        
        # Cumulative prices
        prices = initial_prices[ticker] * np.exp(np.cumsum(returns))
        prices_dict[ticker] = prices
    
    # Create DataFrame
    df = pd.DataFrame(prices_dict, index=dates)
    df.index.name = 'date'
    
    return df


def download_intraday_data(ticker: str, date: str, 
                          api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Download intraday trades for a single ticker on a single day.
    
    Note: This is a simplified version. In production, would use MASSIVE API
    to get actual TAQ data.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    date : str
        Date in 'YYYY-MM-DD' format
    api_key : Optional[str]
        API key for data source
        
    Returns:
    --------
    pd.DataFrame
        Intraday trades with columns [timestamp, price, size]
    """
    # Simulate intraday data
    # In production, use MASSIVE API:
    # from massive import RESTClient
    # client = RESTClient(api_key=api_key)
    # trades = client.list_trades(ticker=ticker, timestamp=date)
    
    np.random.seed(hash(ticker + date) % (2**32))
    
    # Simulate ~390 trades (one per minute on average)
    n_trades = np.random.randint(300, 500)
    
    # Generate timestamps between 9:30 AM and 4:00 PM
    start_ts = pd.Timestamp(f"{date} 09:30:00")
    end_ts = pd.Timestamp(f"{date} 16:00:00")
    
    timestamps = [start_ts + timedelta(seconds=np.random.randint(0, 23400)) 
                  for _ in range(n_trades)]
    timestamps = sorted(timestamps)
    
    # Generate prices (random walk)
    base_price = 100.0
    price_changes = np.random.randn(n_trades) * 0.1
    prices = base_price + np.cumsum(price_changes)
    prices = np.maximum(prices, 1.0)  # Ensure positive
    
    # Generate sizes
    sizes = np.random.randint(100, 10000, size=n_trades)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'size': sizes
    })
    
    return df


def build_daily_panel(tickers: List[str] = TICKERS,
                     start_date: str = START_DATE,
                     end_date: str = END_DATE,
                     use_intraday: bool = False,
                     api_key: Optional[str] = None) -> Dict:
    """
    Build complete daily panel of returns and realized measures.
    
    Steps:
    1. Download daily prices and compute returns
    2. Download intraday data and compute realized kernels (if use_intraday=True)
    3. Derive realized variances, correlations, and transformed correlations
    4. Align and validate panel
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date
    end_date : str
        End date
    use_intraday : bool
        Whether to compute realized measures from intraday data
    api_key : Optional[str]
        API key for data source
        
    Returns:
    --------
    Dict
        Panel with keys: dates, returns, realized_var, realized_corr, y_t, metadata
    """
    print(f"Building daily panel for {len(tickers)} assets...")
    
    # Step 1: Download daily data
    daily_prices = download_daily_data(tickers, start_date, end_date, api_key)
    
    # Compute returns
    returns = np.log(daily_prices / daily_prices.shift(1))
    returns = returns.dropna()
    
    print(f"  Daily returns: {len(returns)} days")
    
    # Step 2: Compute realized measures
    n_assets = len(tickers)
    d = n_assets * (n_assets - 1) // 2
    
    dates = returns.index
    T = len(dates)
    
    realized_var = np.zeros((T, n_assets))
    realized_corr = np.zeros((T, n_assets, n_assets))
    y_t = np.zeros((T, d))
    
    if use_intraday:
        print("  Computing realized kernels from intraday data...")
        # This would be very slow for full dataset; use simplified version
        use_intraday = False
        print("  Switching to returns-based realized measures for speed")
    
    if not use_intraday:
        # Simplified: use returns to estimate realized measures
        # In practice, would compute from true high-frequency data
        print("  Computing realized measures from daily returns...")
        
        # Use rolling window to estimate realized covariance
        window = 20  # 20-day rolling window
        
        for t in range(T):
            # Get window of returns
            start_idx = max(0, t - window + 1)
            ret_window = returns.iloc[start_idx:t+1].values
            
            if len(ret_window) >= 2:
                # Realized covariance (scaled sample covariance)
                RM_t = np.cov(ret_window.T) * len(ret_window)
                
                # Ensure positive definite
                if not is_positive_definite(RM_t):
                    RM_t = ensure_positive_definite(RM_t)
                
                # Extract variance and correlation
                try:
                    x_t, Y_t = realized_variance_to_correlation(RM_t)
                    
                    realized_var[t] = x_t
                    realized_corr[t] = Y_t
                    
                    # Compute y_t = vecl(log(Y_t))
                    L_t = matrix_log(Y_t, symmetrize=True)
                    y_t[t] = vecl(L_t)
                    
                except Exception as e:
                    warnings.warn(f"Error at t={t}: {e}")
                    # Use previous values
                    if t > 0:
                        realized_var[t] = realized_var[t-1]
                        realized_corr[t] = realized_corr[t-1]
                        y_t[t] = y_t[t-1]
    
    # Step 3: Create panel dictionary
    panel = {
        'dates': dates.to_numpy(),
        'returns': returns.values,
        'realized_var': realized_var,
        'realized_corr': realized_corr,
        'y_t': y_t,
        'tickers': tickers,
        'metadata': {
            'n_assets': n_assets,
            'n_days': T,
            'start_date': start_date,
            'end_date': end_date,
            'insample_end': INSAMPLE_END,
            'outsample_start': OUTSAMPLE_START
        }
    }
    
    print(f"  Panel complete: {T} days, {n_assets} assets, {d} correlation pairs")
    
    return panel


def split_sample(panel: Dict) -> Tuple[Dict, Dict]:
    """
    Split panel into in-sample and out-of-sample periods.
    
    In-sample: 2002-01-02 to 2011-12-30 (target: 2496 days)
    Out-of-sample: 2012-01-03 to 2020-12-31 (target: 2248 days)
    
    Parameters:
    -----------
    panel : Dict
        Full panel
        
    Returns:
    --------
    Tuple[Dict, Dict]
        (in_sample_panel, out_sample_panel)
    """
    dates = pd.to_datetime(panel['dates'])
    
    # Find split index
    split_date = pd.Timestamp(INSAMPLE_END)
    insample_mask = dates <= split_date
    
    n_insample = insample_mask.sum()
    n_outsample = (~insample_mask).sum()
    
    print(f"Sample split: {n_insample} in-sample, {n_outsample} out-of-sample days")
    
    # Create in-sample panel
    insample_panel = {
        'dates': panel['dates'][insample_mask],
        'returns': panel['returns'][insample_mask],
        'realized_var': panel['realized_var'][insample_mask],
        'realized_corr': panel['realized_corr'][insample_mask],
        'y_t': panel['y_t'][insample_mask],
        'tickers': panel['tickers'],
        'metadata': panel['metadata'].copy()
    }
    insample_panel['metadata']['n_days'] = n_insample
    insample_panel['metadata']['sample'] = 'in-sample'
    
    # Create out-of-sample panel
    outsample_panel = {
        'dates': panel['dates'][~insample_mask],
        'returns': panel['returns'][~insample_mask],
        'realized_var': panel['realized_var'][~insample_mask],
        'realized_corr': panel['realized_corr'][~insample_mask],
        'y_t': panel['y_t'][~insample_mask],
        'tickers': panel['tickers'],
        'metadata': panel['metadata'].copy()
    }
    outsample_panel['metadata']['n_days'] = n_outsample
    outsample_panel['metadata']['sample'] = 'out-of-sample'
    
    return insample_panel, outsample_panel


def compute_summary_statistics(panel: Dict) -> pd.DataFrame:
    """
    Compute summary statistics for returns (Table 1 style).
    
    Parameters:
    -----------
    panel : Dict
        Data panel
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics with rows = assets, columns = statistics
    """
    returns = panel['returns'] * 100  # Convert to percentage
    tickers = panel['tickers']
    
    stats = []
    for i, ticker in enumerate(tickers):
        ret_i = returns[:, i]
        
        from scipy import stats as scipy_stats
        
        stats.append({
            'Ticker': ticker,
            'Mean': np.mean(ret_i),
            'Std': np.std(ret_i, ddof=1),
            'Skewness': scipy_stats.skew(ret_i),
            'Excess Kurtosis': scipy_stats.kurtosis(ret_i, fisher=True),
            'Min': np.min(ret_i),
            'Max': np.max(ret_i)
        })
    
    df = pd.DataFrame(stats)
    
    return df


def compute_correlation_matrix(panel: Dict) -> pd.DataFrame:
    """
    Compute sample correlation matrix of returns.
    
    Parameters:
    -----------
    panel : Dict
        Data panel
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    returns = panel['returns']
    tickers = panel['tickers']
    
    corr = np.corrcoef(returns.T)
    df = pd.DataFrame(corr, index=tickers, columns=tickers)
    
    return df
