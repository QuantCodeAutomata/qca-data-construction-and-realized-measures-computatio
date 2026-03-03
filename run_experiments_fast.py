"""
Fast version of experiment runner for demonstration (reduced iterations).
"""

import os
import sys
sys.path.insert(0, 'src')

from run_experiments import (
    run_exp_1, run_exp_6, generate_results_summary,
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR
)

# Import simplified versions
import numpy as np
from src.exp_2_realized_garch import create_parameter_table
from src.exp_3_correlation_models import estimate_ccc_equi, estimate_ccc_block, estimate_ccc_full, compute_loglik_improvements
from src.exp_2_realized_garch import extract_standardized_residuals

def run_exp_2_fast(insample_panel):
    """Fast version of Exp 2 with only 1 start."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Univariate Realized GARCH (FAST MODE - 1 start)")
    print("="*80)
    
    from src.exp_2_realized_garch import estimate_univariate_realized_garch
    
    # Use only 1 start for speed
    results = estimate_univariate_realized_garch(insample_panel, n_starts=1)
    
    # Create parameter table
    param_table = create_parameter_table(results, insample_panel['tickers'])
    print("\nParameter Estimates:")
    print(param_table[['Ticker', 'persistence', 'phi', 'sigma_v_sq']])
    
    # Save
    param_table.to_csv(TABLES_DIR / 'realized_garch_parameters.csv', index=False)
    
    print("\nExp 2 complete (fast mode).")
    
    return results


def run_exp_3_fast(insample_panel, stage1_results):
    """Fast version of Exp 3 with only CCC models."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Correlation Models (FAST MODE - CCC only)")
    print("="*80)
    
    T = insample_panel['metadata']['n_days']
    n = insample_panel['metadata']['n_assets']
    
    # Extract standardized residuals
    z_matrix = extract_standardized_residuals(stage1_results, n, T)
    
    results = {}
    
    # CCC+ models only (fast)
    print("  CCC+-Equi...")
    results['CCC+-Equi'] = estimate_ccc_equi(z_matrix)
    
    print("  CCC+-Block...")
    results['CCC+-Block'] = estimate_ccc_block(z_matrix, n)
    
    print("  CCC+-Full...")
    results['CCC+-Full'] = estimate_ccc_full(z_matrix)
    
    # Compute improvements
    improvements = compute_loglik_improvements(results)
    print("\nLog-Likelihood Improvements:")
    print(improvements)
    
    # Save
    improvements.to_csv(TABLES_DIR / 'loglik_improvements.csv', index=False)
    
    print("\nExp 3 complete (fast mode).")
    
    return results


def main():
    """Run experiments in fast mode."""
    print("\n" + "#"*80)
    print("# MULTIVARIATE REALIZED GARCH MODELS (FAST MODE)")
    print("# Implementation of Archakov & Hansen (2024)")
    print("#"*80)
    
    try:
        # Check if panel exists
        if not (RESULTS_DIR / 'panel_data.npz').exists():
            # Experiment 1: Data Construction
            panel, insample_panel, outsample_panel = run_exp_1()
        else:
            print("\nLoading existing panel data...")
            data = np.load(RESULTS_DIR / 'panel_data.npz', allow_pickle=True)
            
            from src.exp_1_data_construction import TICKERS
            
            panel = {
                'dates': data['dates'],
                'returns': data['returns'],
                'realized_var': data['realized_var'],
                'realized_corr': data['realized_corr'],
                'y_t': data['y_t'],
                'tickers': TICKERS,
                'metadata': {
                    'n_assets': 9,
                    'n_days': len(data['dates']),
                    'start_date': '2002-01-02',
                    'end_date': '2020-12-31'
                }
            }
            
            from src.exp_1_data_construction import split_sample
            insample_panel, outsample_panel = split_sample(panel)
            
            print(f"  Loaded panel: {panel['metadata']['n_days']} days")
        
        # Experiment 2: Univariate Realized GARCH (fast)
        stage1_results = run_exp_2_fast(insample_panel)
        
        # Experiment 3: Correlation Models (fast)
        correlation_results = run_exp_3_fast(insample_panel, stage1_results)
        
        # Experiment 6: Gaussianity Diagnostics
        diagnostics = run_exp_6(panel)
        
        # Generate comprehensive summary
        generate_results_summary()
        
        print("\n" + "#"*80)
        print("# ALL EXPERIMENTS COMPLETE (FAST MODE)")
        print("#"*80)
        print(f"\nResults saved to: {RESULTS_DIR.absolute()}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
