#!/usr/bin/env python3
"""
Parameter sweep for predator-prey simulation.
Maps the viable parameter space for two-attractor ecology.
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from overnight_run import PARAMS, initialize, step, classify_population

def run_sweep(param_name, values, generations=5000, seed=42):
    """Sweep a single parameter and measure outcomes."""
    results = []

    for val in values:
        params = PARAMS.copy()
        params[param_name] = val
        params['generations'] = generations

        rng = default_rng(seed)
        N, C, R = initialize(params, rng)

        for gen in range(generations):
            N, C, R = step(N, C, R, params, rng)

        is_pred = classify_population(N, C)
        n_pred = np.sum(is_pred)
        n_prey = np.sum(~is_pred)

        results.append({
            'value': val,
            'n_pred': n_pred,
            'n_prey': n_prey,
            'mean_N_pred': np.mean(N[is_pred]) if np.any(is_pred) else 0,
            'mean_N_prey': np.mean(N[~is_pred]) if np.any(~is_pred) else 0,
            'mean_C_pred': np.mean(C[is_pred]) if np.any(is_pred) else 0,
            'mean_C_prey': np.mean(C[~is_pred]) if np.any(~is_pred) else 0,
            'size_diff': (np.mean(N[~is_pred]) if np.any(~is_pred) else 0) -
                        (np.mean(N[is_pred]) if np.any(is_pred) else 0),
            'pred_fraction': n_pred / len(N) if len(N) > 0 else 0,
        })

    return results

def plot_sweep(results, param_name, output_dir):
    """Plot sweep results."""
    values = [r['value'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Size differentiation
    axes[0, 0].plot(values, [r['mean_N_pred'] for r in results], 'b-o', label='Predator N')
    axes[0, 0].plot(values, [r['mean_N_prey'] for r in results], 'r-o', label='Prey N')
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('Mean Size')
    axes[0, 0].set_title('Size by Type')
    axes[0, 0].legend()

    # Size difference
    axes[0, 1].plot(values, [r['size_diff'] for r in results], 'g-o')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('Size Difference (Prey - Pred)')
    axes[0, 1].set_title('Size Differentiation')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Coherence
    axes[1, 0].plot(values, [r['mean_C_pred'] for r in results], 'b-o', label='Predator C')
    axes[1, 0].plot(values, [r['mean_C_prey'] for r in results], 'r-o', label='Prey C')
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Mean Coherence')
    axes[1, 0].set_title('Coherence by Type')
    axes[1, 0].legend()

    # Population fractions
    axes[1, 1].plot(values, [r['pred_fraction'] * 100 for r in results], 'b-o', label='Predator %')
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel('Predator %')
    axes[1, 1].set_title('Predator Fraction')

    plt.tight_layout()
    plt.savefig(output_dir / f'sweep_{param_name}.png', dpi=150)
    plt.close()

def main():
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'results' / f'sweep_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")

    # Parameter sweeps
    sweeps = {
        'N_ceiling': np.array([20, 30, 40, 50, 60, 80, 100]),
        'encounter_rate': np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2]),
        'f': np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3]),
        'kappa_C': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'C_star': np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
    }

    for param_name, values in sweeps.items():
        print(f"\nSweeping {param_name}: {values}")
        results = run_sweep(param_name, values)

        # Print results
        print(f"  {'Value':>8s}  {'#Pred':>6s}  {'N_pred':>7s}  {'N_prey':>7s}  {'Diff':>7s}  {'Pred%':>6s}")
        for r in results:
            print(f"  {r['value']:8.3f}  {r['n_pred']:6d}  {r['mean_N_pred']:7.1f}  {r['mean_N_prey']:7.1f}  {r['size_diff']:+7.1f}  {r['pred_fraction']*100:6.2f}")

        # Plot
        plot_sweep(results, param_name, output_dir)

    print(f"\nFigures saved to {output_dir}")

if __name__ == '__main__':
    main()
