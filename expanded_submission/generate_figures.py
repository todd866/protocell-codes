#!/usr/bin/env python3
"""
Generate figures from overnight simulation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

def load_results():
    """Load all result JSON files."""
    results_dir = Path('results')

    with open(results_dir / 'scaling_experiment.json') as f:
        scaling = json.load(f)

    with open(results_dir / 'basin_robustness.json') as f:
        basin = json.load(f)

    with open(results_dir / 'padic_analysis.json') as f:
        padic = json.load(f)

    with open(results_dir / 'parameter_sweep.json') as f:
        params = json.load(f)

    return scaling, basin, padic, params


def fig_scaling(scaling):
    """Figure: Lewis game scaling to 64 states."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    n_states = [s['n_states'] for s in scaling]
    coord = [s['mean_coordination'] * 100 for s in scaling]
    coord_std = [s['std_coordination'] * 100 for s in scaling]

    # Calculate code coverage
    coverage = []
    for s in scaling:
        mean_codes = np.mean([sr['unique_codes'] for sr in s['seed_results']])
        coverage.append(mean_codes / s['n_states'] * 100)

    # Panel A: Coordination accuracy
    ax1.errorbar(n_states, coord, yerr=coord_std, marker='o', capsize=5,
                 linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Number of States')
    ax1.set_ylabel('Coordination Accuracy (%)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(n_states)
    ax1.set_xticklabels(n_states)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=1/64*100, color='gray', linestyle='--', alpha=0.5, label='Random (1/64)')
    ax1.set_title('A. Coordination Accuracy')

    # Panel B: Code coverage
    ax2.plot(n_states, coverage, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Number of States')
    ax2.set_ylabel('Code Coverage (%)')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(n_states)
    ax2.set_xticklabels(n_states)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=33, color='gray', linestyle='--', alpha=0.5, label='Genetic code (21/64)')
    ax2.set_title('B. Code Coverage (Distinct Codes Used)')
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('figures/fig_lewis_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_lewis_scaling.png', dpi=300, bbox_inches='tight')
    print('Saved: figures/fig_lewis_scaling.pdf')
    plt.close()


def fig_basin_robustness(basin):
    """Figure: Basin size vs robustness correlation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Example scatter from one seed
    example = basin['all_correlations'][7]  # seed with r=0.91
    sizes = list(example['basin_sizes'].values())
    robust = list(example['robustness'].values())

    ax1.scatter(np.array(sizes)*100, np.array(robust)*100, s=80, alpha=0.7,
                color='#2E86AB', edgecolor='white', linewidth=1)

    # Fit line
    z = np.polyfit(sizes, robust, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sizes), max(sizes), 100)
    ax1.plot(x_line*100, p(x_line)*100, 'r--', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Basin Size (%)')
    ax1.set_ylabel('Robustness (%)')
    ax1.set_title(f'A. Example: r = {example["pearson_r"]:.2f}')

    # Panel B: Distribution of correlations across seeds
    r_values = [c['pearson_r'] for c in basin['all_correlations']]

    ax2.hist(r_values, bins=10, color='#A23B72', edgecolor='white', alpha=0.8)
    ax2.axvline(x=basin['mean_pearson_r'], color='red', linestyle='--',
                linewidth=2, label=f'Mean r = {basin["mean_pearson_r"]:.2f}')
    ax2.set_xlabel('Pearson r')
    ax2.set_ylabel('Count')
    ax2.set_title('B. Correlation Distribution (n=20 seeds)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('figures/fig_basin_robustness.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_basin_robustness.png', dpi=300, bbox_inches='tight')
    print('Saved: figures/fig_basin_robustness.pdf')
    plt.close()


def fig_padic(padic):
    """Figure: P-adic distance predicts synonymy."""
    fig, ax = plt.subplots(figsize=(6, 4))

    distances = padic['distance_analysis']

    # Filter to meaningful distances
    dist_labels = [d['distance_str'] for d in distances[:6]]
    error_rates = [d['error_rate'] * 100 for d in distances[:6]]

    bars = ax.bar(dist_labels, error_rates, color='#2E86AB', edgecolor='white')

    # Color the first bar differently (closest = lowest error)
    bars[0].set_color('#A23B72')

    ax.set_xlabel('P-adic Distance')
    ax.set_ylabel('Coding Error Rate (%)')
    ax.set_title('P-adic Distance Predicts Synonymy')

    # Add annotation for closest pairs
    ax.annotate(f'{padic["closest_pairs"]["same_amino_acid"]}/{padic["closest_pairs"]["total"]}\nsame AA',
                xy=(0, error_rates[0]), xytext=(0.5, 25),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('figures/fig_padic.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_padic.png', dpi=300, bbox_inches='tight')
    print('Saved: figures/fig_padic.pdf')
    plt.close()


def fig_combined_validation(scaling, basin, padic):
    """Combined validation figure for paper."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Lewis scaling - coordination
    ax = axes[0, 0]
    n_states = [s['n_states'] for s in scaling]
    coord = [s['mean_coordination'] * 100 for s in scaling]
    coord_std = [s['std_coordination'] * 100 for s in scaling]
    ax.errorbar(n_states, coord, yerr=coord_std, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Coordination (%)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_states)
    ax.set_xticklabels(n_states)
    ax.set_title('A. Lewis Game Scaling')

    # Panel B: Lewis scaling - code coverage
    ax = axes[0, 1]
    coverage = []
    for s in scaling:
        mean_codes = np.mean([sr['unique_codes'] for sr in s['seed_results']])
        coverage.append(mean_codes / s['n_states'] * 100)
    ax.plot(n_states, coverage, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Code Coverage (%)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_states)
    ax.set_xticklabels(n_states)
    ax.axhline(y=33, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('Genetic code\n(21/64 = 33%)', xy=(48, 33), fontsize=9, color='gray')
    ax.set_title('B. Code Coverage')

    # Panel C: Basin-robustness
    ax = axes[1, 0]
    example = basin['all_correlations'][7]
    sizes = np.array(list(example['basin_sizes'].values())) * 100
    robust = np.array(list(example['robustness'].values())) * 100
    ax.scatter(sizes, robust, s=80, alpha=0.7, color='#2E86AB', edgecolor='white')
    z = np.polyfit(sizes/100, robust/100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sizes)/100, max(sizes)/100, 100)
    ax.plot(x_line*100, p(x_line)*100, 'r--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Basin Size (%)')
    ax.set_ylabel('Robustness (%)')
    ax.set_title(f'C. Basin-Robustness (r = {basin["mean_pearson_r"]:.2f} ± {basin["std_pearson_r"]:.2f})')

    # Panel D: P-adic
    ax = axes[1, 1]
    distances = padic['distance_analysis']
    dist_labels = [d['distance_str'] for d in distances[:6]]
    error_rates = [d['error_rate'] * 100 for d in distances[:6]]
    bars = ax.bar(dist_labels, error_rates, color='#2E86AB', edgecolor='white')
    bars[0].set_color('#A23B72')
    ax.set_xlabel('P-adic Distance')
    ax.set_ylabel('Coding Error (%)')
    ax.set_title(f'D. P-adic Validation ({padic["closest_pairs"]["same_amino_acid"]}/32 same AA)')

    plt.tight_layout()
    plt.savefig('figures/fig_validation_combined.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_validation_combined.png', dpi=300, bbox_inches='tight')
    print('Saved: figures/fig_validation_combined.pdf')
    plt.close()


if __name__ == '__main__':
    print('Loading results...')
    scaling, basin, padic, params = load_results()

    print('\nGenerating figures...')
    Path('figures').mkdir(exist_ok=True)

    fig_scaling(scaling)
    fig_basin_robustness(basin)
    fig_padic(padic)
    fig_combined_validation(scaling, basin, padic)

    print('\nDone! Generated figures in figures/')
    print('\nKey stats for paper:')
    print(f'  Basin-robustness: r = {basin["mean_pearson_r"]:.3f} ± {basin["std_pearson_r"]:.3f}')
    print(f'  P-adic closest pairs: {padic["closest_pairs"]["same_amino_acid"]}/{padic["closest_pairs"]["total"]} same AA')
    print(f'  64-state codes used: {np.mean([s["unique_codes"] for s in scaling[-1]["seed_results"]]):.0f}/64')
