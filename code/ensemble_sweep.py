#!/usr/bin/env python3
"""
Ensemble sweep across random chemistries to test D_eff as predictor of code quality.

Generates:
1. Scatter plot: accuracy vs D_eff (colored by species count)
2. Regression analysis showing D_eff predicts accuracy better than species count
3. Ablation results for h and coupling parameters
"""

import numpy as np
import json
import time
from chemistry_sim import (
    AutocatalyticNetwork, ChemicalVesicleArray,
    hexagonal_grid, get_neighbors, create_spatial_gradient,
    participation_ratio
)
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy import stats
import matplotlib.pyplot as plt


def run_chemistry(n_species, n_reactions, n_outputs, seed, n_envs=16, n_trials=2,
                  n_rings=2, coupling_strength=0.1, hill_coeff=4.0, s0=1.0):
    """Run one chemistry configuration and return metrics."""
    array = ChemicalVesicleArray(
        n_rings=n_rings, n_species=n_species, n_reactions=n_reactions,
        n_outputs=n_outputs, seed=seed, coupling_strength=coupling_strength,
        hill_coeff=hill_coeff, s0=s0  # Now properly passed to substrate_competition
    )

    codes = []
    trial_codes_all = []
    for env_id in range(n_envs):
        env_inputs = create_spatial_gradient(array.coords, env_id)
        trial_codes = []
        for trial in range(n_trials):
            code, _ = array.run(env_inputs, t_span=(0, 50), n_points=200,
                               trial_seed=seed + env_id * 1000 + trial)
            trial_codes.append(code)
        codes.append(np.mean(trial_codes, axis=0))
        trial_codes_all.append(trial_codes)

    codes = np.array(codes)

    # Compute metrics
    d_eff = participation_ratio(codes)

    # Unique codes
    code_dists = squareform(pdist(codes))
    adjacency = (code_dists < 0.01).astype(int)
    np.fill_diagonal(adjacency, 0)
    n_components, _ = connected_components(csr_matrix(adjacency), directed=False)

    # Accuracy (leave-one-out)
    correct = 0
    total = 0
    for test_env_id, test_trials in enumerate(trial_codes_all):
        for test_idx, test_code in enumerate(test_trials):
            centroids = []
            for env_id, env_trials in enumerate(trial_codes_all):
                if env_id == test_env_id:
                    other = [t for i, t in enumerate(env_trials) if i != test_idx]
                    centroids.append(np.mean(other, axis=0) if other else env_trials[0])
                else:
                    centroids.append(np.mean(env_trials, axis=0))
            predicted = np.argmin([np.linalg.norm(test_code - c) for c in centroids])
            if predicted == test_env_id:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    return {
        'n_species': n_species,
        'n_reactions': n_reactions,
        'n_outputs': n_outputs,
        'seed': seed,
        'hill_coeff': hill_coeff,
        's0': s0,
        'coupling_strength': coupling_strength,
        'd_eff': float(d_eff),
        'unique_codes': int(n_components),
        'accuracy': float(accuracy)
    }


def run_ensemble(n_seeds=50, species_counts=[15, 25, 35, 50]):
    """Run ensemble across species counts and random seeds."""
    print("=" * 70)
    print(f"ENSEMBLE SWEEP: {n_seeds} seeds × {len(species_counts)} species counts")
    print("=" * 70)

    results = []
    for n_species in species_counts:
        n_reactions = n_species * 2  # Scale reactions with species
        n_outputs = 8

        print(f"\n{n_species} species:")
        for i in range(n_seeds):
            seed = i * 1000
            start = time.time()
            result = run_chemistry(n_species, n_reactions, n_outputs, seed)
            elapsed = time.time() - start
            results.append(result)
            print(f"  Seed {i+1}/{n_seeds}: D_eff={result['d_eff']:.2f}, "
                  f"Acc={result['accuracy']*100:.0f}%, "
                  f"Unique={result['unique_codes']} [{elapsed:.1f}s]")

    return results


def run_ablations(seeds=[0, 1000, 2000, 3000, 4000]):
    """Run mechanism ablations: h=1 vs h=4, coupling=0 vs 0.1."""
    print("\n" + "=" * 70)
    print("MECHANISM ABLATIONS")
    print("=" * 70)

    n_species, n_reactions, n_outputs = 20, 40, 8
    ablation_results = []

    conditions = [
        ('baseline', {'hill_coeff': 4.0, 's0': 1.0, 'coupling_strength': 0.1}),
        ('h=1 (no cooperativity)', {'hill_coeff': 1.0, 's0': 1.0, 'coupling_strength': 0.1}),
        ('coupling=0 (isolated)', {'hill_coeff': 4.0, 's0': 1.0, 'coupling_strength': 0.0}),
        ('high S0 (weak competition)', {'hill_coeff': 4.0, 's0': 10.0, 'coupling_strength': 0.1}),
    ]

    for cond_name, params in conditions:
        print(f"\n{cond_name}:")
        cond_results = []
        for seed in seeds:
            start = time.time()
            result = run_chemistry(n_species, n_reactions, n_outputs, seed, **params)
            result['condition'] = cond_name
            cond_results.append(result)
            elapsed = time.time() - start
            print(f"  Seed {seed}: D_eff={result['d_eff']:.2f}, "
                  f"Acc={result['accuracy']*100:.0f}% [{elapsed:.1f}s]")

        # Aggregate
        d_eff_mean = np.mean([r['d_eff'] for r in cond_results])
        acc_mean = np.mean([r['accuracy'] for r in cond_results])
        print(f"  Mean: D_eff={d_eff_mean:.2f}, Acc={acc_mean*100:.0f}%")
        ablation_results.extend(cond_results)

    return ablation_results


def run_threshold_robustness(thresholds=[0.001, 0.005, 0.01, 0.02, 0.05]):
    """Test sensitivity of unique code count to collision threshold."""
    print("\n" + "=" * 70)
    print("COLLISION THRESHOLD ROBUSTNESS")
    print("=" * 70)

    n_species, n_reactions, n_outputs = 20, 40, 8
    n_envs, n_trials = 16, 2
    seeds = [0, 1000, 2000]

    # Run chemistries once, then vary threshold
    robustness_results = []

    for seed in seeds:
        array = ChemicalVesicleArray(
            n_rings=2, n_species=n_species, n_reactions=n_reactions,
            n_outputs=n_outputs, seed=seed
        )

        codes = []
        for env_id in range(n_envs):
            env_inputs = create_spatial_gradient(array.coords, env_id)
            trial_codes = []
            for trial in range(n_trials):
                code, _ = array.run(env_inputs, t_span=(0, 50), n_points=200,
                                   trial_seed=seed + env_id * 1000 + trial)
                trial_codes.append(code)
            codes.append(np.mean(trial_codes, axis=0))
        codes = np.array(codes)

        # Test different thresholds
        code_dists = squareform(pdist(codes))
        for thresh in thresholds:
            adjacency = (code_dists < thresh).astype(int)
            np.fill_diagonal(adjacency, 0)
            n_components, _ = connected_components(csr_matrix(adjacency), directed=False)
            robustness_results.append({
                'seed': seed,
                'threshold': thresh,
                'unique_codes': n_components
            })

    # Print summary
    print("\nThreshold → Mean Unique Codes (across seeds):")
    for thresh in thresholds:
        mean_unique = np.mean([r['unique_codes'] for r in robustness_results
                              if r['threshold'] == thresh])
        print(f"  {thresh:.3f}: {mean_unique:.1f}")

    return robustness_results


def generate_figures(ensemble_results, ablation_results, robustness_results):
    """Generate publication figures."""
    import os
    os.makedirs('figures', exist_ok=True)

    # Figure: Accuracy vs D_eff scatter (colored by species count)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {15: '#1f77b4', 25: '#ff7f0e', 35: '#2ca02c', 50: '#d62728'}
    species_counts = sorted(set(r['n_species'] for r in ensemble_results))

    for n_species in species_counts:
        subset = [r for r in ensemble_results if r['n_species'] == n_species]
        d_effs = [r['d_eff'] for r in subset]
        accs = [r['accuracy'] * 100 for r in subset]
        ax.scatter(d_effs, accs, c=colors.get(n_species, 'gray'),
                  label=f'{n_species} species', alpha=0.7, s=50)

    # Add regression line
    all_d_eff = [r['d_eff'] for r in ensemble_results]
    all_acc = [r['accuracy'] * 100 for r in ensemble_results]
    slope, intercept, r_val, p_val, _ = stats.linregress(all_d_eff, all_acc)
    x_line = np.linspace(min(all_d_eff), max(all_d_eff), 100)
    # Format p-value properly (avoid "p = 0.000")
    p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    ax.plot(x_line, slope * x_line + intercept, 'k--',
           label=f'r = {r_val:.2f}, {p_str}')

    ax.set_xlabel('Effective Dimensionality ($D_{eff}$)', fontsize=12)
    ax.set_ylabel('Decode Accuracy (%)', fontsize=12)
    ax.set_title('$D_{eff}$ Predicts Code Quality Across Random Chemistries',
                fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('figures/fig_ensemble_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_ensemble_scatter.png', dpi=300, bbox_inches='tight')
    print("\nSaved figures/fig_ensemble_scatter.pdf")
    plt.close()

    # Figure: Ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = ['baseline', 'h=1 (no cooperativity)',
                 'coupling=0 (isolated)', 'high S0 (weak competition)']
    x = np.arange(len(conditions))
    width = 0.35

    d_eff_means = []
    acc_means = []
    for cond in conditions:
        subset = [r for r in ablation_results if r['condition'] == cond]
        d_eff_means.append(np.mean([r['d_eff'] for r in subset]))
        acc_means.append(np.mean([r['accuracy'] * 100 for r in subset]))

    bars1 = ax.bar(x - width/2, d_eff_means, width, label='$D_{eff}$', color='#1f77b4')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, acc_means, width, label='Accuracy (%)', color='#2ca02c')

    ax.set_ylabel('$D_{eff}$', fontsize=11, color='#1f77b4')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_title('Mechanism Ablations: Substrate Competition is Key',
                fontsize=13, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('figures/fig_ablations.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_ablations.png', dpi=300, bbox_inches='tight')
    print("Saved figures/fig_ablations.pdf")
    plt.close()

    # Figure: Threshold robustness
    fig, ax = plt.subplots(figsize=(6, 4))

    thresholds = sorted(set(r['threshold'] for r in robustness_results))
    means = [np.mean([r['unique_codes'] for r in robustness_results
                     if r['threshold'] == t]) for t in thresholds]
    stds = [np.std([r['unique_codes'] for r in robustness_results
                   if r['threshold'] == t]) for t in thresholds]

    ax.errorbar(thresholds, means, yerr=stds, marker='o', capsize=5, color='#1f77b4')
    ax.axvline(0.01, color='red', linestyle='--', alpha=0.5, label='Used threshold (0.01)')
    ax.set_xlabel('Collision Threshold', fontsize=11)
    ax.set_ylabel('Unique Codes (out of 16)', fontsize=11)
    ax.set_title('Unique Code Count vs Threshold', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()

    plt.tight_layout()
    plt.savefig('figures/fig_threshold_robustness.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_threshold_robustness.png', dpi=300, bbox_inches='tight')
    print("Saved figures/fig_threshold_robustness.pdf")
    plt.close()


def main():
    print("=" * 70)
    print("ENSEMBLE + ABLATIONS + ROBUSTNESS ANALYSIS")
    print("=" * 70)

    # 1. Ensemble sweep
    ensemble_results = run_ensemble(n_seeds=30, species_counts=[15, 25, 35, 50])

    # 2. Ablations
    ablation_results = run_ablations(seeds=[0, 1000, 2000, 3000, 4000])

    # 3. Threshold robustness
    robustness_results = run_threshold_robustness()

    # 4. Generate figures
    generate_figures(ensemble_results, ablation_results, robustness_results)

    # 5. Statistical summary
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    # D_eff vs accuracy correlation
    all_d_eff = [r['d_eff'] for r in ensemble_results]
    all_acc = [r['accuracy'] for r in ensemble_results]
    all_species = [r['n_species'] for r in ensemble_results]

    r_d_eff, p_d_eff = stats.pearsonr(all_d_eff, all_acc)
    r_species, p_species = stats.pearsonr(all_species, all_acc)

    print(f"\nAccuracy correlation with D_eff: r = {r_d_eff:.3f}, p = {p_d_eff:.4f}")
    print(f"Accuracy correlation with species count: r = {r_species:.3f}, p = {p_species:.4f}")

    if abs(r_d_eff) > abs(r_species):
        print("\n✓ D_eff is a BETTER predictor of accuracy than species count")
    else:
        print("\n✗ Species count predicts better (unexpected)")

    # Ablation summary
    print("\nAblation Summary:")
    baseline = [r for r in ablation_results if r['condition'] == 'baseline']
    baseline_acc = np.mean([r['accuracy'] for r in baseline])

    for cond in ['h=1 (no cooperativity)', 'coupling=0 (isolated)', 'high S0 (weak competition)']:
        subset = [r for r in ablation_results if r['condition'] == cond]
        cond_acc = np.mean([r['accuracy'] for r in subset])
        diff = (cond_acc - baseline_acc) * 100
        print(f"  {cond}: {diff:+.1f}% vs baseline")

    # Save all results
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/ensemble_sweep.json', 'w') as f:
        json.dump({
            'ensemble': ensemble_results,
            'ablations': ablation_results,
            'robustness': robustness_results,
            'stats': {
                'r_d_eff_accuracy': r_d_eff,
                'p_d_eff_accuracy': p_d_eff,
                'r_species_accuracy': r_species,
                'p_species_accuracy': p_species
            }
        }, f, indent=2)
    print("\nResults saved to results/ensemble_sweep.json")


if __name__ == "__main__":
    main()
