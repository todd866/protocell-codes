#!/usr/bin/env python3
"""
Thorough timescale separation test across multiple random chemistries.
For each chemistry, compare fast vs mixed using SAME topology (only rates differ).
"""

import numpy as np
import json
import time
from chemistry_sim import (
    AutocatalyticNetwork, ChemicalVesicleArray, 
    hexagonal_grid, get_neighbors, create_spatial_gradient,
    participation_ratio, substrate_competition
)
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class MixedTimescaleNetwork(AutocatalyticNetwork):
    """Network with some reactions slowed down."""
    def __init__(self, n_species=15, n_reactions=30, slow_fraction=0.3,
                 slow_factor=0.01, rng=None):
        super().__init__(n_species, n_reactions, rng)
        n_slow = int(n_reactions * slow_fraction)
        slow_indices = self.rng.choice(n_reactions, n_slow, replace=False)
        for i in slow_indices:
            self.rates[i] *= slow_factor


class MixedTimescaleArray(ChemicalVesicleArray):
    """Vesicle array with mixed timescale chemistry."""
    def __init__(self, n_rings=2, n_species=15, n_reactions=30,
                 n_outputs=8, coupling_strength=0.1, seed=42,
                 slow_fraction=0.3, slow_factor=0.01):
        self.coords = hexagonal_grid(n_rings)
        self.n_vesicles = len(self.coords)
        self.neighbors = get_neighbors(self.coords)
        self.coupling_strength = coupling_strength
        self.n_species = n_species
        self.n_outputs = n_outputs

        master_rng = np.random.default_rng(seed)
        base_network = MixedTimescaleNetwork(
            n_species, n_reactions, slow_fraction, slow_factor, rng=master_rng
        )

        self.networks = []
        for i in range(self.n_vesicles):
            vesicle_rng = np.random.default_rng(seed + i + 1)
            net = base_network.copy_with_perturbation(perturbation=0.1, rng=vesicle_rng)
            self.networks.append(net)

        self.output_species = list(range(n_species - n_outputs, n_species))
        for net in self.networks:
            net.output_species = self.output_species


def run_single_chemistry(seed, n_envs=16, n_trials=2, n_rings=2, 
                         n_species=20, n_reactions=40, n_outputs=8,
                         slow_fraction=0.3, slow_factor=0.01):
    """Run both fast and mixed conditions for one chemistry seed."""
    
    results = {'seed': seed}
    
    # Condition 1: All-fast (uniform timescale)
    array_fast = ChemicalVesicleArray(
        n_rings=n_rings, n_species=n_species, n_reactions=n_reactions,
        n_outputs=n_outputs, seed=seed
    )
    
    fast_codes = []
    fast_trials = []
    for env_id in range(n_envs):
        env_inputs = create_spatial_gradient(array_fast.coords, env_id)
        trial_codes = []
        for trial in range(n_trials):
            code, _ = array_fast.run(env_inputs, t_span=(0, 50), n_points=200,
                                     trial_seed=seed + env_id * 1000 + trial)
            trial_codes.append(code)
        fast_codes.append(np.mean(trial_codes, axis=0))
        fast_trials.append(trial_codes)
    
    fast_codes = np.array(fast_codes)
    results['fast'] = compute_metrics(fast_codes, fast_trials, n_envs)
    
    # Condition 2: Mixed timescales (SAME seed = same topology)
    array_mixed = MixedTimescaleArray(
        n_rings=n_rings, n_species=n_species, n_reactions=n_reactions,
        n_outputs=n_outputs, seed=seed,  # SAME seed
        slow_fraction=slow_fraction, slow_factor=slow_factor
    )
    
    mixed_codes = []
    mixed_trials = []
    for env_id in range(n_envs):
        env_inputs = create_spatial_gradient(array_mixed.coords, env_id)
        trial_codes = []
        for trial in range(n_trials):
            code, _ = array_mixed.run(env_inputs, t_span=(0, 100), n_points=300,
                                      trial_seed=seed + env_id * 1000 + trial)
            trial_codes.append(code)
        mixed_codes.append(np.mean(trial_codes, axis=0))
        mixed_trials.append(trial_codes)
    
    mixed_codes = np.array(mixed_codes)
    results['mixed'] = compute_metrics(mixed_codes, mixed_trials, n_envs)
    
    return results


def compute_metrics(codes, trial_codes_all, n_envs):
    """Compute D_eff, accuracy, unique codes."""
    # D_eff
    d_eff = participation_ratio(codes)
    
    # Unique codes via clustering
    code_dists = squareform(pdist(codes))
    adjacency = (code_dists < 0.01).astype(int)
    np.fill_diagonal(adjacency, 0)
    n_components, _ = connected_components(csr_matrix(adjacency), directed=False)
    
    # Accuracy
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
    
    return {
        'd_eff': float(d_eff),
        'unique_codes': int(n_components),
        'accuracy': float(correct / total) if total > 0 else 0
    }


def main():
    N_CHEMISTRIES = 20
    print("=" * 70)
    print(f"TIMESCALE SEPARATION: {N_CHEMISTRIES} RANDOM CHEMISTRIES")
    print("=" * 70)
    print("\nFor each chemistry, comparing fast vs mixed (SAME topology).")
    print("Only rate constants differ between conditions.\n")
    
    all_results = []
    
    for i in range(N_CHEMISTRIES):
        seed = i * 1000
        print(f"Chemistry {i+1}/{N_CHEMISTRIES} (seed={seed})...", end=" ", flush=True)
        start = time.time()
        
        result = run_single_chemistry(seed)
        all_results.append(result)
        
        fast = result['fast']
        mixed = result['mixed']
        elapsed = time.time() - start
        
        d_eff_change = mixed['d_eff'] - fast['d_eff']
        acc_change = (mixed['accuracy'] - fast['accuracy']) * 100
        
        print(f"D_eff: {fast['d_eff']:.2f}→{mixed['d_eff']:.2f} ({d_eff_change:+.2f}), "
              f"Acc: {fast['accuracy']*100:.0f}%→{mixed['accuracy']*100:.0f}% ({acc_change:+.0f}%) "
              f"[{elapsed:.0f}s]")
    
    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    fast_d_eff = [r['fast']['d_eff'] for r in all_results]
    mixed_d_eff = [r['mixed']['d_eff'] for r in all_results]
    fast_acc = [r['fast']['accuracy'] for r in all_results]
    mixed_acc = [r['mixed']['accuracy'] for r in all_results]
    fast_unique = [r['fast']['unique_codes'] for r in all_results]
    mixed_unique = [r['mixed']['unique_codes'] for r in all_results]
    
    d_eff_diff = np.array(mixed_d_eff) - np.array(fast_d_eff)
    acc_diff = np.array(mixed_acc) - np.array(fast_acc)
    unique_diff = np.array(mixed_unique) - np.array(fast_unique)
    
    print(f"\n{'Metric':<20} {'Fast (mean±std)':<20} {'Mixed (mean±std)':<20} {'Diff':<15}")
    print("-" * 75)
    print(f"{'D_eff':<20} {np.mean(fast_d_eff):.2f} ± {np.std(fast_d_eff):.2f}       "
          f"{np.mean(mixed_d_eff):.2f} ± {np.std(mixed_d_eff):.2f}       "
          f"{np.mean(d_eff_diff):+.2f}")
    print(f"{'Accuracy':<20} {np.mean(fast_acc)*100:.1f}% ± {np.std(fast_acc)*100:.1f}%     "
          f"{np.mean(mixed_acc)*100:.1f}% ± {np.std(mixed_acc)*100:.1f}%     "
          f"{np.mean(acc_diff)*100:+.1f}%")
    print(f"{'Unique codes':<20} {np.mean(fast_unique):.1f} ± {np.std(fast_unique):.1f}       "
          f"{np.mean(mixed_unique):.1f} ± {np.std(mixed_unique):.1f}       "
          f"{np.mean(unique_diff):+.1f}")
    
    # Count wins
    d_eff_wins = sum(1 for d in d_eff_diff if d > 0)
    acc_wins = sum(1 for a in acc_diff if a > 0)
    unique_wins = sum(1 for u in unique_diff if u > 0)
    
    print(f"\nMixed timescales helped in:")
    print(f"  D_eff: {d_eff_wins}/{N_CHEMISTRIES} chemistries ({d_eff_wins/N_CHEMISTRIES*100:.0f}%)")
    print(f"  Accuracy: {acc_wins}/{N_CHEMISTRIES} chemistries ({acc_wins/N_CHEMISTRIES*100:.0f}%)")
    print(f"  Unique codes: {unique_wins}/{N_CHEMISTRIES} chemistries ({unique_wins/N_CHEMISTRIES*100:.0f}%)")
    
    # Statistical test
    from scipy import stats
    t_d_eff, p_d_eff = stats.ttest_rel(mixed_d_eff, fast_d_eff)
    t_acc, p_acc = stats.ttest_rel(mixed_acc, fast_acc)
    
    print(f"\nPaired t-tests (mixed vs fast):")
    print(f"  D_eff: t={t_d_eff:.2f}, p={p_d_eff:.3f}")
    print(f"  Accuracy: t={t_acc:.2f}, p={p_acc:.3f}")
    
    if p_acc < 0.05:
        if np.mean(acc_diff) > 0:
            print("\n✓ TIMESCALE SEPARATION SIGNIFICANTLY IMPROVES ACCURACY (p<0.05)")
        else:
            print("\n✗ TIMESCALE SEPARATION SIGNIFICANTLY WORSENS ACCURACY (p<0.05)")
    else:
        print(f"\n? No significant effect on accuracy (p={p_acc:.3f})")
    
    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/timescale_sweep.json", "w") as f:
        json.dump({
            'n_chemistries': N_CHEMISTRIES,
            'results': all_results,
            'aggregate': {
                'fast_d_eff': {'mean': float(np.mean(fast_d_eff)), 'std': float(np.std(fast_d_eff))},
                'mixed_d_eff': {'mean': float(np.mean(mixed_d_eff)), 'std': float(np.std(mixed_d_eff))},
                'fast_accuracy': {'mean': float(np.mean(fast_acc)), 'std': float(np.std(fast_acc))},
                'mixed_accuracy': {'mean': float(np.mean(mixed_acc)), 'std': float(np.std(mixed_acc))},
                'd_eff_wins': d_eff_wins,
                'acc_wins': acc_wins,
                'p_value_accuracy': float(p_acc),
                'p_value_d_eff': float(p_d_eff)
            }
        }, f, indent=2)
    print("\nResults saved to results/timescale_sweep.json")


if __name__ == "__main__":
    main()
