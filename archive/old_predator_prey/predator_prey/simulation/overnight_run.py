#!/usr/bin/env python3
"""
Overnight Exploratory Simulation: Predator-Prey from Coherence Constraints
============================================================================

Run this overnight on M3 Max. Should take 4-6 hours for full run.

Usage:
    python overnight_run.py

Output:
    results/overnight_YYYYMMDD_HHMMSS/
        - trajectory.npz (population snapshots)
        - summary.json (run metadata)
        - figures/ (auto-generated plots)

Author: Ian Todd
Date: 2026-01-05
"""

import numpy as np
from numpy.random import default_rng
import json
import os
from datetime import datetime
from pathlib import Path
import time

# =============================================================================
# PARAMETERS
# =============================================================================

PARAMS = {
    # Population
    'M': 5000,              # Number of colonies
    'N_max': 200,           # Max colony size
    'N_min': 5,             # Min colony size

    # Coherence cost: Cost = k * N^gamma * C^zeta
    'gamma': 2.0,           # Quadratic in N
    'zeta': 1.5,            # Coherence exponent
    'k': 0.005,             # Cost coefficient

    # Coordination multiplier: phi(C) = 1 + a*(C - C_star)^m for C > C_star
    'a': 2.0,               # Multiplier strength
    'm': 1.5,               # Superlinearity
    'C_star': 0.5,          # Threshold - high so low-C truly viable

    # HARD CEILING: High-C colonies CANNOT exceed this size
    # This is the key constraint that creates size differentiation.
    # Biological interpretation: coherent oscillation breaks down above critical size.
    'N_ceiling': 40,        # Hard cap for high-C colonies

    # Extraction (predation)
    'kappa_C': 3.0,         # Coherence sensitivity - strong C advantage
    'kappa_N': 2.0,         # Size sensitivity - size provides defense
    'f': 0.15,              # Extraction fraction
    'encounter_rate': 0.05, # Fraction of population that encounters per gen

    # Evolution
    'sigma_N': 10.0,        # Mutation std for N
    'sigma_C': 0.08,        # Mutation std for C
    'base_R': 1.0,          # Base resource density

    # Simulation
    'generations': 20000,   # Longer run to see equilibrium dynamics
    'snapshot_interval': 200,  # Save full state every N generations
    'seed': 42,
}

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def phi(C, params):
    """Coordination multiplier - superlinear above threshold."""
    above = C > params['C_star']
    # Use np.maximum to avoid negative values raised to fractional powers
    diff = np.maximum(0, C - params['C_star'])
    return 1.0 + params['a'] * np.where(above, diff**params['m'], 0.0)

def cost(N, C, params):
    """Metabolic cost of coherence.

    Cost = k * N^gamma * C^zeta (superlinear in both)

    Note: Size differentiation comes from HARD CEILING in selection_step(),
    not from this cost function. The cost function shapes fitness landscape,
    but doesn't enforce the N-C tradeoff strongly enough on its own.
    """
    return params['k'] * (N ** params['gamma']) * (C ** params['zeta'])

def fitness(N, C, R, params):
    """Net fitness.

    Key dynamics:
    - Large N with low C: high resource gathering, low cost -> viable
    - Small N with high C: coordination bonus, manageable cost -> viable
    - Large N with high C: coordination bonus BUT huge cost -> not viable
    - Small N with low C: low gathering, no bonus -> marginal
    """
    benefit = N * params['base_R'] * phi(C, params)
    metabolic_cost = cost(N, C, params)

    # Resource-weighted fitness (having resources helps survive)
    resource_factor = np.sqrt(R / (R + 1))  # Diminishing returns

    return benefit * resource_factor - metabolic_cost

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def raid_probability(C_pred, C_prey, N_pred, N_prey, params):
    """Probability of successful raid."""
    # Avoid log(0)
    C_pred = np.maximum(C_pred, 1e-6)
    C_prey = np.maximum(C_prey, 1e-6)
    N_pred = np.maximum(N_pred, 1.0)
    N_prey = np.maximum(N_prey, 1.0)

    logit = (params['kappa_C'] * np.log(C_pred / C_prey) -
             params['kappa_N'] * np.log(N_prey / N_pred))
    return sigmoid(logit)

# =============================================================================
# SIMULATION
# =============================================================================

def initialize(params, rng):
    """Initialize population with random (N, C) in viable region."""
    M = params['M']

    # Start with uniform random
    N = rng.uniform(params['N_min'], params['N_max'], M)
    C = rng.uniform(0.0, 1.0, M)
    R = np.ones(M) * params['base_R'] * 10  # Start with some resources

    return N, C, R

def selection_step(N, C, R, params, rng):
    """Selection: kill low fitness, reproduce high fitness."""
    W = fitness(N, C, R, params)

    # Death: negative fitness dies
    alive = W > 0

    # Also random death to prevent population explosion
    alive = alive & (rng.random(len(N)) > 0.1)

    if np.sum(alive) < 10:
        # Population crash - restart with survivors cloned
        alive[:100] = True

    N = N[alive]
    C = C[alive]
    R = R[alive]
    W = W[alive]

    # Reproduction: proportional to fitness
    n_alive = len(N)
    if n_alive == 0:
        return initialize(params, rng)

    # Target population size
    target = params['M']
    n_offspring = target - n_alive

    if n_offspring > 0 and n_alive > 0:
        # Select parents proportional to fitness
        probs = np.maximum(W, 0)
        probs = probs / probs.sum()

        parent_idx = rng.choice(n_alive, size=n_offspring, p=probs)

        # Offspring with mutation
        N_new = N[parent_idx] + rng.normal(0, params['sigma_N'], n_offspring)
        C_new = C[parent_idx] + rng.normal(0, params['sigma_C'], n_offspring)
        R_new = R[parent_idx] * 0.5  # Split resources with parent
        R[parent_idx] *= 0.5

        # Bounds
        N_new = np.clip(N_new, params['N_min'], params['N_max'])
        C_new = np.clip(C_new, 0.0, 1.0)

        N = np.concatenate([N, N_new])
        C = np.concatenate([C, C_new])
        R = np.concatenate([R, R_new])

    # ==========================================================================
    # HARD CEILING: High-coherence colonies cannot exceed N_ceiling
    # This is the key constraint that creates size differentiation.
    # Soft cost penalties are insufficient - we need a hard physical limit.
    # Biological interpretation: coherent oscillation requires tight coupling,
    # which breaks down above a critical size threshold.
    # ==========================================================================
    N_ceiling = params['N_ceiling']
    C_threshold = 0.5  # Same as predator/prey classification threshold

    # High-C colonies are hard-capped at N_ceiling
    high_C_mask = C > C_threshold
    N = np.where(high_C_mask, np.minimum(N, N_ceiling), N)

    return N, C, R

def extraction_step(N, C, R, params, rng):
    """Predator-prey extraction based on coherence differential."""
    M = len(N)
    if M < 2:
        return N, C, R

    # Number of encounters
    n_encounters = int(params['encounter_rate'] * M)

    # Random pairings
    idx1 = rng.choice(M, size=n_encounters)
    idx2 = rng.choice(M, size=n_encounters)

    # Avoid self-encounters
    valid = idx1 != idx2
    idx1 = idx1[valid]
    idx2 = idx2[valid]

    if len(idx1) == 0:
        return N, C, R

    # Determine predator (higher C) vs prey (lower C)
    is_pred_1 = C[idx1] > C[idx2]

    pred_idx = np.where(is_pred_1, idx1, idx2)
    prey_idx = np.where(is_pred_1, idx2, idx1)

    # Raid probability
    p_raid = raid_probability(C[pred_idx], C[prey_idx],
                               N[pred_idx], N[prey_idx], params)

    # Successful raids
    success = rng.random(len(p_raid)) < p_raid

    # Resource transfer
    transfer = params['f'] * R[prey_idx[success]]

    # Use np.add.at to handle duplicate indices
    np.add.at(R, pred_idx[success], transfer)
    np.add.at(R, prey_idx[success], -transfer)

    # Ensure non-negative resources
    R = np.maximum(R, 0)

    return N, C, R

def resource_gathering(N, C, R, params):
    """Colonies gather resources proportional to size."""
    R_new = R + params['base_R'] * N * phi(C, params) * 0.1
    return R_new

def step(N, C, R, params, rng):
    """One generation."""
    R = resource_gathering(N, C, R, params)
    N, C, R = extraction_step(N, C, R, params, rng)
    N, C, R = selection_step(N, C, R, params, rng)
    return N, C, R

def classify_population(N, C):
    """Classify into predator/prey based on C threshold."""
    C_threshold = 0.5
    is_predator = C > C_threshold
    return is_predator

def compute_stats(N, C, R, params):
    """Compute summary statistics."""
    is_pred = classify_population(N, C)

    stats = {
        'n_total': len(N),
        'n_predator': np.sum(is_pred),
        'n_prey': np.sum(~is_pred),
        'mean_N': np.mean(N),
        'mean_C': np.mean(C),
        'mean_R': np.mean(R),
        'mean_N_pred': np.mean(N[is_pred]) if np.any(is_pred) else 0,
        'mean_N_prey': np.mean(N[~is_pred]) if np.any(~is_pred) else 0,
        'mean_C_pred': np.mean(C[is_pred]) if np.any(is_pred) else 0,
        'mean_C_prey': np.mean(C[~is_pred]) if np.any(~is_pred) else 0,
        'std_N': np.std(N),
        'std_C': np.std(C),
    }
    return stats

# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(snapshots, stats_history, output_dir):
    """Generate analysis figures."""
    import matplotlib.pyplot as plt

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Figure 1: Phase diagram evolution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    times = [0, len(snapshots)//2, -1]
    titles = ['Initial', 'Midpoint', 'Final']

    for ax, t, title in zip(axes, times, titles):
        snap = snapshots[t]
        ax.scatter(snap['N'], snap['C'], alpha=0.3, s=5)
        ax.set_xlabel('Size N')
        ax.set_ylabel('Coherence C')
        ax.set_title(f'{title} (gen {snap["generation"]})')
        ax.set_xlim(0, PARAMS['N_max'])
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(fig_dir / 'phase_diagram.png', dpi=150)
    plt.close()

    # Figure 2: Population dynamics
    gens = [s['generation'] for s in stats_history]
    n_pred = [s['n_predator'] for s in stats_history]
    n_prey = [s['n_prey'] for s in stats_history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Population counts
    axes[0, 0].plot(gens, n_pred, label='Predators (high C)', alpha=0.8)
    axes[0, 0].plot(gens, n_prey, label='Prey (low C)', alpha=0.8)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Population Dynamics')
    axes[0, 0].legend()

    # Phase portrait
    axes[0, 1].plot(n_prey, n_pred, alpha=0.5, lw=0.5)
    axes[0, 1].scatter(n_prey[0], n_pred[0], c='green', s=100, zorder=5, label='Start')
    axes[0, 1].scatter(n_prey[-1], n_pred[-1], c='red', s=100, zorder=5, label='End')
    axes[0, 1].set_xlabel('Prey count')
    axes[0, 1].set_ylabel('Predator count')
    axes[0, 1].set_title('Phase Portrait')
    axes[0, 1].legend()

    # Mean coherence by type
    C_pred = [s['mean_C_pred'] for s in stats_history]
    C_prey = [s['mean_C_prey'] for s in stats_history]
    axes[1, 0].plot(gens, C_pred, label='Predator C', alpha=0.8)
    axes[1, 0].plot(gens, C_prey, label='Prey C', alpha=0.8)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Mean Coherence')
    axes[1, 0].set_title('Coherence by Type')
    axes[1, 0].legend()

    # Mean size by type
    N_pred = [s['mean_N_pred'] for s in stats_history]
    N_prey = [s['mean_N_prey'] for s in stats_history]
    axes[1, 1].plot(gens, N_pred, label='Predator N', alpha=0.8)
    axes[1, 1].plot(gens, N_prey, label='Prey N', alpha=0.8)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Mean Size')
    axes[1, 1].set_title('Size by Type')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / 'dynamics.png', dpi=150)
    plt.close()

    # Figure 3: Final distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    final = snapshots[-1]
    axes[0].hist(final['N'], bins=50, alpha=0.7)
    axes[0].set_xlabel('Size N')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Final Size Distribution')

    axes[1].hist(final['C'], bins=50, alpha=0.7)
    axes[1].set_xlabel('Coherence C')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Final Coherence Distribution')

    plt.tight_layout()
    plt.savefig(fig_dir / 'distributions.png', dpi=150)
    plt.close()

    print(f"Figures saved to {fig_dir}")

# =============================================================================
# MAIN
# =============================================================================

def run_simulation(params):
    """Run full simulation."""

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'results' / f'overnight_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Parameters: {json.dumps(params, indent=2)}")

    # Save params
    with open(output_dir / 'params.json', 'w') as f:
        json.dump(params, f, indent=2)

    # Initialize
    rng = default_rng(params['seed'])
    N, C, R = initialize(params, rng)

    # Storage
    snapshots = []
    stats_history = []

    # Initial snapshot
    snapshots.append({
        'generation': 0,
        'N': N.copy(),
        'C': C.copy(),
        'R': R.copy(),
    })
    stats_history.append({'generation': 0, **compute_stats(N, C, R, params)})

    # Run
    start_time = time.time()

    for gen in range(1, params['generations'] + 1):
        N, C, R = step(N, C, R, params, rng)

        # Stats every generation
        stats = {'generation': gen, **compute_stats(N, C, R, params)}
        stats_history.append(stats)

        # Snapshot at intervals
        if gen % params['snapshot_interval'] == 0:
            snapshots.append({
                'generation': gen,
                'N': N.copy(),
                'C': C.copy(),
                'R': R.copy(),
            })

            elapsed = time.time() - start_time
            rate = gen / elapsed
            eta = (params['generations'] - gen) / rate

            print(f"Gen {gen:5d}/{params['generations']} | "
                  f"Pop: {len(N):4d} | "
                  f"Pred: {stats['n_predator']:4d} | "
                  f"Prey: {stats['n_prey']:4d} | "
                  f"Rate: {rate:.1f} gen/s | "
                  f"ETA: {eta/60:.1f} min")

    total_time = time.time() - start_time
    print(f"\nSimulation complete in {total_time/60:.1f} minutes")

    # Save trajectory
    np.savez_compressed(
        output_dir / 'trajectory.npz',
        snapshots=snapshots,
        stats_history=stats_history,
    )

    # Save summary (convert numpy types to native Python)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    summary = {
        'params': params,
        'total_time_seconds': total_time,
        'final_stats': convert_numpy(stats_history[-1]),
        'n_snapshots': len(snapshots),
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    try:
        plot_results(snapshots, stats_history, output_dir)
    except ImportError:
        print("matplotlib not available, skipping plots")

    print(f"\nResults saved to {output_dir}")
    return output_dir

if __name__ == '__main__':
    run_simulation(PARAMS)
