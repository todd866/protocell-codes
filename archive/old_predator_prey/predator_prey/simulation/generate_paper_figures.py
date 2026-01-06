#!/usr/bin/env python3
"""
Generate publication-quality figures for the predator-prey paper.
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
from overnight_run import PARAMS, initialize, step, classify_population, compute_stats

# Style settings for publication
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def run_simulation(params, generations=20000):
    """Run simulation and collect trajectory."""
    rng = default_rng(params['seed'])
    N, C, R = initialize(params, rng)

    snapshots = []
    stats_history = []

    # Initial snapshot
    snapshots.append({'gen': 0, 'N': N.copy(), 'C': C.copy()})
    stats_history.append({'gen': 0, **compute_stats(N, C, R, params)})

    interval = generations // 100  # 100 snapshots

    for gen in range(1, generations + 1):
        N, C, R = step(N, C, R, params, rng)

        stats = {'gen': gen, **compute_stats(N, C, R, params)}
        stats_history.append(stats)

        if gen % interval == 0:
            snapshots.append({'gen': gen, 'N': N.copy(), 'C': C.copy()})

            if gen % (generations // 10) == 0:
                is_pred = classify_population(N, C)
                print(f"Gen {gen:5d}: Pred={np.sum(is_pred):3d}, N_pred={np.mean(N[is_pred]):.1f}, N_prey={np.mean(N[~is_pred]):.1f}")

    return snapshots, stats_history

def figure_1_phase_evolution(snapshots, output_dir):
    """Figure 1: Phase diagram showing two-attractor emergence."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    times = [0, len(snapshots)//2, -1]
    titles = ['(A) Initial', '(B) Midpoint', '(C) Equilibrium']

    for ax, t, title in zip(axes, times, titles):
        snap = snapshots[t]
        is_pred = snap['C'] > 0.5

        # Plot prey first (background)
        ax.scatter(snap['N'][~is_pred], snap['C'][~is_pred],
                  alpha=0.3, s=8, c='#2171b5', label='Prey')
        # Plot predators (foreground)
        ax.scatter(snap['N'][is_pred], snap['C'][is_pred],
                  alpha=0.6, s=15, c='#cb181d', label='Predator')

        ax.set_xlabel('Colony size $N$')
        ax.set_ylabel('Coherence $C$')
        ax.set_title(f'{title} (gen {snap["gen"]})')
        ax.set_xlim(0, 210)
        ax.set_ylim(0, 1)

        # Add ceiling line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, lw=0.8)
        if t == -1:
            ax.axvline(x=40, color='gray', linestyle='--', alpha=0.5, lw=0.8)

        if t == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_phase_evolution.pdf')
    plt.savefig(output_dir / 'fig1_phase_evolution.png', dpi=300)
    plt.close()
    print("Saved Figure 1: Phase evolution")

def figure_2_dynamics(stats_history, output_dir):
    """Figure 2: Population dynamics over time."""
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig)

    gens = [s['gen'] for s in stats_history]

    # Panel A: Population counts
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gens, [s['n_prey'] for s in stats_history],
            c='#2171b5', lw=1.5, label='Prey')
    ax1.plot(gens, [s['n_predator'] for s in stats_history],
            c='#cb181d', lw=1.5, label='Predator')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population count')
    ax1.set_title('(A) Population dynamics')
    ax1.legend(loc='center right')

    # Panel B: Size by type
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gens, [s['mean_N_prey'] for s in stats_history],
            c='#2171b5', lw=1.5, label='Prey $N$')
    ax2.plot(gens, [s['mean_N_pred'] for s in stats_history],
            c='#cb181d', lw=1.5, label='Predator $N$')
    ax2.axhline(y=40, color='gray', linestyle='--', alpha=0.5, lw=0.8, label='$N_{ceiling}$')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Mean colony size')
    ax2.set_title('(B) Size differentiation')
    ax2.legend(loc='center right')

    # Panel C: Coherence by type
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gens, [s['mean_C_prey'] for s in stats_history],
            c='#2171b5', lw=1.5, label='Prey $C$')
    ax3.plot(gens, [s['mean_C_pred'] for s in stats_history],
            c='#cb181d', lw=1.5, label='Predator $C$')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, lw=0.8)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Mean coherence')
    ax3.set_title('(C) Coherence differentiation')
    ax3.legend(loc='center right')

    # Panel D: Phase portrait (predator vs prey count)
    ax4 = fig.add_subplot(gs[1, 1])
    n_pred = [s['n_predator'] for s in stats_history]
    n_prey = [s['n_prey'] for s in stats_history]
    ax4.plot(n_prey, n_pred, c='gray', lw=0.3, alpha=0.5)
    ax4.scatter(n_prey[0], n_pred[0], c='green', s=80, zorder=5, label='Start')
    ax4.scatter(n_prey[-1], n_pred[-1], c='red', s=80, zorder=5, label='End', marker='*')
    ax4.set_xlabel('Prey count')
    ax4.set_ylabel('Predator count')
    ax4.set_title('(D) Population phase portrait')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_dynamics.pdf')
    plt.savefig(output_dir / 'fig2_dynamics.png', dpi=300)
    plt.close()
    print("Saved Figure 2: Dynamics")

def figure_3_distributions(snapshots, output_dir):
    """Figure 3: Final state distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    final = snapshots[-1]
    is_pred = final['C'] > 0.5

    # Panel A: Size distribution
    ax1 = axes[0]
    ax1.hist(final['N'][~is_pred], bins=40, alpha=0.7, color='#2171b5',
            label=f'Prey (n={np.sum(~is_pred)})', density=True)
    ax1.hist(final['N'][is_pred], bins=20, alpha=0.7, color='#cb181d',
            label=f'Predator (n={np.sum(is_pred)})', density=True)
    ax1.axvline(x=40, color='gray', linestyle='--', alpha=0.8, lw=1, label='$N_{ceiling}$')
    ax1.set_xlabel('Colony size $N$')
    ax1.set_ylabel('Density')
    ax1.set_title('(A) Size distribution')
    ax1.legend(loc='upper left')

    # Panel B: Coherence distribution
    ax2 = axes[1]
    ax2.hist(final['C'][~is_pred], bins=40, alpha=0.7, color='#2171b5',
            label='Prey', density=True)
    ax2.hist(final['C'][is_pred], bins=20, alpha=0.7, color='#cb181d',
            label='Predator', density=True)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.8, lw=1, label='$C_{threshold}$')
    ax2.set_xlabel('Coherence $C$')
    ax2.set_ylabel('Density')
    ax2.set_title('(B) Coherence distribution')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_distributions.pdf')
    plt.savefig(output_dir / 'fig3_distributions.png', dpi=300)
    plt.close()
    print("Saved Figure 3: Distributions")

def figure_4_ceiling_sweep(output_dir):
    """Figure 4: Effect of N_ceiling on size differentiation."""
    from overnight_run import step

    ceilings = [20, 30, 40, 50, 60, 80, 100]
    results = []

    for ceil in ceilings:
        params = PARAMS.copy()
        params['N_ceiling'] = ceil

        rng = default_rng(42)
        N, C, R = initialize(params, rng)

        for gen in range(5000):
            N, C, R = step(N, C, R, params, rng)

        is_pred = classify_population(N, C)
        results.append({
            'ceiling': ceil,
            'N_pred': np.mean(N[is_pred]) if np.any(is_pred) else 0,
            'N_prey': np.mean(N[~is_pred]) if np.any(~is_pred) else 0,
            'diff': np.mean(N[~is_pred]) - np.mean(N[is_pred]) if np.any(is_pred) else 0,
        })

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(ceilings, [r['N_prey'] for r in results], 'o-', c='#2171b5',
           lw=2, markersize=8, label='Prey $N$')
    ax.plot(ceilings, [r['N_pred'] for r in results], 's-', c='#cb181d',
           lw=2, markersize=8, label='Predator $N$')

    # Identity line for predator
    ax.plot(ceilings, ceilings, '--', c='gray', alpha=0.5, lw=1, label='$N = N_{ceiling}$')

    ax.set_xlabel('$N_{ceiling}$ (coherence size limit)')
    ax.set_ylabel('Mean colony size')
    ax.set_title('Size differentiation vs. coherence constraint')
    ax.legend(loc='upper left')

    # Annotate key point
    ax.annotate(f'Prey size\ninvariant', xy=(60, 183), fontsize=8,
               ha='left', va='center', color='#2171b5')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_ceiling_sweep.pdf')
    plt.savefig(output_dir / 'fig4_ceiling_sweep.png', dpi=300)
    plt.close()
    print("Saved Figure 4: Ceiling sweep")

def main():
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'figures' / f'paper_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print("Running main simulation...")

    # Run main simulation
    params = PARAMS.copy()
    snapshots, stats_history = run_simulation(params, generations=20000)

    # Generate figures
    print("\nGenerating figures...")
    figure_1_phase_evolution(snapshots, output_dir)
    figure_2_dynamics(stats_history, output_dir)
    figure_3_distributions(snapshots, output_dir)
    figure_4_ceiling_sweep(output_dir)

    print(f"\nAll figures saved to {output_dir}")

    # Print final statistics
    final = stats_history[-1]
    print(f"\nFinal state (gen {final['gen']}):")
    print(f"  Predators: n={final['n_predator']}, N={final['mean_N_pred']:.1f}, C={final['mean_C_pred']:.2f}")
    print(f"  Prey:      n={final['n_prey']}, N={final['mean_N_prey']:.1f}, C={final['mean_C_prey']:.2f}")
    print(f"  Size diff: {final['mean_N_prey'] - final['mean_N_pred']:.1f}")

if __name__ == '__main__':
    main()
