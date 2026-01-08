#!/usr/bin/env python3
"""
Generate figure showing discretization via substrate competition.
Shows winner-take-most dynamics and pattern reproducibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from chemistry_sim import (
    ChemicalVesicleArray, create_spatial_gradient, substrate_competition
)

def generate_bimodality_figure():
    """Generate figure showing discretization via substrate competition."""

    # Collect data across multiple runs
    winner_margins = []  # max - second_max for each code
    all_max_values = []
    all_other_values = []
    pattern_reproducibility = []

    for seed in range(5):
        array = ChemicalVesicleArray(
            n_rings=2, n_species=20, n_reactions=40,
            n_outputs=8, seed=seed, coupling_strength=0.1,
            hill_coeff=1.0, s0=1.0
        )

        for env_id in range(16):
            env_inputs = create_spatial_gradient(array.coords, env_id)

            # Run multiple trials for same environment to check reproducibility
            trial_codes = []
            for trial in range(3):
                code, _ = array.run(env_inputs, t_span=(0, 50), n_points=200,
                                   trial_seed=seed * 10000 + env_id * 100 + trial)
                trial_codes.append(code)

                # Code is 1D array of output channel values (colony average)
                sorted_outputs = np.sort(code)[::-1]
                margin = sorted_outputs[0] - sorted_outputs[1]
                winner_margins.append(margin)
                all_max_values.append(sorted_outputs[0])
                all_other_values.extend(sorted_outputs[1:])

            # Check reproducibility: do trials produce same winner pattern?
            for i in range(len(trial_codes)):
                for j in range(i+1, len(trial_codes)):
                    code1 = trial_codes[i]
                    code2 = trial_codes[j]
                    # Compare winner indices
                    winner1 = np.argmax(code1)
                    winner2 = np.argmax(code2)
                    match = 1.0 if winner1 == winner2 else 0.0
                    pattern_reproducibility.append(match)

    winner_margins = np.array(winner_margins)
    all_max_values = np.array(all_max_values)
    all_other_values = np.array(all_other_values)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Winner vs Losers distribution
    ax = axes[0]
    bins = np.linspace(0, max(all_max_values.max(), all_other_values.max()) * 1.1, 40)
    ax.hist(all_max_values, bins=bins, density=True, alpha=0.7,
            color='#d62728', edgecolor='black', label='Winners (max channel)')
    ax.hist(all_other_values, bins=bins, density=True, alpha=0.5,
            color='#1f77b4', edgecolor='black', label='Losers (other channels)')
    ax.set_xlabel('Output Value (after substrate competition)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(A) Winner-Take-Most Dynamics', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Compute separation
    winner_mean = np.mean(all_max_values)
    loser_mean = np.mean(all_other_values)
    separation = winner_mean / (loser_mean + 1e-10)
    ax.text(0.95, 0.65, f'Winner mean: {winner_mean:.3f}\nLoser mean: {loser_mean:.3f}\nSeparation: {separation:.1f}×',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: Winner margin distribution
    ax = axes[1]
    ax.hist(winner_margins, bins=40, density=True, alpha=0.7,
            color='#2ca02c', edgecolor='black')
    median_margin = np.median(winner_margins)
    ax.axvline(median_margin, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_margin:.3f}')
    ax.set_xlabel('Winner Margin (max − 2nd max)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(B) Discretization Strength', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Compute discretization metrics
    clear_winners = np.sum(winner_margins > 0.01) / len(winner_margins)
    repro_mean = np.mean(pattern_reproducibility)
    ax.text(0.95, 0.65, f'Clear winners: {clear_winners*100:.0f}%\n(margin > 0.01)\n\nWinner reproducibility:\n{repro_mean*100:.0f}% across trials',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figures/fig_bimodality.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_bimodality.png', dpi=300, bbox_inches='tight')
    print("Saved figures/fig_bimodality.pdf")
    plt.close()

    # Print summary stats
    print(f"\nDiscretization metrics:")
    print(f"  Winner mean: {winner_mean:.4f}")
    print(f"  Loser mean: {loser_mean:.4f}")
    print(f"  Separation ratio: {separation:.1f}×")
    print(f"  Median winner margin: {median_margin:.4f}")
    print(f"  Clear winners (margin > 0.01): {clear_winners*100:.1f}%")
    print(f"  Winner reproducibility: {repro_mean*100:.1f}%")

if __name__ == '__main__':
    generate_bimodality_figure()
