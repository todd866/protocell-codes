#!/usr/bin/env python3
"""
Generate bimodality histogram to show outputs are discretized.
"""

import numpy as np
import matplotlib.pyplot as plt
from chemistry_sim import (
    ChemicalVesicleArray, create_spatial_gradient, substrate_competition
)

def generate_bimodality_figure():
    """Generate histogram showing bimodal distribution of boundary signals."""

    # Run multiple chemistries to collect boundary signal distributions
    all_signals = []

    for seed in range(5):  # 5 random chemistries
        array = ChemicalVesicleArray(
            n_rings=2, n_species=20, n_reactions=40,
            n_outputs=8, seed=seed, coupling_strength=0.1,
            hill_coeff=1.0, s0=1.0  # Using h=1 as per ablation findings
        )

        for env_id in range(16):
            env_inputs = create_spatial_gradient(array.coords, env_id)
            code, trajectory = array.run(env_inputs, t_span=(0, 50), n_points=200,
                                        trial_seed=seed * 1000 + env_id)

            # Collect the boundary signals (steady-state values)
            # These are after substrate competition normalization
            all_signals.extend(code.flatten())

    all_signals = np.array(all_signals)

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Histogram of raw values
    ax = axes[0]
    ax.hist(all_signals, bins=50, density=True, alpha=0.7, color='#1f77b4', edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='High/Low boundary')
    ax.set_xlabel('Normalized Output Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(A) Output Distribution After\nSubstrate Competition', fontsize=12, fontweight='bold')
    ax.legend()

    # Compute saturation ratio
    saturated = np.sum(np.abs(all_signals - 0.5) > 0.3) / len(all_signals)
    ax.text(0.05, 0.95, f'Saturated: {saturated*100:.0f}%\n(values near 0 or 1)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Histogram of centered/signed values (what we use for "bits")
    centered = all_signals - np.mean(all_signals)

    ax = axes[1]
    ax.hist(centered, bins=50, density=True, alpha=0.7, color='#2ca02c', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (bit boundary)')

    # Compute bimodality metrics
    positive = np.sum(centered > 0.05) / len(centered)
    negative = np.sum(centered < -0.05) / len(centered)
    middle = np.sum(np.abs(centered) <= 0.05) / len(centered)

    ax.set_xlabel('Centered Output Value (deviation from mean)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(B) Mean-Centered Distribution\n(Bit = sign of value)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.text(0.05, 0.95, f'Positive: {positive*100:.0f}%\nNegative: {negative*100:.0f}%\nTransition: {middle*100:.0f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('figures/fig_bimodality.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig_bimodality.png', dpi=300, bbox_inches='tight')
    print("Saved figures/fig_bimodality.pdf")
    plt.close()

    # Also compute Sarle's bimodality coefficient
    from scipy import stats as sp_stats
    n = len(centered)
    skewness = sp_stats.skew(centered)
    kurtosis = sp_stats.kurtosis(centered, fisher=False)  # Excess kurtosis
    BC = (skewness**2 + 1) / kurtosis
    print(f"\nBimodality coefficient (Sarle's BC): {BC:.3f}")
    print(f"  BC > 0.555 indicates bimodality")
    print(f"  Saturation ratio: {saturated*100:.1f}%")
    print(f"  Transition zone occupancy: {middle*100:.1f}%")

if __name__ == '__main__':
    generate_bimodality_figure()
