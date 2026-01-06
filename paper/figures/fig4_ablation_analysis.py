#!/usr/bin/env python3
"""
Figure 4: Ablation Analysis
===========================

Generates:
A) Confusion matrix for 32-environment decoding
B) Output distribution showing bimodality (discretization)
C) Ablation comparison: with vs without clipping
D) Hill coefficient sweep showing competition necessity

Author: Ian Todd
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../simulation')
import code_emergence_core as core

# Consistent styling
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})


def collect_codes_and_confusion(n_repeats=3, use_clipping=True):
    """Collect all codes and build confusion matrix."""
    core.configure("medium")

    # Set clipping mode
    original_clipping = core.USE_CLIPPING
    core.USE_CLIPPING = use_clipping

    n_envs = 32
    np.random.seed(42)

    encoder = core.EncoderArray()
    env_to_codes = {i: [] for i in range(n_envs)}
    all_channel_values = []

    for repeat in range(n_repeats):
        for env_idx in range(n_envs):
            config_bits = tuple(int(x) for x in f"{env_idx:05b}")
            encoder.reset()

            interface_codes = []
            for cycle in range(core.N_CYCLES):
                stimulus = core.generate_stimulus_field(config_bits, cycle)
                pattern = encoder.run_to_equilibrium(stimulus)
                code = encoder.emit_code(pattern)
                interface_codes.append(code.copy())
                all_channel_values.extend(code.flatten())

            full_code = np.concatenate(interface_codes)
            env_to_codes[env_idx].append(full_code)

    # Compute centroids
    centroids = {e: np.mean(env_to_codes[e], axis=0) for e in range(n_envs)}

    # Build confusion matrix
    confusion = np.zeros((n_envs, n_envs))
    for true_env in range(n_envs):
        for code in env_to_codes[true_env]:
            dists = {e: np.linalg.norm(code - centroids[e]) for e in range(n_envs)}
            pred_env = min(dists, key=dists.get)
            confusion[true_env, pred_env] += 1

    # Normalize rows
    confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)

    # Compute accuracy
    accuracy = np.trace(confusion) / np.sum(confusion)

    # Restore original clipping
    core.USE_CLIPPING = original_clipping

    return confusion_norm, accuracy, np.array(all_channel_values)


def hill_coefficient_sweep(hill_values, n_repeats=2):
    """Sweep Hill coefficient to show competition necessity."""
    core.configure("medium")

    original_hill = core.HILL_COEFF
    results = []

    for h in hill_values:
        core.HILL_COEFF = h
        r = core.run_balanced_evaluation(n_repeats=n_repeats, seed=42, verbose=False)
        results.append({
            'h': h,
            'accuracy': r['decoding_accuracy'],
            'separation': r['separation_attractor'],
            'bimodality': r['bimodality']
        })
        print(f"  h={h:.1f}: accuracy={100*r['decoding_accuracy']:.1f}%, sep={r['separation_attractor']:.0f}×")

    core.HILL_COEFF = original_hill
    return results


def main():
    print("Generating Figure 4: Ablation Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # A) Confusion matrix (with clipping - baseline)
    print("  A) Computing confusion matrix...")
    confusion_clip, acc_clip, values_clip = collect_codes_and_confusion(n_repeats=3, use_clipping=True)

    ax = axes[0, 0]
    im = ax.imshow(confusion_clip, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Predicted Environment')
    ax.set_ylabel('True Environment')
    ax.set_title(f'A) Confusion Matrix (Accuracy: {100*acc_clip:.1f}%)')
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.set_yticks([0, 8, 16, 24, 31])
    plt.colorbar(im, ax=ax, shrink=0.8, label='P(pred | true)')

    # B) Output distribution (bimodality)
    print("  B) Plotting output distribution...")
    ax = axes[0, 1]

    # With clipping
    ax.hist(values_clip, bins=50, density=True, alpha=0.7, label='With clipping', color='steelblue')

    # Add vertical lines at -1, 0, 1
    ax.axvline(-1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Output Channel Value')
    ax.set_ylabel('Density')
    ax.set_title('B) Output Distribution (Bimodality)')
    ax.legend()

    # Compute bimodality statistic
    low_mode = np.mean(values_clip < -0.5)
    high_mode = np.mean(values_clip > 0.5)
    bimodality = low_mode + high_mode
    ax.text(0.02, 0.98, f'Bimodal: {100*bimodality:.0f}%\n(|x| > 0.5)',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # C) Ablation: with vs without clipping
    print("  C) Computing no-clipping ablation...")
    confusion_noclip, acc_noclip, values_noclip = collect_codes_and_confusion(n_repeats=3, use_clipping=False)

    ax = axes[1, 0]

    # Compare distributions
    ax.hist(values_clip, bins=50, density=True, alpha=0.6, label=f'With clipping ({100*acc_clip:.0f}%)', color='steelblue')
    ax.hist(values_noclip, bins=50, density=True, alpha=0.6, label=f'No clipping ({100*acc_noclip:.0f}%)', color='coral')

    ax.axvline(-1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Output Channel Value')
    ax.set_ylabel('Density')
    ax.set_title('C) Clipping Ablation: Not the Discretizer')
    ax.legend(loc='upper right')

    # Key insight annotation
    ax.text(0.02, 0.98, 'Clipping saturates extremes\nbut does NOT create bimodality.\nBimodality comes from\nsubstrate competition (h>1).',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # D) Hill coefficient sweep
    print("  D) Hill coefficient sweep...")
    hill_values = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    hill_results = hill_coefficient_sweep(hill_values, n_repeats=2)

    ax = axes[1, 1]

    hs = [r['h'] for r in hill_results]
    accs = [100*r['accuracy'] for r in hill_results]
    bimod = [100*r['bimodality'] for r in hill_results]

    ax.plot(hs, accs, 'o-', color='steelblue', linewidth=2, markersize=8, label='Decoding Accuracy')
    ax.plot(hs, bimod, 's--', color='coral', linewidth=2, markersize=8, label='Bimodality')

    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(1.05, 50, 'h=1\n(linear)', fontsize=7, color='gray')

    ax.axvline(4.0, color='green', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(4.05, 50, 'h=4\n(paper)', fontsize=7, color='green')

    ax.set_xlabel('Hill Coefficient (h)')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('D) Substrate Competition Is Essential')
    ax.legend(loc='lower right')
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig('fig4_ablation_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig4_ablation_analysis.pdf', bbox_inches='tight')
    print("\nSaved fig4_ablation_analysis.png/pdf")

    # Print summary
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print(f"With clipping:    {100*acc_clip:.1f}% accuracy")
    print(f"Without clipping: {100*acc_noclip:.1f}% accuracy")
    print(f"Difference:       {100*abs(acc_clip-acc_noclip):.1f}% → clipping is NOT the discretizer")
    print()
    print("Hill coefficient effect:")
    print(f"  h=1 (linear):   {accs[0]:.1f}% accuracy, {bimod[0]:.1f}% bimodal")
    print(f"  h=4 (paper):    {accs[4]:.1f}% accuracy, {bimod[4]:.1f}% bimodal")
    print(f"  → Substrate competition creates discretization")


if __name__ == "__main__":
    main()
