#!/usr/bin/env python3
"""
Reproduce Paper Results
=======================

Generates the exact numbers reported in the Discover Life manuscript.
Run this to verify all claims. NO PLACEHOLDERS - everything is computed.

Usage:
    python reproduce_paper.py [--quick]

Output: Prints tables matching the paper's Section 4 results.

Author: Ian Todd
Date: January 2026
"""

import numpy as np
import sys
import code_emergence_core as core


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def reproduce_primary_results(n_seeds: int = 5, n_repeats: int = 3):
    """Table 1: Primary Results (Section 4.1)"""
    print_separator("TABLE 4.1: PRIMARY RESULTS")

    core.configure("medium")

    # Run multiple seeds for statistics
    all_results = []
    for seed in range(n_seeds):
        r = core.run_balanced_evaluation(n_repeats=n_repeats, seed=seed, verbose=False)
        all_results.append(r)

    # Aggregate statistics
    sep_att = [r['separation_attractor'] for r in all_results]
    sep_int = [r['separation_interface'] for r in all_results]
    acc = [r['decoding_accuracy'] for r in all_results]
    repro = [r['reproducibility'] for r in all_results]
    bimod = [r['bimodality'] for r in all_results]
    corr = [r['correlation'] for r in all_results]
    deff = [r['d_eff'] for r in all_results]

    print("Metric                              Result")
    print("-" * 55)
    print(f"Decoding accuracy                   {100*np.mean(acc):.1f}% ± {100*np.std(acc):.1f}%")
    print(f"Reproducibility ({n_seeds} seeds)           {100*np.mean(repro):.1f}% ± {100*np.std(repro):.1f}%")
    print(f"Separation ratio (attractor space)  {np.mean(sep_att):.0f}× ± {np.std(sep_att):.0f}")
    print(f"Separation ratio (interface space)  {np.mean(sep_int):.0f}× ± {np.std(sep_int):.0f}")
    print(f"Env-Attractor correlation           {np.mean(corr):.2f} ± {np.std(corr):.2f}")
    print(f"Bimodality (emergent discretization) {100*np.mean(bimod):.0f}% ± {100*np.std(bimod):.0f}%")
    print(f"Effective dimensionality (D_eff)    {np.mean(deff):.1f} ± {np.std(deff):.1f}")

    return all_results


def reproduce_ablation_table(n_repeats: int = 3):
    """Table 2: Ablation Controls (Section 4.2)"""
    print_separator("TABLE 4.2: ABLATION CONTROLS")

    core.configure("medium")

    conditions = []

    # Store original settings
    orig_coupling = core.COUPLING_STRENGTH
    orig_hill = core.HILL_COEFF
    orig_centering = core.USE_MEAN_CENTERING
    orig_noise = core.INPUT_NOISE_LEVEL
    orig_cycles = core.ABLATE_TEMPORAL_CYCLES
    orig_topology = core.USE_RANDOM_TOPOLOGY
    orig_clipping = core.USE_CLIPPING

    def run_ablation(name):
        core.rebuild_topology()  # In case topology changed
        r = core.run_balanced_evaluation(n_repeats=n_repeats, seed=42, verbose=False)
        return (name, r['decoding_accuracy'], r['reproducibility'], r['separation_attractor'])

    # 1. Full model (baseline)
    conditions.append(run_ablation("Full model"))

    # 2. No coupling (κ=0)
    core.COUPLING_STRENGTH = 0.0
    conditions.append(run_ablation("No coupling (κ=0)"))
    core.COUPLING_STRENGTH = orig_coupling

    # 3. No competition (h=1)
    core.HILL_COEFF = 1.0
    conditions.append(run_ablation("No competition (h=1)"))
    core.HILL_COEFF = orig_hill

    # 4. Noisy input (±20%)
    core.INPUT_NOISE_LEVEL = 0.2
    conditions.append(run_ablation("Noisy input (±20%)"))
    core.INPUT_NOISE_LEVEL = orig_noise

    # 5. No temporal cycles
    core.ABLATE_TEMPORAL_CYCLES = True
    conditions.append(run_ablation("No temporal cycles"))
    core.ABLATE_TEMPORAL_CYCLES = orig_cycles

    # 6. Random topology
    core.USE_RANDOM_TOPOLOGY = True
    conditions.append(run_ablation("Random topology"))
    core.USE_RANDOM_TOPOLOGY = orig_topology
    core.rebuild_topology()

    # 7. No mean-centering
    core.USE_MEAN_CENTERING = False
    conditions.append(run_ablation("No mean-centering"))
    core.USE_MEAN_CENTERING = orig_centering

    # 8. No clipping (proves clipping isn't the digitizer)
    core.USE_CLIPPING = False
    conditions.append(run_ablation("No clipping"))
    core.USE_CLIPPING = orig_clipping

    # Print table
    print(f"{'Condition':<25} {'Accuracy':<12} {'Repro':<12} {'Sep. (att)':<15}")
    print("-" * 65)
    for name, acc, repro, sep in conditions:
        print(f"{name:<25} {100*acc:.1f}%{'':<6} {100*repro:.1f}%{'':<5} {sep:.0f}×")

    return conditions


def reproduce_coupling_table(n_repeats: int = 3):
    """Table 3: Manifold Expansion (Section 4.5)"""
    print_separator("TABLE 4.5: MANIFOLD EXPANSION (Coupling Sweep)")

    core.configure("medium")

    original_coupling = core.COUPLING_STRENGTH
    coupling_values = [0.0, 0.15, 0.30, 0.50]

    print(f"{'κ':<10} {'D_eff':<10} {'Sep (interface)':<18} {'Sep (attractor)':<18}")
    print("-" * 60)

    results = []
    for kappa in coupling_values:
        core.COUPLING_STRENGTH = kappa
        r = core.run_balanced_evaluation(n_repeats=n_repeats, seed=42, verbose=False)
        results.append((kappa, r['d_eff'], r['separation_interface'], r['separation_attractor']))
        print(f"{kappa:<10.2f} {r['d_eff']:<10.2f} {r['separation_interface']:.0f}×{'':<12} {r['separation_attractor']:.0f}×")

    core.COUPLING_STRENGTH = original_coupling
    return results


def reproduce_scale_table(n_repeats: int = 3):
    """Table: Scale Dependence (Section 4.4)"""
    print_separator("TABLE 4.4: SCALE DEPENDENCE")

    scales = ["small", "medium", "large"]  # Skip massive for speed
    results = []

    print(f"{'Scale':<15} {'Accuracy':<12} {'Repro':<12} {'Sep (att)':<15} {'D_eff':<10}")
    print("-" * 70)

    for scale in scales:
        core.configure(scale)
        r = core.run_balanced_evaluation(n_repeats=n_repeats, seed=42, verbose=False)
        results.append((scale, r))
        print(f"{scale:<15} {100*r['decoding_accuracy']:.1f}%{'':<6} {100*r['reproducibility']:.1f}%{'':<5} {r['separation_attractor']:.0f}×{'':<9} {r['d_eff']:.1f}")

    return results


def reproduce_mi_table(n_repeats: int = 5):
    """Information-theoretic analysis (Section 4.3)"""
    print_separator("TABLE 4.3: INFORMATION-THEORETIC CAPACITY")

    core.configure("medium")

    # Run balanced evaluation to get confusion matrix
    np.random.seed(42)
    n_envs = 32
    n_cycles = core.N_CYCLES

    encoder = core.EncoderArray()
    env_to_codes = {i: [] for i in range(n_envs)}

    for repeat in range(n_repeats):
        for env_idx in range(n_envs):
            config_bits = tuple(int(x) for x in f"{env_idx:05b}")
            encoder.reset()

            interface_codes = []
            for cycle in range(n_cycles):
                stimulus = core.generate_stimulus_field(config_bits, cycle)
                pattern = encoder.run_to_equilibrium(stimulus)
                code = encoder.emit_code(pattern)
                interface_codes.append(code.copy())

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

    # Compute accuracy
    accuracy = np.trace(confusion) / np.sum(confusion)
    error_rate = 1 - accuracy

    # Compute mutual information from confusion matrix
    # P(E, E_hat) = confusion / sum(confusion)
    joint = confusion / np.sum(confusion)
    p_true = np.sum(joint, axis=1)  # P(E)
    p_pred = np.sum(joint, axis=0)  # P(E_hat)

    # MI = sum P(e, e_hat) * log(P(e, e_hat) / (P(e) * P(e_hat)))
    mi = 0
    for i in range(n_envs):
        for j in range(n_envs):
            if joint[i, j] > 0 and p_true[i] > 0 and p_pred[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_true[i] * p_pred[j]))

    # Miller-Madow correction (approximate)
    n_samples = np.sum(confusion)
    n_nonzero = np.sum(joint > 0)
    mm_correction = (n_nonzero - 1) / (2 * n_samples * np.log(2))
    mi_corrected = mi + mm_correction

    # Fano bound
    h_error = -error_rate * np.log2(error_rate + 1e-10) - (1 - error_rate) * np.log2(1 - error_rate + 1e-10) if 0 < error_rate < 1 else 0
    fano_bound = np.log2(n_envs) - h_error - error_rate * np.log2(n_envs - 1)

    # Maximum possible
    max_mi = np.log2(n_envs)

    print(f"Number of classes (K):           {n_envs}")
    print(f"Maximum possible MI:             {max_mi:.2f} bits")
    print(f"Measured MI (Miller-Madow):      {mi_corrected:.2f} bits")
    print(f"Fano lower bound:                {fano_bound:.2f} bits")
    print(f"Capacity ratio:                  {100*mi_corrected/max_mi:.0f}%")
    print(f"Decoding accuracy:               {100*accuracy:.1f}%")
    print(f"Error rate:                      {100*error_rate:.1f}%")

    return {
        'mi': mi_corrected,
        'fano_bound': fano_bound,
        'max_mi': max_mi,
        'accuracy': accuracy,
        'confusion': confusion
    }


def main():
    # Check for quick mode
    quick = "--quick" in sys.argv
    n_seeds = 3 if quick else 5
    n_repeats = 2 if quick else 3

    print("=" * 70)
    print("REPRODUCING PAPER RESULTS")
    print("Discover Life Manuscript: Codes and Predation from Coherence")
    print("=" * 70)
    if quick:
        print("\n*** QUICK MODE: Reduced seeds/repeats for faster testing ***\n")

    reproduce_primary_results(n_seeds=n_seeds, n_repeats=n_repeats)
    reproduce_ablation_table(n_repeats=n_repeats)
    reproduce_coupling_table(n_repeats=n_repeats)
    reproduce_scale_table(n_repeats=n_repeats)
    reproduce_mi_table(n_repeats=n_repeats)

    print_separator("REPRODUCTION COMPLETE")
    print("All values above are COMPUTED, not hard-coded.")
    print("Run with --quick for faster (less accurate) results.")
    print("\nKey claims to verify:")
    print("  - Decoding accuracy > 90%")
    print("  - Separation ratio > 100× (interface), > 1000× (attractor)")
    print("  - Reproducibility > 80%")
    print("  - No-clipping ablation shows similar results (proves clip isn't digitizer)")
    print("  - h=1 (no competition) collapses accuracy/separation")


if __name__ == "__main__":
    main()
