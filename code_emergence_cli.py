#!/usr/bin/env python3
"""
Coupled Compartment Code Generator v7 - CLI Interface
======================================================

Command-line interface for the code emergence simulation.

Usage:
    python code_emergence_cli.py --scale=medium --mode=systematic --trials=10

For programmatic use, import code_emergence_core instead.

Author: Ian Todd
Date: January 2026
"""

import argparse
import numpy as np
import sys

# Import core module
import code_emergence_core as core


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coupled Compartment Code Generator - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  main        Run main experiment (default)
  sweep       20-seed robustness test
  ablate      Channel-blocked + readout-perturbed controls
  bimodal     Check discretization quality
  noclip      Prove bimodality without clip()
  nocenter    Prove bimodality without mean-centering
  coupling    Coupling sweep (manifold expansion test)
  systematic  All 32 configs × N trials + decoding accuracy
  codebook    Print full encoding table
  full        Complete validation suite

Scales:
  small       19 vesicles × 64D   (fast, ~1 min)
  medium      61 vesicles × 128D  (default paper results)
  large       127 vesicles × 256D (slow, ~30 min)
  massive     169 vesicles × 512D (overnight, ~2 hrs)
        """
    )
    parser.add_argument("--scale", choices=["small", "medium", "large", "massive"],
                        default="small", help="Simulation scale")
    parser.add_argument("--mode", choices=["main", "sweep", "ablate", "bimodal", "noclip",
                                           "nocenter", "coupling", "systematic", "codebook", "full"],
                        default="main", help="Run mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials (for systematic)")
    return parser.parse_args()


def run_seed_sweep(n_seeds: int = 20):
    """Run experiment across multiple seeds."""
    print("=" * 70)
    print("SEED SWEEP (Robustness Analysis)")
    print("=" * 70)
    print(f"Running {n_seeds} seeds...\n")

    results = []
    for seed in range(n_seeds):
        r = core.run_experiment(seed=seed, verbose=False)
        results.append(r)
        print(f"  Seed {seed}: separation={r['separation_ratio']:.1f}x, corr={r['correlation']:.3f}")

    separations = [r['separation_ratio'] for r in results]
    correlations = [r['correlation'] for r in results]

    print(f"\n" + "=" * 70)
    print("SEED SWEEP SUMMARY")
    print("=" * 70)
    print(f"Separation ratio: {np.mean(separations):.1f} ± {np.std(separations):.1f}")
    print(f"Correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")

    n_sep_pass = sum(s > 1.5 for s in separations)
    n_corr_pass = sum(c > 0.2 for c in correlations)

    print(f"\nSuccess rate:")
    print(f"  Separation > 1.5: {n_sep_pass}/{n_seeds} ({100*n_sep_pass/n_seeds:.0f}%)")
    print(f"  Correlation > 0.2: {n_corr_pass}/{n_seeds} ({100*n_corr_pass/n_seeds:.0f}%)")

    return results


def run_bimodality_check(seed: int = 42):
    """Check that mass-action creates bimodal outputs."""
    print("=" * 70)
    print("BIMODALITY CHECK (Mass-Action Digitization)")
    print("=" * 70)
    np.random.seed(seed)

    encoder = core.EncoderArray()
    all_readouts = []

    for config_idx in range(2**core.N_ENV_BITS):
        config_bits = tuple(int(x) for x in f"{config_idx:0{core.N_ENV_BITS}b}")
        encoder.reset()
        for cycle in range(core.N_CYCLES):
            stimulus_field = core.generate_stimulus_field(config_bits, cycle)
            for _ in range(70):
                encoder.step(stimulus_field)
            for _ in range(20):
                encoder.step(stimulus_field)
                for v in encoder.vesicles:
                    all_readouts.extend(v.readout.tolist())

    readouts = np.array(all_readouts)

    near_negative = np.sum(readouts < -0.5)
    near_positive = np.sum(readouts > 0.5)
    near_zero = np.sum(np.abs(readouts) < 0.3)
    bimodal_fraction = (near_negative + near_positive) / len(readouts)

    print(f"  Total samples: {len(readouts)}")
    print(f"  Values < -0.5: {100*near_negative/len(readouts):.1f}%")
    print(f"  Values > +0.5: {100*near_positive/len(readouts):.1f}%")
    print(f"  Values near 0: {100*near_zero/len(readouts):.1f}%")
    print(f"\n  Bimodal fraction (|x| > 0.5): {100*bimodal_fraction:.1f}%")

    if bimodal_fraction > 0.6:
        print("  ✓ BIMODAL: Mass-action creates discrete states")
    else:
        print("  ✗ Not strongly bimodal - may need parameter tuning")

    return readouts, bimodal_fraction


def run_no_centering_ablation(seed: int = 42):
    """Test if discretization persists without mean-centering."""
    print("=" * 70)
    print("NO MEAN-CENTERING ABLATION")
    print("=" * 70)
    print("Testing if discretization persists without mean-centering...\n")

    # Baseline with centering
    core.USE_MEAN_CENTERING = True
    np.random.seed(seed)
    baseline = core.run_experiment(seed=seed, verbose=False)

    # Without centering
    core.USE_MEAN_CENTERING = False
    np.random.seed(seed)
    ablated = core.run_experiment(seed=seed, verbose=False)

    # Reset
    core.USE_MEAN_CENTERING = True

    print(f"  {'Metric':<25} {'With centering':<20} {'Without centering':<20}")
    print("-" * 70)
    print(f"  {'Separation ratio':<25} {baseline['separation_ratio']:.1f}x{'':<12} {ablated['separation_ratio']:.1f}x")
    print(f"  {'Correlation':<25} {baseline['correlation']:.3f}{'':<15} {ablated['correlation']:.3f}")
    print(f"  {'D_eff':<25} {baseline['d_eff']:.2f}{'':<16} {ablated['d_eff']:.2f}")

    # Check bimodality
    np.random.seed(seed)
    core.USE_MEAN_CENTERING = False
    encoder = core.EncoderArray()
    all_readouts = []

    for config_idx in range(2**core.N_ENV_BITS):
        config_bits = tuple(int(x) for x in f"{config_idx:0{core.N_ENV_BITS}b}")
        encoder.reset()
        for cycle in range(core.N_CYCLES):
            stimulus_field = core.generate_stimulus_field(config_bits, cycle)
            for _ in range(70):
                encoder.step(stimulus_field)
            for _ in range(10):
                encoder.step(stimulus_field)
                for v in encoder.vesicles:
                    all_readouts.extend(v.readout.tolist())

    core.USE_MEAN_CENTERING = True

    readouts = np.array(all_readouts)
    near_zero = np.sum(readouts < 0.1)
    near_max = np.sum(readouts > 0.5)
    bimodal_fraction = (near_zero + near_max) / len(readouts)

    print(f"\n  Bimodality check (without centering):")
    print(f"    Values < 0.1: {100*near_zero/len(readouts):.1f}%")
    print(f"    Values > 0.5: {100*near_max/len(readouts):.1f}%")
    print(f"    Winner-take-most fraction: {100*bimodal_fraction:.1f}%")

    if ablated['separation_ratio'] > 10 and bimodal_fraction > 0.7:
        print("\n  ✓ DISCRETIZATION PERSISTS WITHOUT MEAN-CENTERING")
        print("    → Mean-centering is instrumentation, not digitizer")
    else:
        print("\n  ⚠ Results degraded - mean-centering may contribute to digitization")

    return {'baseline': baseline, 'ablated': ablated, 'bimodal_fraction': bimodal_fraction}


def run_coupling_sweep(seed: int = 42):
    """Test manifold expansion with coupling strength."""
    coupling_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    print("=" * 70)
    print("COUPLING SWEEP (Manifold Expansion Test)")
    print("=" * 70)
    print("Testing: D_eff should increase with coupling strength\n")

    original_coupling = core.COUPLING_STRENGTH
    results = []

    for kappa in coupling_values:
        core.COUPLING_STRENGTH = kappa
        r = core.run_experiment(seed=seed, verbose=False)
        results.append({
            'coupling': kappa,
            'd_eff': r['d_eff'],
            'separation': r['separation_ratio'],
            'correlation': r['correlation'],
        })
        print(f"  κ = {kappa:.2f}: D_eff = {r['d_eff']:.2f}, sep = {r['separation_ratio']:.1f}x")

    core.COUPLING_STRENGTH = original_coupling

    print(f"\n" + "-" * 70)
    print("MANIFOLD EXPANSION SUMMARY")
    print("-" * 70)

    d_effs = [r['d_eff'] for r in results]
    if d_effs[-1] > d_effs[0]:
        print(f"  ✓ D_eff increases with coupling: {d_effs[0]:.2f} → {d_effs[-1]:.2f}")
    else:
        print(f"  ✗ D_eff does not increase with coupling")

    return results


def run_channel_blocked_ablation(seed: int = 42):
    """Channel-blocked control: replace code signal with noise."""
    from collections import defaultdict

    print("=" * 70)
    print("CHANNEL-BLOCKED ABLATION")
    print("=" * 70)
    np.random.seed(seed)

    encoder = core.EncoderArray()
    receiver = core.ReceiverArray()

    env_to_attractors = defaultdict(list)

    for ep in range(core.N_EPISODES):
        env_state = np.random.randint(0, 2**core.N_ENV_BITS)
        config_bits = tuple(int(x) for x in f"{env_state:0{core.N_ENV_BITS}b}")

        encoder.reset()
        receiver.reset()

        for cycle in range(core.N_CYCLES):
            stimulus_field = core.generate_stimulus_field(config_bits, cycle)
            encoder.run_to_equilibrium(stimulus_field)
            # BLOCKED: Use random noise instead of real code
            blocked_signal = np.random.randn(2 * core.N_READOUT) * 0.3
            receiver_attractor = receiver.run_to_equilibrium(blocked_signal)

        signature = receiver.get_attractor_signature(receiver_attractor)
        env_to_attractors[env_state].append(signature)

    # Compute metrics
    env_states = [k for k, v in env_to_attractors.items() if len(v) >= 2]
    within_distances = []
    between_distances = []

    for env in env_states:
        attractors = env_to_attractors[env]
        if len(attractors) >= 2:
            for i in range(len(attractors)):
                for j in range(i + 1, len(attractors)):
                    dist = np.linalg.norm(attractors[i] - attractors[j])
                    within_distances.append(dist)

    for i, env1 in enumerate(env_states):
        for env2 in env_states[i + 1:]:
            mean1 = np.mean(env_to_attractors[env1], axis=0)
            mean2 = np.mean(env_to_attractors[env2], axis=0)
            dist = np.linalg.norm(mean1 - mean2)
            between_distances.append(dist)

    within_mean = np.mean(within_distances) if within_distances else 0
    between_mean = np.mean(between_distances) if between_distances else 0
    blocked_sep = between_mean / within_mean if within_mean > 0 else float('inf')

    print(f"  Blocked separation ratio: {blocked_sep:.2f}x")
    if blocked_sep < 10:
        print("  ✓ Channel matters: blocking destroys separation")
    else:
        print("  ✗ Warning: separation persists without real code")

    return blocked_sep


def main():
    args = parse_args()

    # Configure scale BEFORE any simulation runs
    core.configure(args.scale)

    print(f"\nScale: {args.scale} ({core.N_VESICLES} vesicles × {core.N_INTERNAL}D × {core.N_READOUT} readout)")
    print()

    if args.mode == "sweep":
        run_seed_sweep(n_seeds=20)
    elif args.mode == "ablate":
        print("\n")
        run_channel_blocked_ablation(seed=args.seed)
    elif args.mode == "bimodal":
        run_bimodality_check(seed=args.seed)
    elif args.mode == "codebook":
        core.run_codebook(seed=args.seed)
    elif args.mode == "systematic":
        core.run_systematic_trials(n_trials=args.trials, seed=args.seed, return_confusion=True)
    elif args.mode == "noclip":
        # Run no-clip test (would need to add to core)
        print("No-clip test not yet in core module")
    elif args.mode == "nocenter":
        run_no_centering_ablation(seed=args.seed)
    elif args.mode == "coupling":
        run_coupling_sweep(seed=args.seed)
    elif args.mode == "full":
        core.run_experiment(seed=args.seed)
        print("\n")
        run_bimodality_check(seed=args.seed)
        print("\n")
        run_no_centering_ablation(seed=args.seed)
        print("\n")
        run_channel_blocked_ablation(seed=args.seed)
        print("\n")
        run_coupling_sweep(seed=args.seed)
        print("\n")
        core.run_systematic_trials(n_trials=args.trials, seed=args.seed)
        print("\n")
        core.run_codebook(seed=args.seed)
    else:  # main
        results = core.run_experiment(seed=args.seed)

        print("\n" + "=" * 70)
        print("COMPLIANCE NOTES")
        print("=" * 70)
        print("Digitization mechanism: Mass-action kinetics")
        print("  - Finite substrate pool with replenishment (open system)")
        print("  - High-activity channels outcompete low ones")
        print("  - Michaelis-Menten + Hill kinetics (not tanh saturation)")
        print("  - Resource limitation enforces winner-take-all")
        print("\nNo engineered digitizer. Chemistry does the work.")


if __name__ == "__main__":
    main()
