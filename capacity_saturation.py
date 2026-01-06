#!/usr/bin/env python3
"""
Capacity Saturation Experiment
==============================

Tests whether substrate competition has an intrinsic capacity limit.

Key question: If we throw N environments at the system, how many
distinguishable output attractors emerge?

If output diversity SATURATES (flattens), that ceiling is the prediction.
If output scales linearly with input, this is just a fancy hash function.

The saturation point would explain why ~20 amino acids: not because we
engineered it, but because the dynamics can't sustain more.

Usage:
    python capacity_saturation.py [--quick]

Author: Ian Todd
Date: January 2026
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from datetime import datetime

import code_emergence_core as core


def generate_environment_configs(n_envs: int, n_bits: int = None) -> list:
    """Generate n_envs distinct environment configurations.

    For n_envs <= 2^n_bits, use all combinations.
    For n_envs > 2^n_bits, use random continuous values.

    Note: n_bits is fixed to 5 to match N_ENV_BITS in code_emergence_core.
    """
    if n_bits is None:
        # FIXED to 5 bits to match code_emergence_core.N_ENV_BITS
        n_bits = 5

    if n_envs <= 2**n_bits:
        # Use binary configurations
        configs = []
        for i in range(n_envs):
            bits = tuple(int(x) for x in format(i, f'0{n_bits}b'))
            configs.append(bits)
        return configs
    else:
        # Use continuous random configurations
        np.random.seed(42)
        return [tuple(np.random.rand(n_bits)) for _ in range(n_envs)]


def collect_codes_for_environments(env_configs: list, n_repeats: int = 3) -> dict:
    """Run encoder for each environment, collect output codes."""

    core.configure("medium")
    encoder = core.EncoderArray()

    env_to_codes = {}

    for env_idx, config in enumerate(env_configs):
        codes = []
        for repeat in range(n_repeats):
            encoder.reset()

            # Run through temporal cycles
            interface_codes = []
            for cycle in range(core.N_CYCLES):
                # Generate stimulus - handle both binary and continuous configs
                if all(isinstance(b, int) or b in [0, 1, 0.0, 1.0] for b in config):
                    stimulus = core.generate_stimulus_field(config, cycle)
                else:
                    # Continuous config - create gradient field
                    stimulus = np.zeros(core.N_VESICLES)
                    for i, val in enumerate(config):
                        phase = 2 * np.pi * i / len(config)
                        for v in range(core.N_VESICLES):
                            x, y = v % 10, v // 10  # Approximate grid position
                            stimulus[v] += val * np.sin(phase + 0.1 * (x + y))
                    stimulus = (stimulus - stimulus.min()) / (stimulus.max() - stimulus.min() + 1e-6)

                pattern = encoder.run_to_equilibrium(stimulus)
                code = encoder.emit_code(pattern)
                interface_codes.append(code.copy())

            full_code = np.concatenate(interface_codes)
            codes.append(full_code)

        env_to_codes[env_idx] = codes

    return env_to_codes


def count_distinguishable_attractors(env_to_codes: dict, method: str = 'kmeans_elbow') -> dict:
    """Count how many distinguishable output clusters exist.

    Methods:
    - 'kmeans_elbow': Find elbow in inertia curve
    - 'dbscan': Density-based clustering
    - 'silhouette': Maximize silhouette score
    - 'centroid_distance': Count centroids with sufficient separation
    """

    # Collect all codes
    all_codes = []
    labels = []
    for env_idx, codes in env_to_codes.items():
        for code in codes:
            all_codes.append(code)
            labels.append(env_idx)

    X = np.array(all_codes)
    n_envs = len(env_to_codes)

    results = {}

    # Method 1: K-means with elbow detection
    if method in ['kmeans_elbow', 'all']:
        inertias = []
        k_range = range(2, min(n_envs + 5, 50))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Find elbow using second derivative
        inertias = np.array(inertias)
        d1 = np.diff(inertias)
        d2 = np.diff(d1)
        elbow_idx = np.argmax(d2) + 2  # +2 because of double diff
        k_elbow = list(k_range)[elbow_idx] if elbow_idx < len(k_range) else n_envs

        results['kmeans_elbow'] = k_elbow
        results['inertias'] = inertias.tolist()
        results['k_range'] = list(k_range)

    # Method 2: Silhouette score optimization
    if method in ['silhouette', 'all']:
        best_k = 2
        best_score = -1
        for k in range(2, min(n_envs + 1, 40)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(X, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        results['silhouette_k'] = best_k
        results['silhouette_score'] = best_score

    # Method 3: Centroid separation
    if method in ['centroid_distance', 'all']:
        # Compute centroid for each environment
        centroids = []
        for env_idx in sorted(env_to_codes.keys()):
            centroid = np.mean(env_to_codes[env_idx], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(centroids)

        # Count clusters with sufficient separation
        # Use hierarchical clustering with distance threshold
        from scipy.cluster.hierarchy import linkage, fcluster
        if len(centroids) > 1:
            Z = linkage(centroids, method='average')
            # Threshold at 10% of max distance
            threshold = 0.1 * np.max(distances) if len(distances) > 0 else 1.0
            clusters = fcluster(Z, threshold, criterion='distance')
            n_distinct = len(set(clusters))
        else:
            n_distinct = 1

        results['centroid_clusters'] = n_distinct
        results['mean_centroid_distance'] = np.mean(distances) if len(distances) > 0 else 0

    # Method 4: Decoding accuracy (ground truth)
    # If we can decode back to environment with >90% accuracy, count as distinguishable
    if method in ['decoding', 'all']:
        centroids = {env: np.mean(codes, axis=0) for env, codes in env_to_codes.items()}

        correct = 0
        total = 0
        for env_idx, codes in env_to_codes.items():
            for code in codes:
                # Find nearest centroid
                dists = {e: np.linalg.norm(code - c) for e, c in centroids.items()}
                pred = min(dists, key=dists.get)
                if pred == env_idx:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        results['decoding_accuracy'] = accuracy
        results['distinguishable_by_decoding'] = n_envs if accuracy > 0.9 else int(accuracy * n_envs)

    return results


def run_saturation_experiment(env_counts: list = None, n_repeats: int = 3, quick: bool = False):
    """Run the full saturation experiment."""

    if env_counts is None:
        if quick:
            env_counts = [8, 16, 32, 48, 64]
        else:
            env_counts = [8, 16, 32, 48, 64, 96, 128, 192, 256]

    results = []

    print("=" * 70)
    print("CAPACITY SATURATION EXPERIMENT")
    print("=" * 70)
    print(f"\nTesting environment counts: {env_counts}")
    print(f"Repeats per environment: {n_repeats}")
    print()

    for n_envs in env_counts:
        print(f"\n{'='*50}")
        print(f"Testing N_environments = {n_envs}")
        print(f"{'='*50}")

        # Generate environment configurations
        configs = generate_environment_configs(n_envs)
        print(f"  Generated {len(configs)} environment configs")

        # Collect codes
        print(f"  Collecting codes...")
        env_to_codes = collect_codes_for_environments(configs, n_repeats=n_repeats)

        # Count distinguishable attractors
        print(f"  Counting distinguishable attractors...")
        attractor_counts = count_distinguishable_attractors(env_to_codes, method='all')

        result = {
            'n_environments': n_envs,
            'n_configs': len(configs),
            **attractor_counts
        }
        results.append(result)

        print(f"\n  Results for N={n_envs}:")
        print(f"    K-means elbow:        {attractor_counts.get('kmeans_elbow', 'N/A')}")
        print(f"    Silhouette optimal k: {attractor_counts.get('silhouette_k', 'N/A')}")
        print(f"    Centroid clusters:    {attractor_counts.get('centroid_clusters', 'N/A')}")
        print(f"    Decoding accuracy:    {attractor_counts.get('decoding_accuracy', 0)*100:.1f}%")

    return results


def plot_saturation_curve(results: list, output_dir: Path):
    """Plot input diversity vs output diversity."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    n_envs = [r['n_environments'] for r in results]

    # Panel A: All methods
    ax = axes[0, 0]
    if 'kmeans_elbow' in results[0]:
        ax.plot(n_envs, [r['kmeans_elbow'] for r in results], 'o-', label='K-means elbow', linewidth=2)
    if 'silhouette_k' in results[0]:
        ax.plot(n_envs, [r['silhouette_k'] for r in results], 's-', label='Silhouette optimal', linewidth=2)
    if 'centroid_clusters' in results[0]:
        ax.plot(n_envs, [r['centroid_clusters'] for r in results], '^-', label='Centroid clustering', linewidth=2)

    ax.plot(n_envs, n_envs, 'k--', alpha=0.5, label='y=x (no saturation)')
    ax.axhline(y=20, color='red', linestyle=':', alpha=0.7, label='Genetic code (20 AA)')
    ax.set_xlabel('Input environments')
    ax.set_ylabel('Distinguishable outputs')
    ax.set_title('Capacity Saturation')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Panel B: Decoding accuracy
    ax = axes[0, 1]
    accuracies = [r.get('decoding_accuracy', 0) * 100 for r in results]
    ax.plot(n_envs, accuracies, 'o-', linewidth=2, color='green')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    ax.set_xlabel('Input environments')
    ax.set_ylabel('Decoding accuracy (%)')
    ax.set_title('Decoding Performance vs Scale')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel C: Saturation ratio
    ax = axes[1, 0]
    saturation_ratio = [r.get('silhouette_k', r['n_environments']) / r['n_environments'] for r in results]
    ax.plot(n_envs, saturation_ratio, 'o-', linewidth=2, color='purple')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Input environments')
    ax.set_ylabel('Output/Input ratio')
    ax.set_title('Saturation Ratio (1.0 = linear scaling)')
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    # Panel D: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    # Find saturation point
    if 'silhouette_k' in results[0]:
        k_values = [r['silhouette_k'] for r in results]
        max_k = max(k_values)
        saturation_env = n_envs[k_values.index(max_k)]

        summary = f"""CAPACITY SATURATION SUMMARY

Input range tested: {min(n_envs)} to {max(n_envs)} environments

Key findings:
• Maximum distinguishable outputs: {max_k}
• Saturation begins at ~{saturation_env} inputs
• Saturation ratio at max: {max_k/max(n_envs):.2f}

Interpretation:
{"SATURATION DETECTED" if max_k < max(n_envs) * 0.8 else "NO CLEAR SATURATION"}

Genetic code comparison:
• Actual: 64 codons → 20 amino acids (ratio: 0.31)
• Model:  {max(n_envs)} inputs → {max_k} outputs (ratio: {max_k/max(n_envs):.2f})
"""
    else:
        summary = "Insufficient data for summary"

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'capacity_saturation.png', dpi=150)
    plt.savefig(output_dir / 'capacity_saturation.pdf')
    plt.close()

    print(f"\nFigures saved to {output_dir}")


def main():
    quick = '--quick' in sys.argv

    # Setup output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'results' / f'saturation_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results = run_saturation_experiment(quick=quick, n_repeats=2 if quick else 3)

    # Save results
    with open(output_dir / 'saturation_results.json', 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump([{k: convert(v) for k, v in r.items()} for r in results], f, indent=2)

    # Plot
    plot_saturation_curve(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    if results:
        n_envs = [r['n_environments'] for r in results]
        if 'silhouette_k' in results[0]:
            k_values = [r['silhouette_k'] for r in results]
            print(f"\nInput environments: {n_envs}")
            print(f"Output clusters:    {k_values}")
            print(f"\nSaturation ratio at max: {k_values[-1]/n_envs[-1]:.2f}")

            if k_values[-1] < n_envs[-1] * 0.5:
                print("\n*** SATURATION DETECTED ***")
                print(f"System capacity appears to saturate around {max(k_values)} distinguishable outputs")
                print("This could explain why ~20 amino acids: intrinsic capacity limit")
            else:
                print("\n*** NO CLEAR SATURATION ***")
                print("Output scales with input - may need larger scale or different parameters")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
