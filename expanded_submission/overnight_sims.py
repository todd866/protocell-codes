#!/usr/bin/env python3
"""
Overnight Simulations for Expanded HeroX Submission
====================================================

Runs comprehensive simulations to strengthen the paper:
1. Lewis games at 64 states (matching genetic code)
2. Basin-robustness correlation with statistical power
3. Parameter sweeps for robustness claims
4. P-adic analysis validation

Run with: python3 overnight_sims.py

Results saved to results/ folder for paper figures.
"""

import numpy as np
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create results directory
os.makedirs('results', exist_ok=True)

def log(msg):
    """Timestamped logging."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# IMPROVED LEWIS SIGNALING GAMES
# =============================================================================

def run_lewis_game(n_states, n_signals=None, n_generations=500, pop_size=100,
                   mutation_rate=0.02, temperature=1.0, seed=None, verbose=True):
    """
    Improved Lewis signaling game with better hyperparameters.

    Key improvements:
    - Larger population (100 vs 50)
    - Lower mutation rate (0.02 vs 0.05) for stability
    - More generations (500 vs 200)
    - Temperature parameter for softmax sharpening
    - Returns detailed metrics
    """
    if seed is not None:
        np.random.seed(seed)

    if n_signals is None:
        n_signals = n_states  # Default: same number of signals as states

    # Initialize
    sender_W = np.random.randn(pop_size, n_signals, n_states) * 0.1
    receiver_W = np.random.randn(pop_size, n_states, n_signals) * 0.1

    history = []
    best_fitness_ever = 0
    best_sender = None
    best_receiver = None

    for gen in range(n_generations):
        fitness = np.zeros(pop_size)

        # Evaluate each agent
        n_trials_per_state = 5  # Multiple trials per state for robustness
        for i in range(pop_size):
            correct = 0
            total = 0

            for state in range(n_states):
                for _ in range(n_trials_per_state):
                    # Sender chooses signal
                    sender_logits = sender_W[i, :, state] / temperature
                    signal_probs = softmax(sender_logits)
                    signal = np.random.choice(n_signals, p=signal_probs)

                    # Receiver chooses action
                    receiver_logits = receiver_W[i, :, signal] / temperature
                    action_probs = softmax(receiver_logits)
                    action = np.random.choice(n_states, p=action_probs)

                    if action == state:
                        correct += 1
                    total += 1

            fitness[i] = correct / total

        # Track best ever
        if fitness.max() > best_fitness_ever:
            best_fitness_ever = fitness.max()
            best_idx = np.argmax(fitness)
            best_sender = sender_W[best_idx].copy()
            best_receiver = receiver_W[best_idx].copy()

        # Selection + mutation (elitism + tournament)
        top_k = pop_size // 5
        top_idx = np.argsort(fitness)[-top_k:]

        # Keep best unchanged (elitism)
        elite_sender = sender_W[np.argmax(fitness)].copy()
        elite_receiver = receiver_W[np.argmax(fitness)].copy()

        # Reproduce with mutation
        for i in range(pop_size - 1):
            parent = np.random.choice(top_idx)
            sender_W[i] = sender_W[parent] + np.random.randn(n_signals, n_states) * mutation_rate
            receiver_W[i] = receiver_W[parent] + np.random.randn(n_states, n_signals) * mutation_rate

        # Restore elite
        sender_W[-1] = elite_sender
        receiver_W[-1] = elite_receiver

        # Adaptive temperature (cool down over time)
        temperature = max(0.5, 1.0 - gen / (n_generations * 2))

        history.append({
            'gen': gen,
            'max_fitness': fitness.max(),
            'mean_fitness': fitness.mean(),
            'best_ever': best_fitness_ever
        })

        if verbose and (gen % 100 == 0 or gen == n_generations - 1):
            log(f"  Gen {gen:4d}: max={fitness.max():.1%}, mean={fitness.mean():.1%}, best_ever={best_fitness_ever:.1%}")

    # Analyze final code mapping
    code_map = {}
    for state in range(n_states):
        signal = np.argmax(best_sender[:, state])
        code_map[state] = signal

    unique_signals = len(set(code_map.values()))

    return {
        'final_coordination': best_fitness_ever,
        'unique_codes': unique_signals,
        'n_states': n_states,
        'code_map': code_map,
        'history': history,
        'sender_W': best_sender,
        'receiver_W': best_receiver
    }


def analyze_basin_structure(sender_W, n_samples=2000, noise_level=0.3):
    """
    Measure basin size and robustness for each code.
    """
    n_signals, n_states = sender_W.shape

    # Sample random inputs (continuous states)
    inputs = np.random.randn(n_samples, n_states)

    # Get codes for each input
    codes = np.array([np.argmax(sender_W @ inp) for inp in inputs])

    # Basin sizes
    basin_sizes = {}
    for code in range(n_signals):
        basin_sizes[code] = np.mean(codes == code)

    # Robustness: add noise, check if code changes
    robustness = {}
    for code in range(n_signals):
        code_inputs = inputs[codes == code]
        if len(code_inputs) == 0:
            robustness[code] = 0
            continue

        # Sample up to 200 inputs for this code
        sample_size = min(200, len(code_inputs))
        sample_inputs = code_inputs[:sample_size]

        stable = 0
        for inp in sample_inputs:
            noisy_inp = inp + np.random.randn(n_states) * noise_level
            noisy_code = np.argmax(sender_W @ noisy_inp)
            if noisy_code == code:
                stable += 1

        robustness[code] = stable / sample_size

    return basin_sizes, robustness


# =============================================================================
# SIMULATION 1: SCALE TO 64 STATES
# =============================================================================

def run_scaling_experiment():
    """Test Lewis games at increasing scales up to 64 states."""
    log("=" * 70)
    log("SIMULATION 1: SCALING TO 64 STATES")
    log("=" * 70)

    scales = [4, 8, 16, 32, 64]
    n_seeds = 5
    results = []

    for n_states in scales:
        log(f"\nTesting n_states = {n_states}")

        seed_results = []
        for seed in range(n_seeds):
            log(f"  Seed {seed + 1}/{n_seeds}")
            result = run_lewis_game(
                n_states=n_states,
                n_generations=800,  # More generations for larger state spaces
                pop_size=150,       # Larger population for larger state spaces
                mutation_rate=0.015,
                seed=seed * 1000 + n_states,
                verbose=False
            )
            seed_results.append({
                'seed': seed,
                'coordination': result['final_coordination'],
                'unique_codes': result['unique_codes'],
                'complete': result['unique_codes'] == n_states
            })
            log(f"    -> coordination={result['final_coordination']:.1%}, codes={result['unique_codes']}/{n_states}")

        # Aggregate
        mean_coord = np.mean([r['coordination'] for r in seed_results])
        std_coord = np.std([r['coordination'] for r in seed_results])
        complete_rate = np.mean([r['complete'] for r in seed_results])

        results.append({
            'n_states': n_states,
            'mean_coordination': mean_coord,
            'std_coordination': std_coord,
            'complete_code_rate': complete_rate,
            'seed_results': seed_results
        })

        log(f"  SUMMARY: coord={mean_coord:.1%}±{std_coord:.1%}, complete={complete_rate:.0%}")

    # Save results
    with open('results/scaling_experiment.json', 'w') as f:
        json.dump(results, f, indent=2)

    log("\nScaling results saved to results/scaling_experiment.json")
    return results


# =============================================================================
# SIMULATION 2: BASIN-ROBUSTNESS CORRELATION
# =============================================================================

def run_basin_robustness_experiment():
    """Statistical analysis of basin-robustness correlation across seeds."""
    log("=" * 70)
    log("SIMULATION 2: BASIN-ROBUSTNESS CORRELATION")
    log("=" * 70)

    n_seeds = 20
    n_states = 8
    all_correlations = []

    for seed in range(n_seeds):
        log(f"Seed {seed + 1}/{n_seeds}")

        # Evolve codes
        result = run_lewis_game(
            n_states=n_states,
            n_signals=12,  # More signals than states
            n_generations=500,
            pop_size=100,
            mutation_rate=0.02,
            seed=seed * 1000,
            verbose=False
        )

        # Analyze basin structure
        basin_sizes, robustness = analyze_basin_structure(result['sender_W'])

        # Get used codes (basin size > 1%)
        used_codes = [c for c, s in basin_sizes.items() if s > 0.01]

        if len(used_codes) >= 3:
            sizes = [basin_sizes[c] for c in used_codes]
            robust = [robustness[c] for c in used_codes]

            r, p = pearsonr(sizes, robust)
            rho, p_spearman = spearmanr(sizes, robust)

            all_correlations.append({
                'seed': seed,
                'n_codes_used': len(used_codes),
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spearman,
                'basin_sizes': {str(c): basin_sizes[c] for c in used_codes},
                'robustness': {str(c): robustness[c] for c in used_codes}
            })

            log(f"  -> {len(used_codes)} codes, r={r:.3f}, rho={rho:.3f}")
        else:
            log(f"  -> Only {len(used_codes)} codes used, skipping")

    # Aggregate
    if all_correlations:
        mean_r = np.mean([c['pearson_r'] for c in all_correlations])
        std_r = np.std([c['pearson_r'] for c in all_correlations])
        mean_rho = np.mean([c['spearman_rho'] for c in all_correlations])

        summary = {
            'n_seeds_analyzed': len(all_correlations),
            'mean_pearson_r': mean_r,
            'std_pearson_r': std_r,
            'mean_spearman_rho': mean_rho,
            'all_correlations': all_correlations
        }

        log(f"\nSUMMARY: Pearson r = {mean_r:.3f} ± {std_r:.3f} (n={len(all_correlations)})")
        log(f"         Spearman rho = {mean_rho:.3f}")
    else:
        summary = {'error': 'No valid correlations computed'}
        log("\nERROR: No valid correlations computed")

    with open('results/basin_robustness.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log("\nBasin-robustness results saved to results/basin_robustness.json")
    return summary


# =============================================================================
# SIMULATION 3: PARAMETER SWEEP
# =============================================================================

def run_parameter_sweep():
    """Sweep key parameters to test robustness of claims."""
    log("=" * 70)
    log("SIMULATION 3: PARAMETER SWEEP")
    log("=" * 70)

    results = []

    # Sweep population size
    log("\nSweeping population size...")
    for pop_size in [20, 50, 100, 200]:
        result = run_lewis_game(
            n_states=8,
            n_generations=400,
            pop_size=pop_size,
            seed=42,
            verbose=False
        )
        results.append({
            'param': 'pop_size',
            'value': pop_size,
            'coordination': result['final_coordination'],
            'unique_codes': result['unique_codes']
        })
        log(f"  pop_size={pop_size}: coord={result['final_coordination']:.1%}")

    # Sweep mutation rate
    log("\nSweeping mutation rate...")
    for mutation_rate in [0.005, 0.01, 0.02, 0.05, 0.1]:
        result = run_lewis_game(
            n_states=8,
            n_generations=400,
            pop_size=100,
            mutation_rate=mutation_rate,
            seed=42,
            verbose=False
        )
        results.append({
            'param': 'mutation_rate',
            'value': mutation_rate,
            'coordination': result['final_coordination'],
            'unique_codes': result['unique_codes']
        })
        log(f"  mutation_rate={mutation_rate}: coord={result['final_coordination']:.1%}")

    # Sweep n_signals (overcomplete vs minimal)
    log("\nSweeping n_signals (for n_states=8)...")
    for n_signals in [8, 12, 16, 24, 32]:
        result = run_lewis_game(
            n_states=8,
            n_signals=n_signals,
            n_generations=400,
            pop_size=100,
            seed=42,
            verbose=False
        )
        results.append({
            'param': 'n_signals',
            'value': n_signals,
            'coordination': result['final_coordination'],
            'unique_codes': result['unique_codes']
        })
        log(f"  n_signals={n_signals}: coord={result['final_coordination']:.1%}, codes={result['unique_codes']}")

    with open('results/parameter_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)

    log("\nParameter sweep results saved to results/parameter_sweep.json")
    return results


# =============================================================================
# SIMULATION 4: P-ADIC VALIDATION (GENETIC CODE)
# =============================================================================

GENETIC_CODE = {
    'UUU': 'Phe', 'UUC': 'Phe', 'UUA': 'Leu', 'UUG': 'Leu',
    'CUU': 'Leu', 'CUC': 'Leu', 'CUA': 'Leu', 'CUG': 'Leu',
    'AUU': 'Ile', 'AUC': 'Ile', 'AUA': 'Ile', 'AUG': 'Met',
    'GUU': 'Val', 'GUC': 'Val', 'GUA': 'Val', 'GUG': 'Val',
    'UCU': 'Ser', 'UCC': 'Ser', 'UCA': 'Ser', 'UCG': 'Ser',
    'CCU': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACU': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCU': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'UAU': 'Tyr', 'UAC': 'Tyr', 'UAA': 'Stop', 'UAG': 'Stop',
    'CAU': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
    'AAU': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
    'GAU': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
    'UGU': 'Cys', 'UGC': 'Cys', 'UGA': 'Stop', 'UGG': 'Trp',
    'CGU': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
    'AGU': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGU': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
}

def nucleotide_to_bits(n):
    """Nucleotide to 2-bit (Khrennikov encoding)."""
    mapping = {'A': (0, 0), 'U': (1, 0), 'T': (1, 0), 'G': (0, 1), 'C': (1, 1)}
    return mapping[n]

def codon_to_2adic(codon):
    """Codon to 6-bit 2-adic integer."""
    bits = []
    for n in codon:
        bits.extend(nucleotide_to_bits(n))
    return sum(b * (2**i) for i, b in enumerate(bits))

def p_adic_distance(x, y, p=2):
    """p-adic distance."""
    if x == y:
        return 0
    diff = abs(x - y)
    k = 0
    while diff % p == 0:
        diff //= p
        k += 1
    return p ** (-k)

def run_padic_analysis():
    """Comprehensive p-adic analysis of genetic code."""
    log("=" * 70)
    log("SIMULATION 4: P-ADIC ANALYSIS")
    log("=" * 70)

    codons = list(GENETIC_CODE.keys())
    n = len(codons)

    # Compute all pairwise distances and errors
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = codons[i], codons[j]
            v1, v2 = codon_to_2adic(c1), codon_to_2adic(c2)
            d = p_adic_distance(v1, v2)

            aa1, aa2 = GENETIC_CODE[c1], GENETIC_CODE[c2]
            same_aa = 1 if aa1 == aa2 else 0

            # Position of difference
            diff_positions = [pos for pos in range(3) if c1[pos] != c2[pos]]

            results.append({
                'codon1': c1,
                'codon2': c2,
                'padic_dist': d,
                'same_aa': same_aa,
                'aa1': aa1,
                'aa2': aa2,
                'diff_positions': diff_positions
            })

    # Analyze by distance
    p_adic_dists = np.array([r['padic_dist'] for r in results])
    same_aa = np.array([r['same_aa'] for r in results])

    # Correlation
    r, p = pearsonr(p_adic_dists, 1 - same_aa)  # Correlate distance with error

    log(f"Correlation (p-adic distance vs coding error): r = {r:.3f}, p = {p:.2e}")

    # Error rate by distance
    log("\nError rate by p-adic distance:")
    unique_dists = sorted(set(p_adic_dists))
    distance_analysis = []

    for d in unique_dists[:8]:
        mask = p_adic_dists == d
        error_rate = 1 - same_aa[mask].mean()
        count = mask.sum()
        denom = int(1/d) if d > 0 else float('inf')

        distance_analysis.append({
            'distance': d,
            'distance_str': f'1/{denom}',
            'error_rate': error_rate,
            'n_pairs': int(count)
        })
        log(f"  d = 1/{denom:>3}: error = {error_rate:5.1%} (n={count:4d})")

    # Closest pairs (d = 1/32)
    closest_pairs = [r for r in results if r['padic_dist'] < 0.04]
    pos3_only = sum(1 for r in closest_pairs if r['diff_positions'] == [2])
    same_aa_closest = sum(1 for r in closest_pairs if r['same_aa'])

    log(f"\nClosest pairs (d = 1/32):")
    log(f"  Differ at position 3 only: {pos3_only}/{len(closest_pairs)}")
    log(f"  Code for SAME amino acid: {same_aa_closest}/{len(closest_pairs)}")

    summary = {
        'correlation_r': r,
        'correlation_p': p,
        'distance_analysis': distance_analysis,
        'closest_pairs': {
            'total': len(closest_pairs),
            'position_3_only': pos3_only,
            'same_amino_acid': same_aa_closest
        }
    }

    with open('results/padic_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log("\nP-adic analysis saved to results/padic_analysis.json")
    return summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    log("=" * 70)
    log("OVERNIGHT SIMULATIONS FOR EXPANDED HEROX SUBMISSION")
    log("=" * 70)
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    # Run all simulations
    try:
        # 1. Scaling experiment (~30 min)
        scaling_results = run_scaling_experiment()

        # 2. Basin-robustness correlation (~20 min)
        basin_results = run_basin_robustness_experiment()

        # 3. Parameter sweep (~15 min)
        param_results = run_parameter_sweep()

        # 4. P-adic analysis (~1 min)
        padic_results = run_padic_analysis()

        log("\n" + "=" * 70)
        log("ALL SIMULATIONS COMPLETE")
        log("=" * 70)
        log(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("\nResults saved to results/ folder:")
        log("  - scaling_experiment.json")
        log("  - basin_robustness.json")
        log("  - parameter_sweep.json")
        log("  - padic_analysis.json")

        # Print key findings
        log("\n" + "=" * 70)
        log("KEY FINDINGS")
        log("=" * 70)

        if scaling_results:
            for r in scaling_results:
                log(f"  {r['n_states']:2d} states: {r['mean_coordination']:.1%} coordination, {r['complete_code_rate']:.0%} complete")

        if basin_results and 'mean_pearson_r' in basin_results:
            log(f"\n  Basin-robustness correlation: r = {basin_results['mean_pearson_r']:.3f} ± {basin_results['std_pearson_r']:.3f}")

        if padic_results:
            log(f"\n  P-adic distance vs error: r = {padic_results['correlation_r']:.3f}")
            log(f"  Closest pairs (d=1/32): {padic_results['closest_pairs']['same_amino_acid']}/{padic_results['closest_pairs']['total']} same AA")

    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
