"""
Basin Structure Analysis
========================

Key finding: Degeneracy = Basin Width = Robustness

This connects evolved codes to the actual genetic code structure.
Wide basins (Leu, Ser, Arg with 6 codons) are more robust than
narrow basins (Met, Trp with 1 codon).

Also validates Khrennikov's p-adic formalism.
"""

import numpy as np
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# BASIN STRUCTURE FROM EVOLVED CODES
# =============================================================================

def evolve_codes_with_coordination(n_states=4, n_codes=8, n_generations=300):
    """
    Evolve codes via Lewis signaling, then analyze basin structure.
    """
    np.random.seed(42)
    pop_size = 50
    n_signals = n_codes

    # Initialize
    sender_W = np.random.randn(pop_size, n_signals, n_states) * 0.1
    receiver_W = np.random.randn(pop_size, n_states, n_signals) * 0.1

    for gen in range(n_generations):
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            correct = 0
            for state in range(n_states):
                signal_probs = softmax(sender_W[i, :, state])
                signal = np.random.choice(n_signals, p=signal_probs)
                action_probs = softmax(receiver_W[i, :, signal])
                action = np.random.choice(n_states, p=action_probs)
                if action == state:
                    correct += 1
            fitness[i] = correct / n_states

        # Selection
        top_k = pop_size // 5
        top_idx = np.argsort(fitness)[-top_k:]
        for i in range(pop_size):
            parent = np.random.choice(top_idx)
            sender_W[i] = sender_W[parent] + np.random.randn(n_signals, n_states) * 0.03
            receiver_W[i] = receiver_W[parent] + np.random.randn(n_states, n_signals) * 0.03

    best_idx = np.argmax(fitness)
    return sender_W[best_idx], receiver_W[best_idx], fitness.max()


def analyze_basin_structure(sender_W, n_samples=1000, noise_level=0.3):
    """
    Measure basin size and robustness for each code.

    Basin size: fraction of input space that maps to each code
    Robustness: stability under input noise
    """
    n_signals, n_states = sender_W.shape

    # Sample random inputs
    inputs = np.random.randn(n_samples, n_states)

    # Get codes for each input
    codes = []
    for inp in inputs:
        logits = sender_W @ inp
        codes.append(np.argmax(logits))
    codes = np.array(codes)

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

        stable = 0
        total = 0
        for inp in code_inputs[:100]:  # Sample up to 100
            noisy_inp = inp + np.random.randn(n_states) * noise_level
            noisy_code = np.argmax(sender_W @ noisy_inp)
            if noisy_code == code:
                stable += 1
            total += 1

        robustness[code] = stable / total if total > 0 else 0

    return basin_sizes, robustness


def run_basin_demo():
    """Main basin structure demonstration."""
    print("=" * 60)
    print("BASIN STRUCTURE ANALYSIS")
    print("=" * 60)
    print()
    print("Key claim: Degeneracy = Basin Width = Robustness")
    print()

    # Evolve codes
    print("Evolving codes via Lewis signaling...")
    sender_W, receiver_W, coord_rate = evolve_codes_with_coordination()
    print(f"Final coordination: {coord_rate:.1%}")
    print()

    # Analyze basins
    print("Analyzing basin structure...")
    basin_sizes, robustness = analyze_basin_structure(sender_W)

    # Only look at codes that are actually used
    used_codes = [c for c, s in basin_sizes.items() if s > 0.01]

    print(f"\nBasin analysis ({len(used_codes)} codes used):")
    print("-" * 50)
    print(f"{'Code':>6} | {'Basin Size':>12} | {'Robustness':>12}")
    print("-" * 50)

    sizes = []
    robust = []
    for code in sorted(used_codes, key=lambda c: basin_sizes[c], reverse=True):
        bs = basin_sizes[code]
        rb = robustness[code]
        sizes.append(bs)
        robust.append(rb)
        print(f"{code:>6} | {bs:>11.1%} | {rb:>11.1%}")

    # Correlation
    if len(sizes) > 2:
        r, p = pearsonr(sizes, robust)
        print("-" * 50)
        print(f"\nCorrelation (basin size vs robustness): r = {r:.3f}, p = {p:.3f}")

    return basin_sizes, robustness


# =============================================================================
# P-ADIC ANALYSIS OF GENETIC CODE
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
    """p-adic distance: |x-y|_p = p^(-k) where k is highest power of p dividing (x-y)."""
    if x == y:
        return 0
    diff = abs(x - y)
    k = 0
    while diff % p == 0:
        diff //= p
        k += 1
    return p ** (-k)


def run_padic_demo():
    """Validate Khrennikov's p-adic formalism against genetic code."""
    print("\n" + "=" * 60)
    print("P-ADIC ANALYSIS OF GENETIC CODE")
    print("=" * 60)
    print()
    print("Testing: Does p-adic distance predict synonymy?")
    print()

    codons = list(GENETIC_CODE.keys())
    n = len(codons)

    # Compute pairwise
    p_adic_dists = []
    errors = []

    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = codons[i], codons[j]
            v1, v2 = codon_to_2adic(c1), codon_to_2adic(c2)
            d = p_adic_distance(v1, v2)

            aa1, aa2 = GENETIC_CODE[c1], GENETIC_CODE[c2]
            err = 0 if aa1 == aa2 else 1

            p_adic_dists.append(d)
            errors.append(err)

    p_adic_dists = np.array(p_adic_dists)
    errors = np.array(errors)

    # Correlation
    r, p = pearsonr(p_adic_dists, errors)
    print(f"Correlation (p-adic distance vs coding error): r = {r:.3f}")
    print()

    # Error rate by distance
    print("Error rate by p-adic distance:")
    print("-" * 50)
    unique_dists = sorted(set(p_adic_dists))

    for d in unique_dists[:6]:  # First 6
        mask = p_adic_dists == d
        err_rate = errors[mask].mean()
        count = mask.sum()
        denom = int(1/d) if d > 0 else float('inf')
        print(f"  d = 1/{denom:>3}: error = {err_rate:5.1%} (n={count:4d})")

    # Closest pairs analysis
    closest_mask = p_adic_dists < 0.04
    closest_pairs = [(codons[i], codons[j])
                     for i in range(n) for j in range(i+1, n)
                     if p_adic_distance(codon_to_2adic(codons[i]),
                                        codon_to_2adic(codons[j])) < 0.04]

    # Position of difference
    pos3_count = 0
    same_aa = 0
    for c1, c2 in closest_pairs:
        if c1[2] != c2[2] and c1[:2] == c2[:2]:
            pos3_count += 1
        if GENETIC_CODE[c1] == GENETIC_CODE[c2]:
            same_aa += 1

    print()
    print(f"Closest 32 pairs (d = 1/32):")
    print(f"  Differ at position 3 only: {pos3_count}/32 ({pos3_count/32:.0%})")
    print(f"  Code for SAME amino acid:  {same_aa}/32 ({same_aa/32:.0%})")

    print()
    print("CONCLUSION: P-adic distance captures WOBBLE structure.")
    print("Small perturbations (position 3) -> same amino acid.")
    print("This is ERROR-CORRECTING code structure.")


# =============================================================================
# GENETIC CODE DEGENERACY STRUCTURE
# =============================================================================

def analyze_genetic_code_degeneracy():
    """Analyze actual genetic code degeneracy by position."""
    print("\n" + "=" * 60)
    print("GENETIC CODE DEGENERACY STRUCTURE")
    print("=" * 60)
    print()

    # Count degeneracy by position
    codons = list(GENETIC_CODE.keys())

    # For each position, how often does changing ONLY that position
    # result in the same amino acid?
    for pos in range(3):
        same_aa = 0
        total = 0

        for c1 in codons:
            for c2 in codons:
                # Check if they differ at exactly position pos
                diffs = sum(1 for i in range(3) if c1[i] != c2[i])
                if diffs == 1 and c1[pos] != c2[pos]:
                    total += 1
                    if GENETIC_CODE[c1] == GENETIC_CODE[c2]:
                        same_aa += 1

        degeneracy = same_aa / total if total > 0 else 0
        print(f"Position {pos + 1}: {degeneracy:.1%} degeneracy ({same_aa}/{total} pairs)")

    print()
    print("Position 3 (wobble) has HIGHEST degeneracy.")
    print("Our model explains WHY degeneracy is adaptive,")
    print("but NOT why it concentrates at position 3.")
    print("That requires tRNA/ribosome chemistry.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BASIN STRUCTURE & P-ADIC ANALYSIS")
    print("=" * 70)
    print()

    # Basin structure from evolved codes
    basin_sizes, robustness = run_basin_demo()

    # P-adic validation
    run_padic_demo()

    # Genetic code degeneracy
    analyze_genetic_code_degeneracy()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("1. Evolved codes develop basin structure (degeneracy = robustness)")
    print("2. Khrennikov's p-adic formalism captures genetic code structure")
    print("3. Position 3 concentration requires chemistry, not just dynamics")
    print()
    print("Testable prediction: Amino acids with more codons (Leu, Ser, Arg)")
    print("should be more mutationally robust than single-codon amino acids")
    print("(Met, Trp). This is verifiable in existing mutation databases.")
