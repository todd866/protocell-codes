#!/usr/bin/env python3
"""
Chemical Reaction Network Simulation
=====================================

Replaces the reservoir (tanh neural network) with actual mass-action chemistry.
Same architecture: hexagonal vesicle array, substrate competition, boundary coupling.

This silences the "neural net toy" objection - if codes emerge from ODEs with
mass-action kinetics, that's chemistry, not machine learning.

CRITICAL: The chemistry (reaction network) is created ONCE, then tested across
all environments. Only initial conditions vary per trial. This properly tests
whether one chemistry can produce multiple distinguishable codes.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import time

# =============================================================================
# CHEMICAL REACTION NETWORK
# =============================================================================

class AutocatalyticNetwork:
    """
    A random autocatalytic reaction network with mass-action kinetics.

    Reactions of the form:
        A + B -> 2A  (autocatalysis)
        A -> B       (conversion)
        A + B -> C   (synthesis)

    This is NOT a neural network - it's ODEs with polynomial right-hand sides
    determined by stoichiometry and mass-action kinetics.
    """

    def __init__(self, n_species=15, n_reactions=30, rng=None):
        """
        Args:
            n_species: Number of molecular species
            n_reactions: Number of reactions
            rng: numpy random generator (use np.random.default_rng)
        """
        if rng is None:
            rng = np.random.default_rng()

        self.n_species = n_species
        self.n_reactions = n_reactions
        self.rng = rng

        # Generate random reaction network
        self.reactants = []  # List of (species_i, species_j) or (species_i,)
        self.products = []   # List of product species indices
        self.rates = []      # Rate constants

        for _ in range(n_reactions):
            reaction_type = rng.choice(['autocatalytic', 'conversion', 'synthesis'],
                                       p=[0.3, 0.4, 0.3])

            if reaction_type == 'autocatalytic':
                # A + B -> 2A (A catalyzes conversion of B to A)
                a = rng.integers(n_species)
                b = rng.integers(n_species)
                if b == a:
                    b = (a + 1) % n_species
                self.reactants.append((a, b))
                self.products.append((a, a))  # 2A
                self.rates.append(rng.uniform(0.1, 0.5))

            elif reaction_type == 'conversion':
                # A -> B
                a = rng.integers(n_species)
                b = rng.integers(n_species)
                self.reactants.append((a,))
                self.products.append((b,))
                self.rates.append(rng.uniform(0.05, 0.2))

            else:  # synthesis
                # A + B -> C
                a = rng.integers(n_species)
                b = rng.integers(n_species)
                c = rng.integers(n_species)
                self.reactants.append((a, b))
                self.products.append((c,))
                self.rates.append(rng.uniform(0.01, 0.1))

        # Dilution/degradation for all species (prevents blowup)
        self.dilution_rate = 0.1

        # Input species (affected by environment)
        self.input_species = list(range(min(5, n_species)))

        # Output species (for boundary signals)
        self.output_species = list(range(n_species - 8, n_species))

    def dynamics(self, t, x, env_input):
        """
        Mass-action kinetics: dx/dt = S @ v(x)
        where v(x) are reaction rates (products of concentrations)
        """
        x = np.maximum(x, 0)  # Concentrations must be non-negative

        dxdt = np.zeros(self.n_species)

        # Reaction fluxes
        for i, (reactants, products, rate) in enumerate(zip(
                self.reactants, self.products, self.rates)):

            # Flux = rate * product of reactant concentrations
            flux = rate
            for r in reactants:
                flux *= x[r]

            # Subtract from reactants
            for r in reactants:
                dxdt[r] -= flux

            # Add to products
            for p in products:
                dxdt[p] += flux

        # Dilution (all species)
        dxdt -= self.dilution_rate * x

        # Environmental input (constant influx to input species)
        for i, sp in enumerate(self.input_species):
            if i < len(env_input):
                dxdt[sp] += env_input[i] * 0.5  # Influx rate

        return dxdt

    def copy_with_perturbation(self, perturbation=0.1, rng=None):
        """Create a copy with slightly perturbed rates (same structure)."""
        if rng is None:
            rng = np.random.default_rng()

        new_net = AutocatalyticNetwork.__new__(AutocatalyticNetwork)
        new_net.n_species = self.n_species
        new_net.n_reactions = self.n_reactions
        new_net.rng = rng
        new_net.reactants = [list(r) for r in self.reactants]  # Deep copy
        new_net.products = [list(p) for p in self.products]
        new_net.rates = [r * rng.uniform(1 - perturbation, 1 + perturbation)
                         for r in self.rates]
        new_net.dilution_rate = self.dilution_rate
        new_net.input_species = self.input_species.copy()
        new_net.output_species = self.output_species.copy()
        return new_net


# =============================================================================
# VESICLE ARRAY
# =============================================================================

def hexagonal_grid(n_rings=2):
    """Generate hexagonal grid coordinates."""
    coords = [(0, 0)]
    for ring in range(1, n_rings + 1):
        for i in range(6):
            angle = i * np.pi / 3
            for j in range(ring):
                x = ring * np.cos(angle) - j * np.cos(angle + np.pi/3)
                y = ring * np.sin(angle) - j * np.sin(angle + np.pi/3)
                coords.append((x, y))
    return np.array(coords)


def get_neighbors(coords, threshold=1.5):
    """Find neighboring vesicles."""
    dists = squareform(pdist(coords))
    neighbors = defaultdict(list)
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i != j and dists[i, j] < threshold:
                neighbors[i].append(j)
    return neighbors


def substrate_competition(activations, h=4):
    """
    Substrate competition: Hill-like competitive binding.
    This IS chemistry - it's the quasi-steady-state of competitive enzyme kinetics.
    """
    powered = np.maximum(activations, 1e-10) ** h
    total = powered.sum() + 1  # +1 = unbound substrate pool
    return powered / total


# =============================================================================
# COUPLED VESICLE SIMULATION
# =============================================================================

class ChemicalVesicleArray:
    """
    Array of vesicles, each containing a chemical reaction network.
    Vesicles are coupled via diffusion of boundary species.

    IMPORTANT: The chemistry is fixed at construction time. Only initial
    conditions vary when run() is called multiple times.
    """

    def __init__(self, n_rings=2, n_species=15, n_reactions=30,
                 n_outputs=8, coupling_strength=0.1, seed=42):
        """
        Create a vesicle array with fixed chemistry.

        Args:
            n_rings: Number of rings in hexagonal grid
            n_species: Species per vesicle
            n_reactions: Reactions per vesicle
            n_outputs: Number of output channels
            coupling_strength: Diffusive coupling between neighbors
            seed: Random seed for reproducible chemistry
        """
        self.coords = hexagonal_grid(n_rings)
        self.n_vesicles = len(self.coords)
        self.neighbors = get_neighbors(self.coords)
        self.coupling_strength = coupling_strength
        self.n_species = n_species
        self.n_outputs = n_outputs

        # Create master RNG for reproducibility
        master_rng = np.random.default_rng(seed)

        # Create base network (defines the chemistry)
        base_network = AutocatalyticNetwork(n_species, n_reactions, rng=master_rng)

        # Each vesicle gets a perturbed copy (same structure, ~10% rate variation)
        # This models "same chemistry type, different microenvironment"
        self.networks = []
        for i in range(self.n_vesicles):
            # Create independent RNG for each vesicle's perturbation
            vesicle_rng = np.random.default_rng(seed + i + 1)
            net = base_network.copy_with_perturbation(perturbation=0.1, rng=vesicle_rng)
            self.networks.append(net)

        # Set output species
        self.output_species = list(range(n_species - n_outputs, n_species))
        for net in self.networks:
            net.output_species = self.output_species

    def run(self, env_inputs, t_span=(0, 100), n_points=500, x0=None, trial_seed=None):
        """
        Run the coupled system.

        Args:
            env_inputs: array of shape (n_vesicles, n_input_species)
            t_span: Integration time span
            n_points: Number of output time points
            x0: Initial concentrations (if None, random)
            trial_seed: Seed for random initial conditions

        Returns: boundary signals after substrate competition
        """
        # Initial conditions
        if x0 is not None:
            x0_flat = x0.flatten()
        else:
            if trial_seed is not None:
                trial_rng = np.random.default_rng(trial_seed)
            else:
                trial_rng = np.random.default_rng()
            x0 = trial_rng.uniform(0.1, 0.5, (self.n_vesicles, self.n_species))
            x0_flat = x0.flatten()

        def coupled_dynamics(t, x_flat):
            x = x_flat.reshape(self.n_vesicles, self.n_species)
            dxdt = np.zeros_like(x)

            # Internal dynamics for each vesicle
            for i in range(self.n_vesicles):
                dxdt[i] = self.networks[i].dynamics(t, x[i], env_inputs[i])

            # Coupling: diffusion of output species between neighbors
            for i in range(self.n_vesicles):
                for j in self.neighbors[i]:
                    for sp in self.output_species:
                        # Diffusive coupling
                        flux = self.coupling_strength * (x[j, sp] - x[i, sp])
                        dxdt[i, sp] += flux

            return dxdt.flatten()

        # Integrate
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(coupled_dynamics, t_span, x0_flat,
                        t_eval=t_eval, method='LSODA',
                        rtol=1e-6, atol=1e-9)

        if not sol.success:
            print(f"  Warning: integration failed - {sol.message}")

        # Extract final state
        x_final = sol.y[:, -1].reshape(self.n_vesicles, self.n_species)

        # Get boundary signals (output species after substrate competition)
        boundary_signals = np.zeros((self.n_vesicles, self.n_outputs))
        for i in range(self.n_vesicles):
            outputs = x_final[i, self.output_species]
            boundary_signals[i] = substrate_competition(outputs)

        # Return colony-level code (mean boundary signal)
        code = boundary_signals.mean(axis=0)

        return code, sol


# =============================================================================
# ENVIRONMENT ENCODING (same as reservoir version)
# =============================================================================

def encode_environment(env_id, n_bits=5, n_input_species=5):
    """
    Convert environment ID (0-31) to input pattern.
    Same encoding as reservoir version for direct comparison.
    """
    bits = [(env_id >> i) & 1 for i in range(n_bits)]

    # Map bits to input concentrations
    inputs = np.zeros(n_input_species)
    for i, bit in enumerate(bits):
        if i < n_input_species:
            inputs[i] = 1.0 if bit else 0.2

    return inputs


def create_spatial_gradient(coords, env_id):
    """
    Create spatially varying environment across vesicle array.
    """
    n_vesicles = len(coords)
    n_inputs = 5

    base_inputs = encode_environment(env_id, n_bits=5, n_input_species=n_inputs)

    # Spatial modulation based on position
    env_inputs = np.zeros((n_vesicles, n_inputs))
    for i, (x, y) in enumerate(coords):
        # Modulate by position
        spatial_mod = np.array([
            1 + 0.3 * x,  # Left-right gradient
            1 + 0.3 * y,  # Top-bottom gradient
            1.0,          # Uniform
            1 - 0.2 * np.sqrt(x**2 + y**2),  # Center-edge gradient
            1.0
        ])
        env_inputs[i] = base_inputs * np.abs(spatial_mod[:n_inputs])

    return env_inputs


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_code_emergence_experiment(n_envs=32, n_trials=3, n_rings=2,
                                   n_species=15, n_reactions=30, n_outputs=8,
                                   seed=42):
    """
    Test whether codes emerge from chemical reaction networks.

    CRITICAL: Creates ONE chemistry, tests across ALL environments.
    """
    print("=" * 70)
    print("CHEMICAL REACTION NETWORK - CODE EMERGENCE TEST")
    print("=" * 70)
    print(f"\nThis is NOT a neural network. It's mass-action ODEs.")
    print(f"One chemistry is tested across all {n_envs} environments.\n")

    n_vesicles = len(hexagonal_grid(n_rings))
    print(f"Parameters:")
    print(f"  Vesicles: {n_vesicles} (hexagonal, {n_rings} rings)")
    print(f"  Species per vesicle: {n_species}")
    print(f"  Reactions per vesicle: {n_reactions}")
    print(f"  Output channels: {n_outputs}")
    print(f"  Environments: {n_envs}")
    print(f"  Trials per environment: {n_trials}")
    print()

    # CREATE CHEMISTRY ONCE - this is the key fix
    print("Creating chemistry (fixed for all environments)...")
    array = ChemicalVesicleArray(
        n_rings=n_rings,
        n_species=n_species,
        n_reactions=n_reactions,
        n_outputs=n_outputs,
        seed=seed
    )
    print(f"  Created array with {array.n_vesicles} vesicles\n")

    all_codes = []
    env_labels = []
    trial_codes_all = []  # Store all trial codes for within-class variance

    start_time = time.time()

    for env_id in range(n_envs):
        print(f"Environment {env_id+1}/{n_envs}...", end=" ", flush=True)
        env_start = time.time()

        # Create environment (same across trials)
        env_inputs = create_spatial_gradient(array.coords, env_id)

        trial_codes = []
        for trial in range(n_trials):
            # Only vary initial conditions, NOT the chemistry
            code, _ = array.run(
                env_inputs,
                t_span=(0, 50),
                n_points=200,
                trial_seed=seed + env_id * 1000 + trial
            )
            trial_codes.append(code)

        # Store mean code for this environment
        mean_code = np.mean(trial_codes, axis=0)
        all_codes.append(mean_code)
        env_labels.append(env_id)
        trial_codes_all.append(trial_codes)

        env_time = time.time() - env_start
        print(f"({env_time:.1f}s)")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    # Analyze codes
    all_codes = np.array(all_codes)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # 1. Uniqueness: are codes distinguishable?
    code_dists = squareform(pdist(all_codes))
    np.fill_diagonal(code_dists, np.inf)

    # Check for collisions
    collisions = 0
    for i in range(n_envs):
        for j in range(i+1, n_envs):
            if code_dists[i, j] < 0.01:  # Very close = collision
                collisions += 1

    unique_codes = n_envs - collisions

    print(f"\n1. CODE UNIQUENESS")
    print(f"   Unique codes: {unique_codes}/{n_envs}")
    print(f"   Collisions (dist < 0.01): {collisions}")

    # 2. Separation ratio (with absolute distances)
    between_class = []
    for i in range(n_envs):
        for j in range(i+1, n_envs):
            between_class.append(code_dists[i, j])

    # Within-class distances from trial variance
    within_class = []
    for env_trials in trial_codes_all:
        if len(env_trials) > 1:
            trial_dists = pdist(np.array(env_trials))
            within_class.extend(trial_dists)

    mean_between = np.mean(between_class)
    min_between = np.min(between_class)
    mean_within = np.mean(within_class) if within_class else 0

    separation_ratio = mean_between / mean_within if mean_within > 0 else float('inf')

    print(f"\n2. CODE SEPARATION")
    print(f"   Mean between-class distance: {mean_between:.4f}")
    print(f"   Min between-class distance: {min_between:.4f}")
    print(f"   Mean within-class distance: {mean_within:.4f}")
    print(f"   Separation ratio: {separation_ratio:.1f}x")

    # 3. Discretization check
    from scipy.stats import entropy
    code_flat = all_codes.flatten()
    hist, _ = np.histogram(code_flat, bins=20, density=True)
    code_entropy = entropy(hist + 1e-10)
    uniform_entropy = np.log(20)
    discretization = 1 - code_entropy / uniform_entropy

    print(f"\n3. DISCRETIZATION")
    print(f"   Output entropy: {code_entropy:.2f} (uniform would be {uniform_entropy:.2f})")
    print(f"   Discretization score: {discretization:.2f} (1.0 = fully discrete)")

    # 4. Winner-take-most check
    winner_fractions = np.max(all_codes, axis=1)
    mean_winner = np.mean(winner_fractions)
    random_winner = 1.0 / n_outputs

    print(f"\n4. WINNER-TAKE-MOST")
    print(f"   Mean winner fraction: {mean_winner:.2f} (random would be {random_winner:.3f})")

    # 5. Decoding accuracy (nearest centroid)
    correct = 0
    confusion = np.zeros((n_envs, n_envs), dtype=int)

    for env_id, env_trials in enumerate(trial_codes_all):
        for trial_code in env_trials:
            # Find nearest centroid
            dists_to_centroids = [np.linalg.norm(trial_code - c) for c in all_codes]
            predicted = np.argmin(dists_to_centroids)
            confusion[env_id, predicted] += 1
            if predicted == env_id:
                correct += 1

    total_trials = sum(len(t) for t in trial_codes_all)
    accuracy = correct / total_trials if total_trials > 0 else 0

    print(f"\n5. DECODING ACCURACY")
    print(f"   Nearest-centroid accuracy: {accuracy:.1%}")
    print(f"   ({correct}/{total_trials} trials correctly classified)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success = (unique_codes >= n_envs * 0.6 and mean_winner > 0.2 and discretization > 0.2)

    if success:
        print("\n✓ CODES EMERGE FROM CHEMISTRY")
        print("  Mass-action kinetics + substrate competition → discrete codes")
        print("  This is NOT a neural network artifact.")
    else:
        print("\n✗ Code emergence weak or absent")
        print("  May need parameter tuning or longer integration time")

    return all_codes, env_labels, {
        'unique_codes': unique_codes,
        'collisions': collisions,
        'mean_between': mean_between,
        'mean_within': mean_within,
        'separation_ratio': separation_ratio,
        'discretization': discretization,
        'mean_winner': mean_winner,
        'accuracy': accuracy,
        'confusion': confusion
    }


def run_quick_test():
    """Quick test with fewer environments."""
    print("Running quick test (8 environments)...\n")
    return run_code_emergence_experiment(n_envs=8, n_trials=2, n_rings=1, seed=42)


def run_full_test():
    """Full test with all 32 environments."""
    print("Running full test (32 environments)...\n")
    return run_code_emergence_experiment(n_envs=32, n_trials=3, n_rings=2, seed=42)


def run_messy_test():
    """
    "Messy chemistry" test: high species count, many reactions.
    Should produce SHARPER codes per the theory.

    Tests the "messier is better" prediction: more species → higher D
    → more orthogonal directions for competition → sharper discretization.
    """
    print("Running MESSY test (high-dimensional chemistry)...\n")
    print("Theory predicts: more species → better discretization\n")

    return run_code_emergence_experiment(
        n_envs=32,
        n_trials=3,
        n_rings=3,      # 37 vesicles
        n_species=50,   # More species
        n_reactions=150,  # More reactions
        n_outputs=15,   # More output channels
        seed=42
    )


def run_comparison():
    """
    Run both standard and messy chemistry to compare.
    Tests the "messier is better" prediction directly.
    """
    print("=" * 70)
    print("COMPARISON: Standard vs Messy Chemistry")
    print("=" * 70)
    print("\nTheory predicts messy chemistry produces BETTER codes.\n")

    print("\n--- STANDARD CHEMISTRY (15 species, 30 reactions) ---\n")
    _, _, standard_stats = run_code_emergence_experiment(
        n_envs=32, n_trials=3, n_rings=2,
        n_species=15, n_reactions=30, n_outputs=8,
        seed=42
    )

    print("\n--- MESSY CHEMISTRY (50 species, 150 reactions) ---\n")
    _, _, messy_stats = run_code_emergence_experiment(
        n_envs=32, n_trials=3, n_rings=3,
        n_species=50, n_reactions=150, n_outputs=15,
        seed=42
    )

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Standard':>12} {'Messy':>12}")
    print("-" * 50)
    print(f"{'Unique codes':<25} {standard_stats['unique_codes']:>12} {messy_stats['unique_codes']:>12}")
    print(f"{'Discretization':<25} {standard_stats['discretization']:>11.0%} {messy_stats['discretization']:>11.0%}")
    print(f"{'Winner fraction':<25} {standard_stats['mean_winner']:>12.2f} {messy_stats['mean_winner']:>12.2f}")
    print(f"{'Separation ratio':<25} {standard_stats['separation_ratio']:>11.1f}x {messy_stats['separation_ratio']:>11.1f}x")
    print(f"{'Decode accuracy':<25} {standard_stats['accuracy']:>11.0%} {messy_stats['accuracy']:>11.0%}")

    if messy_stats['discretization'] > standard_stats['discretization']:
        print("\n✓ MESSIER IS BETTER: Prediction confirmed!")
    else:
        print("\n✗ Unexpected: messy chemistry did not outperform standard")

    return standard_stats, messy_stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            codes, labels, stats = run_full_test()
        elif sys.argv[1] == "--messy":
            codes, labels, stats = run_messy_test()
        elif sys.argv[1] == "--compare":
            run_comparison()
        else:
            print("Unknown option:", sys.argv[1])
    else:
        print("Usage: python chemistry_sim.py [--full|--messy|--compare]")
        print("  Default: quick test (8 envs)")
        print("  --full:    full test (32 envs, 15 species)")
        print("  --messy:   messy chemistry (32 envs, 50 species)")
        print("  --compare: run both and compare\n")
        codes, labels, stats = run_quick_test()
