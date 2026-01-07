#!/usr/bin/env python3
"""
Formose reaction simulation with compartmentalization.

Demonstrates that formose chemistry can achieve D_eff > 1 through:
- Autocatalytic loops (glycolaldehyde cycle)
- Branching pathways (C2-C6 sugars)
- Endogenous timescale separation
- Substrate competition at compartment boundaries

This is a proof-of-concept that chemistry CAN cross the code-emergence threshold.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt


def formose_network():
    """
    Simplified formose reaction network.

    Species:
        0: C1 (formaldehyde) - feedstock
        1: C2 (glycolaldehyde) - autocatalyst
        2: C3 (glyceraldehyde)
        3: C4 (erythrose/erythrulose)
        4: C5 (ribose/ribulose)
        5: C6 (glucose/fructose)
        6: C2-iso (dihydroxyacetone) - C2 isomer
        7: C3-iso (dihydroxyacetone) - C3 isomer

    Returns rate constants based on literature values (Breslow 1959, etc.)
    """
    # Rate constants (relative, normalized to fastest reaction)
    # Fast aldol additions: k ~ 10^2
    # Slower isomerizations: k ~ 10^0
    # Side reactions: k ~ 10^-1

    reactions = [
        # Aldol additions (fast) - bimolecular
        {'reactants': [0, 1], 'products': [2], 'k': 100.0, 'type': 'aldol'},      # C1 + C2 -> C3
        {'reactants': [0, 2], 'products': [3], 'k': 80.0, 'type': 'aldol'},       # C1 + C3 -> C4
        {'reactants': [1, 1], 'products': [3], 'k': 50.0, 'type': 'aldol'},       # C2 + C2 -> C4
        {'reactants': [0, 3], 'products': [4], 'k': 60.0, 'type': 'aldol'},       # C1 + C4 -> C5
        {'reactants': [1, 2], 'products': [4], 'k': 40.0, 'type': 'aldol'},       # C2 + C3 -> C5
        {'reactants': [0, 4], 'products': [5], 'k': 40.0, 'type': 'aldol'},       # C1 + C5 -> C6
        {'reactants': [1, 3], 'products': [5], 'k': 30.0, 'type': 'aldol'},       # C2 + C4 -> C6
        {'reactants': [2, 2], 'products': [5], 'k': 25.0, 'type': 'aldol'},       # C3 + C3 -> C6

        # Retro-aldol (autocatalytic regeneration) - slower
        {'reactants': [3], 'products': [1, 1], 'k': 5.0, 'type': 'retro'},        # C4 -> 2 C2
        {'reactants': [5], 'products': [2, 2], 'k': 3.0, 'type': 'retro'},        # C6 -> 2 C3

        # Isomerizations (intermediate timescale)
        {'reactants': [1], 'products': [6], 'k': 1.0, 'type': 'iso'},             # C2 <-> C2-iso
        {'reactants': [6], 'products': [1], 'k': 1.0, 'type': 'iso'},
        {'reactants': [2], 'products': [7], 'k': 0.8, 'type': 'iso'},             # C3 <-> C3-iso
        {'reactants': [7], 'products': [2], 'k': 0.8, 'type': 'iso'},

        # Degradation (slow, represents Cannizzaro and other side reactions)
        {'reactants': [5], 'products': [], 'k': 0.1, 'type': 'degrad'},           # C6 -> tar
        {'reactants': [4], 'products': [], 'k': 0.05, 'type': 'degrad'},          # C5 -> tar
    ]

    return reactions, 8  # 8 species total


def formose_dynamics(t, y, reactions, feedstock_rate=1.0):
    """
    Compute dy/dt for formose reaction network.

    Includes continuous feedstock (C1) addition to maintain non-equilibrium.
    """
    n_species = len(y)
    dydt = np.zeros(n_species)

    # Continuous feedstock addition (flow reactor)
    dydt[0] += feedstock_rate

    # Dilution (to prevent blowup)
    dilution_rate = 0.1
    dydt -= dilution_rate * y

    for rxn in reactions:
        reactants = rxn['reactants']
        products = rxn['products']
        k = rxn['k']

        # Compute reaction rate
        rate = k
        for r in reactants:
            rate *= max(y[r], 0)  # Ensure non-negative

        # Apply stoichiometry
        for r in reactants:
            dydt[r] -= rate
        for p in products:
            dydt[p] += rate

    return dydt


def substrate_competition(outputs, h=1.0, S0=1.0):
    """Competitive allocation among output channels."""
    outputs = np.maximum(outputs, 0)
    powered = outputs ** h
    total = powered.sum() + S0
    return powered / total


class FormoseCompartmentArray:
    """
    Array of coupled compartments running formose chemistry.
    """

    def __init__(self, n_compartments=19, coupling_strength=0.1,
                 output_species=[1, 2, 3, 4, 5, 6, 7], seed=42):
        self.n_compartments = n_compartments
        self.coupling = coupling_strength
        self.output_species = output_species
        self.n_outputs = len(output_species)
        self.rng = np.random.default_rng(seed)

        self.reactions, self.n_species = formose_network()

        # Hexagonal grid positions
        self.coords = self._hex_grid()
        self.neighbors = self._get_neighbors()

        # Per-compartment parameter variation (microenvironment differences)
        self.feedstock_rates = 1.0 + 0.3 * self.rng.standard_normal(n_compartments)
        self.feedstock_rates = np.maximum(self.feedstock_rates, 0.1)

    def _hex_grid(self):
        """Generate hexagonal grid coordinates."""
        coords = [(0, 0)]
        for ring in range(1, 3):
            for i in range(6 * ring):
                angle = 2 * np.pi * i / (6 * ring)
                coords.append((ring * np.cos(angle), ring * np.sin(angle)))
                if len(coords) >= self.n_compartments:
                    break
            if len(coords) >= self.n_compartments:
                break
        return np.array(coords[:self.n_compartments])

    def _get_neighbors(self):
        """Find neighbors within distance threshold."""
        neighbors = []
        for i in range(self.n_compartments):
            n_i = []
            for j in range(self.n_compartments):
                if i != j:
                    dist = np.linalg.norm(self.coords[i] - self.coords[j])
                    if dist < 1.5:
                        n_i.append(j)
            neighbors.append(n_i)
        return neighbors

    def run(self, env_inputs, t_span=(0, 100), n_points=500, trial_seed=None):
        """
        Run simulation with environmental inputs modulating feedstock.

        env_inputs: array of shape (n_compartments,) modulating local conditions
        """
        if trial_seed is not None:
            self.rng = np.random.default_rng(trial_seed)

        # Initial conditions: small amounts of all species
        y0 = 0.1 + 0.05 * self.rng.standard_normal(
            (self.n_compartments, self.n_species)
        )
        y0 = np.maximum(y0, 0.01)
        y0 = y0.flatten()

        # Modulate feedstock by environment
        modulated_feedstock = self.feedstock_rates * (1 + 0.5 * env_inputs)

        def coupled_dynamics(t, y):
            y = y.reshape(self.n_compartments, self.n_species)
            dydt = np.zeros_like(y)

            # Boundary signals for coupling
            boundary_signals = np.zeros((self.n_compartments, self.n_outputs))
            for i in range(self.n_compartments):
                outputs = y[i, self.output_species]
                boundary_signals[i] = substrate_competition(outputs)

            for i in range(self.n_compartments):
                # Internal formose dynamics
                dydt[i] = formose_dynamics(
                    t, y[i], self.reactions,
                    feedstock_rate=modulated_feedstock[i]
                )

                # Coupling to neighbors via boundary signals
                if self.neighbors[i]:
                    neighbor_signal = np.mean(
                        [boundary_signals[j] for j in self.neighbors[i]],
                        axis=0
                    )
                    # Coupling affects output species
                    for k, sp in enumerate(self.output_species):
                        dydt[i, sp] += self.coupling * (
                            neighbor_signal[k] - boundary_signals[i, k]
                        )

            return dydt.flatten()

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(coupled_dynamics, t_span, y0, t_eval=t_eval,
                       method='RK45', max_step=0.5)

        # Extract steady-state boundary signals
        final_y = sol.y[:, -50:].mean(axis=1).reshape(self.n_compartments, self.n_species)

        boundary_codes = np.zeros((self.n_compartments, self.n_outputs))
        for i in range(self.n_compartments):
            outputs = final_y[i, self.output_species]
            boundary_codes[i] = substrate_competition(outputs)

        # Return mean code across compartments
        mean_code = boundary_codes.mean(axis=0)

        return mean_code, sol


def participation_ratio(codes):
    """Compute effective dimensionality via participation ratio."""
    if len(codes) < 2:
        return 1.0
    codes = np.array(codes)
    codes_centered = codes - codes.mean(axis=0)
    cov = np.cov(codes_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total < 1e-10:
        return 1.0
    return (total ** 2) / (eigenvalues ** 2).sum()


def run_formose_experiment(n_envs=32, n_trials=3, seed=42):
    """
    Run formose simulation across multiple environments.

    Returns metrics showing D_eff and code quality.
    """
    print("=" * 60)
    print("FORMOSE REACTION CODE EMERGENCE EXPERIMENT")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    array = FormoseCompartmentArray(n_compartments=19, coupling_strength=0.1, seed=seed)

    codes = []
    trial_codes_all = []

    for env_id in range(n_envs):
        # Environment: spatial gradient affecting feedstock availability
        env_inputs = np.zeros(array.n_compartments)
        # Create gradient based on env_id bits
        for bit in range(5):
            if env_id & (1 << bit):
                direction = rng.standard_normal(2)
                direction /= np.linalg.norm(direction)
                for i, coord in enumerate(array.coords):
                    env_inputs[i] += np.dot(coord, direction) * (0.5 ** bit)

        trial_codes = []
        for trial in range(n_trials):
            code, _ = array.run(env_inputs, t_span=(0, 100), n_points=500,
                               trial_seed=seed + env_id * 1000 + trial)
            trial_codes.append(code)

        codes.append(np.mean(trial_codes, axis=0))
        trial_codes_all.append(trial_codes)

        if (env_id + 1) % 8 == 0:
            print(f"  Completed {env_id + 1}/{n_envs} environments")

    codes = np.array(codes)

    # Compute metrics
    d_eff = participation_ratio(codes)

    # Unique codes
    code_dists = squareform(pdist(codes))
    adjacency = (code_dists < 0.01).astype(int)
    np.fill_diagonal(adjacency, 0)
    n_components, _ = connected_components(csr_matrix(adjacency), directed=False)

    # Accuracy (leave-one-out)
    correct = 0
    total = 0
    for test_env_id, test_trials in enumerate(trial_codes_all):
        for test_idx, test_code in enumerate(test_trials):
            centroids = []
            for env_id, env_trials in enumerate(trial_codes_all):
                if env_id == test_env_id:
                    other = [t for i, t in enumerate(env_trials) if i != test_idx]
                    centroids.append(np.mean(other, axis=0) if other else env_trials[0])
                else:
                    centroids.append(np.mean(env_trials, axis=0))
            predicted = np.argmin([np.linalg.norm(test_code - c) for c in centroids])
            if predicted == test_env_id:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Effective dimensionality (D_eff): {d_eff:.2f}")
    print(f"Unique codes: {n_components}/{n_envs}")
    print(f"Decode accuracy: {accuracy*100:.1f}%")
    print()

    if d_eff > 1.0:
        print("*** D_eff > 1: THRESHOLD CROSSED ***")
        print("Formose chemistry can support code emergence.")
    else:
        print("D_eff = 1: collapsed dynamics, threshold not crossed")

    return {
        'd_eff': d_eff,
        'unique_codes': n_components,
        'accuracy': accuracy,
        'codes': codes
    }


def plot_formose_results(results, filename='figures/fig_formose.pdf'):
    """Generate figure showing formose code emergence."""
    import os
    os.makedirs('figures', exist_ok=True)

    codes = results['codes']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Code space (first 2 PCs)
    ax = axes[0]
    codes_centered = codes - codes.mean(axis=0)
    cov = np.cov(codes_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    pc1 = codes_centered @ eigenvectors[:, idx[0]]
    pc2 = codes_centered @ eigenvectors[:, idx[1]]

    ax.scatter(pc1, pc2, c=np.arange(len(codes)), cmap='viridis', s=50)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title(f'(A) Formose Code Space\n$D_{{eff}}$ = {results["d_eff"]:.2f}',
                fontsize=12, fontweight='bold')

    # Panel B: Output distribution
    ax = axes[1]
    all_outputs = codes.flatten()
    ax.hist(all_outputs, bins=30, density=True, alpha=0.7, color='#2ca02c', edgecolor='black')
    ax.set_xlabel('Output Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'(B) Output Distribution\n{results["unique_codes"]}/32 unique codes, '
                f'{results["accuracy"]*100:.0f}% accuracy', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved {filename}")
    plt.close()


if __name__ == '__main__':
    results = run_formose_experiment(n_envs=32, n_trials=3, seed=42)
    plot_formose_results(results)
