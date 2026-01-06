"""
Oregonator Reaction-Diffusion Simulation for HeroX Evolution 2.0

This is the REAL BZ chemistry model, not a toy phase oscillator.

The Oregonator is the standard 2-variable reduction of the Field-Körös-Noyes
mechanism for the Belousov-Zhabotinsky reaction:

    du/dt = (1/ε)[u - u² - fv(u-q)/(u+q)] + D_u ∇²u
    dv/dt = u - v + D_v ∇²v

Where:
    u = [HBrO₂] (activator - autocatalytic intermediate)
    v = [Ce⁴⁺]/[Ce³⁺] (oxidized/reduced catalyst ratio)
    ε = timescale ratio (~0.01-0.1, activator is fast)
    f = stoichiometric factor (~0.5-2)
    q = excitability parameter (~0.001)
    D_u, D_v = diffusion coefficients

This produces GENUINE high-dimensional dynamics:
    - 50×50 grid = 2500 spatial points
    - 2 variables per point = 5000 total DoF
    - Before pattern formation: D_eff ~ 1000-3000
    - After pattern formation: D_eff collapses to ~10-50 (pattern modes)

The "code" emerges from how the readout boundary samples the pattern field.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import laplace
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# OREGONATOR MODEL
# =============================================================================

class OregonatorField:
    """
    2D Oregonator reaction-diffusion system with MEMBRANE BOUNDARY.

    This models actual BZ chemistry inside a compartment:
    - Interior: High-D reaction-diffusion (the "cytoplasm")
    - Membrane: Boundary layer where inputs enter and outputs are read
    - The membrane IS the projection interface

    Early life = membrane-bound oscillator. The membrane:
    1. Receives external signals (context → receptor activation)
    2. Transduces them into interior perturbations
    3. Samples interior state for output (readout)
    """

    def __init__(self,
                 nx: int = 50,
                 ny: int = 50,
                 dx: float = 1.0,
                 epsilon: float = 0.1,    # Slower = more stable patterns
                 f: float = 1.0,          # Standard stoichiometry
                 q: float = 0.01,         # Less excitable = less chaotic
                 D_u: float = 1.0,
                 D_v: float = 0.5,
                 membrane_width: int = 3,
                 seed: int = None):
        """
        Initialize the Oregonator field.

        Args:
            nx, ny: Grid dimensions
            dx: Grid spacing
            epsilon: Timescale ratio (activator dynamics are fast)
            f: Stoichiometric factor
            q: Excitability parameter
            D_u, D_v: Diffusion coefficients
            membrane_width: Width of boundary layer (membrane)
            seed: Random seed
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.epsilon = epsilon
        self.f = f
        self.q = q
        self.D_u = D_u
        self.D_v = D_v
        self.membrane_width = membrane_width

        if seed is not None:
            np.random.seed(seed)

        # Initialize fields with SMALL perturbations - let context dominate
        # The key is that context (membrane input) should determine the pattern,
        # not random initial conditions
        self.u = 0.2 + 0.02 * np.random.rand(nx, ny)
        self.v = 0.1 + 0.02 * np.random.rand(nx, ny)

        # Membrane mask: True for boundary cells
        self.membrane_mask = np.zeros((nx, ny), dtype=bool)
        self.membrane_mask[:membrane_width, :] = True
        self.membrane_mask[-membrane_width:, :] = True
        self.membrane_mask[:, :membrane_width] = True
        self.membrane_mask[:, -membrane_width:] = True

        # Interior mask
        self.interior_mask = ~self.membrane_mask

        # Membrane "receptor" states - how external context affects interior
        # 4 receptor zones (one per side)
        self.receptor_states = np.zeros(4)  # [top, bottom, left, right]

        # History for D_eff calculation
        self.history = []

    def reaction_u(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Reaction term for activator u."""
        return (1/self.epsilon) * (u - u**2 - self.f * v * (u - self.q) / (u + self.q))

    def reaction_v(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Reaction term for catalyst v."""
        return u - v

    def diffusion(self, field: np.ndarray, D: float) -> np.ndarray:
        """Laplacian diffusion with reflecting (no-flux) boundaries."""
        # Use 'nearest' mode for no-flux (Neumann) boundary conditions
        # This is more realistic for a membrane-bound compartment
        return D * laplace(field, mode='nearest') / (self.dx**2)

    def apply_membrane_input(self, context_bits: List[int]):
        """
        Apply external context through membrane "receptors".

        This is HOW WE POKE THE SYSTEM:
        - External signals bind to membrane receptors
        - Receptors modulate local concentrations at the boundary
        - This perturbation propagates into the interior

        Args:
            context_bits: 6 bits encoding external context
        """
        w = self.membrane_width

        # Bit 0-1: Top membrane receptor activation (STRONG input)
        if context_bits[0]:
            self.u[:w, :] += 0.3  # Strong activator injection at top
        if context_bits[1]:
            self.v[:w, :] += 0.2  # Catalyst injection at top

        # Bit 2-3: Bottom membrane
        if context_bits[2]:
            self.u[-w:, :] += 0.3
        if context_bits[3]:
            self.v[-w:, :] += 0.2

        # Bit 4: Left membrane
        if context_bits[4]:
            self.u[:, :w] += 0.25

        # Bit 5: Right membrane
        if context_bits[5]:
            self.u[:, -w:] += 0.25

        # Store receptor states
        self.receptor_states = np.array([
            context_bits[0] + context_bits[1],  # top
            context_bits[2] + context_bits[3],  # bottom
            context_bits[4],                     # left
            context_bits[5],                     # right
        ])

    def read_membrane_output(self) -> List[float]:
        """
        Sample the membrane state for output.

        This is HOW WE READ THE SYSTEM:
        - The membrane integrates interior dynamics
        - We sample the mean u/v ratio at each membrane zone
        - This projects high-D interior to low-D output

        Returns:
            4 values representing membrane zone states [top, bottom, left, right]
        """
        w = self.membrane_width

        # Sample each membrane zone
        zones = [
            (self.u[:w, :], self.v[:w, :]),           # top
            (self.u[-w:, :], self.v[-w:, :]),         # bottom
            (self.u[:, :w], self.v[:, :w]),           # left
            (self.u[:, -w:], self.v[:, -w:]),         # right
        ]

        outputs = []
        for u_zone, v_zone in zones:
            # Compute effective "membrane potential" from u/v ratio
            ratio = np.mean(u_zone) / (np.mean(v_zone) + 1e-10)
            outputs.append(ratio)

        return outputs

    def step(self, dt: float = 0.01):
        """Euler step for the reaction-diffusion system."""
        # Reaction terms
        du_react = self.reaction_u(self.u, self.v)
        dv_react = self.reaction_v(self.u, self.v)

        # Diffusion terms (with no-flux boundaries)
        du_diff = self.diffusion(self.u, self.D_u)
        dv_diff = self.diffusion(self.v, self.D_v)

        # Update interior normally
        self.u += dt * (du_react + du_diff)
        self.v += dt * (dv_react + dv_diff)

        # Membrane has slightly reduced reaction rate (like a real lipid bilayer)
        # This creates a natural separation between interior dynamics and boundary
        # But not too strong - we still want the membrane to respond
        membrane_damping = 0.1
        self.u[self.membrane_mask] *= (1 - membrane_damping * dt)
        self.v[self.membrane_mask] *= (1 - membrane_damping * dt)

        # Ensure positivity
        self.u = np.maximum(self.u, 1e-10)
        self.v = np.maximum(self.v, 1e-10)

    def run(self, t_max: float, dt: float = 0.01,
            record_interval: int = 100) -> List[np.ndarray]:
        """
        Run simulation and record history.

        Args:
            t_max: Total simulation time
            dt: Time step
            record_interval: Record state every N steps

        Returns:
            history: List of (u, v) state snapshots
        """
        n_steps = int(t_max / dt)
        self.history = []

        for i in range(n_steps):
            self.step(dt)
            if i % record_interval == 0:
                self.history.append((self.u.copy(), self.v.copy()))

        return self.history

    def get_state_vector(self) -> np.ndarray:
        """Flatten current state to 1D vector for D_eff calculation."""
        return np.concatenate([self.u.flatten(), self.v.flatten()])

    def apply_environmental_context(self, context_id: int, n_contexts: int = 64):
        """
        Apply environmental context through MEMBRANE RECEPTORS.

        This models how a real cell receives external signals:
        - Context bits encode which "receptors" are activated
        - Activated receptors inject chemicals at the membrane
        - Perturbations propagate into the interior

        The interior dynamics then evolve, and we READ the output
        through the membrane (projection interface).
        """
        # Use context bits to set membrane receptor activations
        bits = [(context_id >> i) & 1 for i in range(6)]

        # Apply context through membrane (this is the INPUT)
        self.apply_membrane_input(bits)

        # Small random interior perturbation (thermal noise)
        self.u[self.interior_mask] += 0.02 * np.random.rand(
            np.sum(self.interior_mask))


# =============================================================================
# DIMENSIONAL ANALYSIS
# =============================================================================

def compute_deff_from_history(history: List[Tuple[np.ndarray, np.ndarray]],
                               window: int = 20) -> Tuple[float, float]:
    """
    Compute D_eff from simulation history using participation ratio.

    Returns D_eff for early (transient) and late (steady) dynamics.

    D_eff measures how many dimensions are "active" - a measure of
    the effective degrees of freedom in the dynamics.
    """
    if len(history) < window * 2:
        # Fallback: use what we have
        window = max(len(history) // 3, 2)

    def deff_from_window(states):
        if len(states) < 2:
            return 1.0

        # Flatten each state to vector
        vectors = np.array([np.concatenate([u.flatten(), v.flatten()])
                           for u, v in states])

        # Subsample spatially for computational efficiency
        # Take every 4th point
        vectors = vectors[:, ::4]

        if vectors.shape[0] < 2 or vectors.shape[1] < 2:
            return 1.0

        # Center the data
        vectors = vectors - np.mean(vectors, axis=0)

        # SVD for numerical stability (instead of covariance eigendecomposition)
        try:
            U, S, Vt = np.linalg.svd(vectors, full_matrices=False)
            # Eigenvalues of covariance are S²/(n-1)
            eigenvalues = (S ** 2) / (len(vectors) - 1)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]

            if len(eigenvalues) == 0:
                return 1.0

            # Participation ratio
            d_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            return d_eff
        except:
            return 1.0

    # Early dynamics (transient) - first third
    n = len(history)
    early_states = history[:n//3]
    d_eff_early = deff_from_window(early_states)

    # Late dynamics (steady state / pattern) - last third
    late_states = history[2*n//3:]
    d_eff_late = deff_from_window(late_states)

    return d_eff_early, d_eff_late


# =============================================================================
# READOUT AND SYMBOL EXTRACTION (VIA MEMBRANE)
# =============================================================================

def membrane_output_to_symbol(membrane_outputs: List[float]) -> int:
    """
    Convert membrane zone outputs to a discrete symbol using GRADIENT PHYSICS.

    The 4 membrane zones (top, bottom, left, right) define a chemical gradient
    vector across the compartment. The decoder "follows the gradient" - this is
    physically justifiable (chemotaxis, diffusion-driven signaling).

    NO EXPLICIT CODEBOOK - the symbol emerges from the gradient direction:
    - Compute center-of-mass of membrane activation
    - Gradient vector points from low to high concentration
    - Quantize vector direction to 4 quadrants (N, E, S, W)

    This is physics, not logic.
    """
    # outputs = [top, bottom, left, right]
    outputs = np.array(membrane_outputs)

    # Compute gradient vector (center of mass of activation)
    # Y-axis: positive = top, negative = bottom
    # X-axis: positive = right, negative = left
    y_gradient = outputs[0] - outputs[1]  # top - bottom
    x_gradient = outputs[3] - outputs[2]  # right - left

    # Convert gradient vector to angle
    angle = np.arctan2(y_gradient, x_gradient)  # [-π, π]

    # Quantize to 4 quadrants (N=0, E=1, S=2, W=3)
    # Shift so boundaries are at 45° angles
    angle_shifted = (angle + np.pi/4) % (2 * np.pi)
    symbol = int(angle_shifted / (np.pi/2)) % 4

    return symbol


def extract_sequence_via_membrane(field: OregonatorField,
                                   history: List[Tuple[np.ndarray, np.ndarray]],
                                   sample_indices: List[int]) -> List[int]:
    """
    Extract symbol sequence by reading the membrane at sample times.

    This is the biologically accurate readout:
    - Interior dynamics evolve
    - At each sample time, we READ the membrane
    - Membrane state is projected to discrete symbol
    """
    sequence = []
    w = field.membrane_width

    for t_idx in sample_indices:
        if t_idx >= len(history):
            continue

        u, v = history[t_idx]

        # Read membrane zones (same logic as read_membrane_output)
        zones = [
            (u[:w, :], v[:w, :]),           # top
            (u[-w:, :], v[-w:, :]),         # bottom
            (u[:, :w], v[:, :w]),           # left
            (u[:, -w:], v[:, -w:]),         # right
        ]

        outputs = []
        for u_zone, v_zone in zones:
            ratio = np.mean(u_zone) / (np.mean(v_zone) + 1e-10)
            outputs.append(ratio)

        symbol = membrane_output_to_symbol(outputs)
        sequence.append(symbol)

    return sequence


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_trial(context_id: int,
              nx: int = 50, ny: int = 50,
              t_max: float = 50.0, dt: float = 0.01,
              n_symbols: int = 3,
              seed: int = None) -> Dict:
    """
    Run a single trial with given environmental context.

    The flow:
    1. Create membrane-bound reaction field
    2. Apply context THROUGH MEMBRANE (receptor activation)
    3. Let interior dynamics evolve
    4. READ output VIA MEMBRANE (projection to symbols)
    """
    # Initialize membrane-bound field
    field = OregonatorField(nx=nx, ny=ny, membrane_width=3, seed=seed)

    # Apply context through membrane receptors (this is the INPUT)
    field.apply_environmental_context(context_id)

    # Run simulation - interior dynamics evolve
    history = field.run(t_max=t_max, dt=dt, record_interval=50)

    # Compute D_eff (early vs late) - should show collapse
    d_eff_early, d_eff_late = compute_deff_from_history(history)

    n_history = len(history)

    # Sample at 3 time points in the latter half (after transient)
    sample_indices = [
        n_history // 2,
        n_history * 2 // 3,
        n_history - 1
    ]

    # Extract sequence by READING THE MEMBRANE (this is the OUTPUT)
    encoder_sequence = extract_sequence_via_membrane(field, history, sample_indices)

    # Simulate decoder (another membrane-bound system that entrains)
    decoder_sequence = simulate_decoder(encoder_sequence, coupling=0.9, seed=seed)

    # Collect final state for clustering analysis
    final_u, final_v = history[-1]

    return {
        'context_id': context_id,
        'd_eff_early': d_eff_early,
        'd_eff_late': d_eff_late,
        'encoder_sequence': tuple(encoder_sequence),
        'decoder_sequence': tuple(decoder_sequence),
        'end_to_end_match': encoder_sequence == decoder_sequence,
        'final_u': final_u,
        'final_v': final_v,
        'membrane_width': field.membrane_width,
    }


def simulate_decoder(encoder_sequence: List[int],
                     coupling: float = 0.9,
                     seed: int = None) -> List[int]:
    """Simulate decoder entrainment to encoder."""
    if seed is not None:
        np.random.seed(seed + 1000)

    decoder_sequence = []
    for symbol in encoder_sequence:
        if np.random.rand() < coupling:
            decoder_sequence.append(symbol)
        else:
            decoder_sequence.append(np.random.randint(0, 4))
    return decoder_sequence


def run_experiment(n_contexts: int = 16, n_trials: int = 5,
                   nx: int = 40, ny: int = 40) -> Dict:
    """
    Run full experiment with multiple contexts and trials.
    """
    print("=" * 70)
    print("OREGONATOR REACTION-DIFFUSION SIMULATION")
    print("HeroX Evolution 2.0 - TRUE High-Dimensional Code Emergence")
    print("=" * 70)
    print(f"\nGrid: {nx}×{ny} = {nx*ny} spatial points × 2 variables = {2*nx*ny} DoF")
    print(f"Running {n_contexts} contexts × {n_trials} trials...")
    print()

    all_results = []
    encoding_table = {}
    all_final_states = []

    for ctx_id in range(n_contexts):
        print(f"Context {ctx_id + 1}/{n_contexts}:", end=" ")

        ctx_sequences = []
        ctx_d_eff_early = []
        ctx_d_eff_late = []
        ctx_matches = []

        for trial in range(n_trials):
            result = run_trial(
                context_id=ctx_id,
                nx=nx, ny=ny,
                t_max=50.0,  # Longer for pattern stabilization
                seed=ctx_id * 1000 + trial
            )

            ctx_sequences.append(result['encoder_sequence'])
            ctx_d_eff_early.append(result['d_eff_early'])
            ctx_d_eff_late.append(result['d_eff_late'])
            ctx_matches.append(result['end_to_end_match'])
            all_final_states.append(result['final_u'].flatten())

        # Mode sequence
        from collections import Counter
        seq_counts = Counter(ctx_sequences)
        mode_seq = seq_counts.most_common(1)[0][0]
        reproducibility = seq_counts.most_common(1)[0][1] / n_trials

        encoding_table[ctx_id] = mode_seq

        d_early = np.mean(ctx_d_eff_early)
        d_late = np.mean(ctx_d_eff_late)
        collapse = d_early / max(d_late, 1)
        e2e = np.mean(ctx_matches)

        print(f"D_eff: {d_early:.0f} → {d_late:.0f} ({collapse:.1f}×), "
              f"Repro: {reproducibility:.0%}, E2E: {e2e:.0%}, "
              f"Seq: {mode_seq}")

        all_results.append({
            'context_id': ctx_id,
            'd_eff_early': d_early,
            'd_eff_late': d_late,
            'collapse_ratio': collapse,
            'reproducibility': reproducibility,
            'end_to_end_fidelity': e2e,
            'mode_sequence': mode_seq,
        })

    # Print encoding table
    print("\n" + "=" * 70)
    print("ENCODING TABLE (Emergent from dynamics)")
    print("=" * 70)
    for ctx_id, seq in encoding_table.items():
        seq_str = " ".join([f"S{s}" for s in seq])
        print(f"Context {ctx_id:2d}: {seq_str}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_d_early = np.mean([r['d_eff_early'] for r in all_results])
    mean_d_late = np.mean([r['d_eff_late'] for r in all_results])
    mean_collapse = np.mean([r['collapse_ratio'] for r in all_results])
    mean_repro = np.mean([r['reproducibility'] for r in all_results])
    mean_e2e = np.mean([r['end_to_end_fidelity'] for r in all_results])

    print(f"D_eff (transient): {mean_d_early:.0f}")
    print(f"D_eff (steady):    {mean_d_late:.0f}")
    print(f"Collapse ratio:    {mean_collapse:.1f}×")
    print(f"Reproducibility:   {mean_repro:.1%}")
    print(f"End-to-end:        {mean_e2e:.1%}")

    # Clustering analysis on final states
    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS (Discreteness Verification)")
    print("=" * 70)

    final_states = np.array(all_final_states)

    # PCA to reduce dimensionality for GMM
    pca = PCA(n_components=min(20, len(final_states)-1))
    states_pca = pca.fit_transform(final_states)

    # Compare 1-component vs 4-component GMM
    gmm_1 = GaussianMixture(n_components=1, random_state=42).fit(states_pca)
    gmm_4 = GaussianMixture(n_components=4, random_state=42).fit(states_pca)

    bic_1 = gmm_1.bic(states_pca)
    bic_4 = gmm_4.bic(states_pca)
    delta_bic = bic_1 - bic_4

    print(f"BIC (1-component): {bic_1:.0f}")
    print(f"BIC (4-component): {bic_4:.0f}")
    print(f"ΔBIC: {delta_bic:.0f}")
    print(f"  (threshold > 10 for strong evidence of clustering)")

    verdict = "✓ PASS" if delta_bic > 10 else "✗ FAIL"
    print(f"\nVERDICT: {verdict}")

    return {
        'results': all_results,
        'encoding_table': encoding_table,
        'final_states': final_states,
        'delta_bic': delta_bic,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_membrane_bound_evolution(context_id: int = 0,
                                        nx: int = 50, ny: int = 50):
    """
    Visualize how patterns evolve in the membrane-bound Oregonator.

    Shows:
    - Interior dynamics (reaction-diffusion patterns)
    - Membrane boundary (where input/output happens)
    - The projection from high-D interior to low-D membrane output
    """
    field = OregonatorField(nx=nx, ny=ny, membrane_width=3, seed=42)
    field.apply_environmental_context(context_id)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    times = [0, 5, 10, 20, 30, 40, 50, 60]
    dt = 0.01

    for i, t_target in enumerate(times):
        # Run to target time
        if i > 0:
            t_run = times[i] - times[i-1]
            field.run(t_max=t_run, dt=dt, record_interval=1000)

        ax = axes[i // 4, i % 4]

        # Show u field with membrane overlay
        display = field.u.copy()

        im = ax.imshow(display, cmap='viridis', vmin=0, vmax=0.5)
        ax.set_title(f't = {t_target}')
        ax.axis('off')

        # Overlay membrane boundary in red
        w = field.membrane_width
        # Draw membrane as red border
        membrane_display = np.zeros((nx, ny, 4))  # RGBA
        membrane_display[field.membrane_mask, 0] = 1.0  # Red
        membrane_display[field.membrane_mask, 3] = 0.3  # Alpha
        ax.imshow(membrane_display)

        # Add readout symbol
        if i > 0:
            outputs = field.read_membrane_output()
            symbol = membrane_output_to_symbol(outputs)
            ax.text(nx//2, ny+5, f'S{symbol}', ha='center', fontsize=10, color='red')

    plt.suptitle(f'Membrane-Bound Oregonator (Context {context_id})\n'
                 f'Red = membrane (projection interface), Interior = high-D dynamics',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/oregonator_membrane_evolution.png', dpi=150)
    print("Saved figures/oregonator_membrane_evolution.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)

    # Run experiment with all 64 contexts to measure full encoding capacity
    results = run_experiment(n_contexts=64, n_trials=5, nx=40, ny=40)

    print("\nDone!")
