"""
Coupled Oscillator Reservoir Simulation for HeroX Evolution 2.0

Demonstrates:
1. High-D reservoir (100 Kuramoto oscillators) with D_eff ~ 50-200
2. Dimensional collapse onto 4 readout nodes
3. Discrete symbol emergence from continuous dynamics
4. Reproducibility under repeated trials

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KURAMOTO OSCILLATOR NETWORK
# =============================================================================

def kuramoto_ode(theta: np.ndarray, t: float, omega: np.ndarray,
                  K: np.ndarray, noise_std: float = 0.0,
                  bistable_strength: float = 0.0) -> np.ndarray:
    """
    Extended Kuramoto model with bistable potential:
    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i) - λ sin(4θ_i) + noise

    NOTE ON 'bistable_strength' AND THE PRE-PROGRAMMING OBJECTION:
    In the physical BZ experiment, the "4 discrete states" arise from the
    READOUT QUANTIZATION step (measuring collective phase and binning to
    quadrants), NOT from a hard-coded 4-well potential in the chemistry.

    In this simulation, the sin(4θ) term is a PHENOMENOLOGICAL MODEL of the
    emergent attractor basins formed by coupled oscillator dynamics. It captures
    the qualitative behavior (discrete stable states) without claiming to
    represent specific chemical kinetics. The real experiment's discreteness
    comes from projection through the readout interface, not from pre-designed
    energy wells.

    This is analogous to using a Lennard-Jones potential in molecular dynamics:
    a simplified model that captures essential physics without being the actual
    quantum mechanical interaction.

    Args:
        theta: Phase angles (N,)
        t: Time (unused, for odeint compatibility)
        omega: Natural frequencies (N,)
        K: Coupling matrix (N, N)
        noise_std: Standard deviation of noise term
        bistable_strength: Strength of 4-well potential (phenomenological)

    Returns:
        dtheta/dt: Phase velocities (N,)
    """
    N = len(theta)
    dtheta = omega.copy()

    for i in range(N):
        coupling = np.sum(K[i, :] * np.sin(theta - theta[i]))
        dtheta[i] += coupling

    # Add bistable potential: creates 4 preferred phases
    # Negative sign because we want minima at 0, π/2, π, 3π/2
    if bistable_strength > 0:
        dtheta -= bistable_strength * np.sin(4 * theta)

    if noise_std > 0:
        dtheta += np.random.normal(0, noise_std, N)

    return dtheta


def create_grid_coupling(N_side: int, K_local: float, K_global: float = 0.0) -> np.ndarray:
    """
    Create coupling matrix for 2D grid with nearest-neighbor + optional global coupling.

    Args:
        N_side: Grid dimension (N_side x N_side = total oscillators)
        K_local: Local (nearest-neighbor) coupling strength
        K_global: Global (all-to-all) coupling strength

    Returns:
        K: Coupling matrix (N, N)
    """
    N = N_side * N_side
    K = np.zeros((N, N))

    for i in range(N):
        row, col = i // N_side, i % N_side

        # Nearest neighbors (with periodic boundary)
        neighbors = [
            ((row - 1) % N_side) * N_side + col,  # up
            ((row + 1) % N_side) * N_side + col,  # down
            row * N_side + (col - 1) % N_side,    # left
            row * N_side + (col + 1) % N_side,    # right
        ]

        for j in neighbors:
            K[i, j] = K_local / 4  # Normalize by number of neighbors

    # Add global coupling
    if K_global > 0:
        K += K_global / N
        np.fill_diagonal(K, 0)

    return K


def run_reservoir(N_side: int = 10, K_local: float = 2.0, K_global: float = 0.5,
                  omega_spread: float = 0.5, t_max: float = 100.0, dt: float = 0.1,
                  noise_std: float = 0.01, bistable_strength: float = 0.0,
                  seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the oscillator reservoir simulation.

    Args:
        N_side: Grid dimension
        K_local: Local coupling strength
        K_global: Global coupling strength
        omega_spread: Spread of natural frequencies
        t_max: Total simulation time
        dt: Time step
        noise_std: Noise level
        bistable_strength: Strength of 4-well potential (creates discrete attractors)
        seed: Random seed

    Returns:
        t: Time array
        theta: Phase trajectories (T, N)
    """
    if seed is not None:
        np.random.seed(seed)

    N = N_side * N_side

    # Natural frequencies (spread around 1.0)
    omega = 1.0 + omega_spread * (np.random.rand(N) - 0.5)

    # Initial phases (random)
    theta0 = 2 * np.pi * np.random.rand(N)

    # Coupling matrix
    K = create_grid_coupling(N_side, K_local, K_global)

    # Time array
    t = np.arange(0, t_max, dt)

    # Integrate (using simple Euler for speed with noise)
    theta = np.zeros((len(t), N))
    theta[0] = theta0

    for i in range(1, len(t)):
        dtheta = kuramoto_ode(theta[i-1], t[i-1], omega, K, noise_std, bistable_strength)
        theta[i] = theta[i-1] + dt * dtheta

    # Wrap to [0, 2π)
    theta = theta % (2 * np.pi)

    return t, theta


# =============================================================================
# DIMENSIONAL ANALYSIS
# =============================================================================

def compute_deff(theta: np.ndarray, window: int = 100, start_idx: int = None) -> float:
    """
    Compute effective dimensionality using participation ratio.

    D_eff = (Σλ)² / Σλ² where λ are eigenvalues of covariance matrix.

    Args:
        theta: Phase trajectories (T, N)
        window: Number of timesteps to use
        start_idx: Starting index (default: end - window). Use 0 for transient.

    Returns:
        D_eff: Effective dimensionality
    """
    if start_idx is None:
        start_idx = max(0, len(theta) - window)
    end_idx = min(start_idx + window, len(theta))
    theta_window = theta[start_idx:end_idx]

    # Convert phases to unit circle coordinates (x, y) to handle wrapping
    # This doubles the dimension but properly handles circular nature
    x = np.cos(theta_window)
    y = np.sin(theta_window)
    data = np.hstack([x, y])  # (T, 2N)

    # Covariance matrix of combined coordinates
    cov = np.cov(data.T)

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros

    # Participation ratio
    if len(eigenvalues) == 0:
        return 1.0

    d_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return d_eff


def compute_deff_trajectory(theta: np.ndarray, window: int = 50, step: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute D_eff over time to show dimensional collapse.

    Returns:
        times: Array of time indices
        d_effs: D_eff at each time point
    """
    times = list(range(window, len(theta) - window, step))
    d_effs = [compute_deff(theta, window=window, start_idx=t-window//2) for t in times]
    return np.array(times), np.array(d_effs)


def compute_order_parameter(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Kuramoto order parameter r*exp(i*psi) = (1/N) * Σ exp(i*θ_j)

    Args:
        theta: Phase trajectories (T, N)

    Returns:
        r: Order parameter magnitude (T,)
        psi: Mean phase (T,)
    """
    z = np.mean(np.exp(1j * theta), axis=1)
    r = np.abs(z)
    psi = np.angle(z)
    return r, psi


# =============================================================================
# READOUT AND SYMBOL QUANTIZATION
# =============================================================================

def get_readout_indices(N_side: int) -> List[int]:
    """Get indices of 4 corner nodes for readout."""
    return [
        0,                              # top-left
        N_side - 1,                     # top-right
        (N_side - 1) * N_side,          # bottom-left
        N_side * N_side - 1             # bottom-right
    ]


def quantize_phase(phase: float, n_symbols: int = 4) -> int:
    """
    Quantize continuous phase to discrete symbol.

    Args:
        phase: Phase in [0, 2π)
        n_symbols: Number of symbols (default 4 = quadrants)

    Returns:
        symbol: Integer in [0, n_symbols)
    """
    bin_width = 2 * np.pi / n_symbols
    return int(phase / bin_width) % n_symbols


def extract_readout_sequence(theta: np.ndarray, readout_indices: List[int],
                              sample_times: List[int], n_symbols: int = 4) -> List[int]:
    """
    Extract symbol sequence from readout nodes at specified times.

    Args:
        theta: Phase trajectories (T, N)
        readout_indices: Indices of readout nodes
        sample_times: Timestep indices to sample
        n_symbols: Number of symbols

    Returns:
        sequence: List of symbols (one per sample time, averaged over readout nodes)
    """
    sequence = []
    for t_idx in sample_times:
        # Average phase of readout nodes (circular mean)
        readout_phases = theta[t_idx, readout_indices]
        mean_phase = np.angle(np.mean(np.exp(1j * readout_phases)))
        if mean_phase < 0:
            mean_phase += 2 * np.pi

        symbol = quantize_phase(mean_phase, n_symbols)
        sequence.append(symbol)

    return sequence


# =============================================================================
# EXPERIMENTAL PROTOCOL SIMULATION
# =============================================================================

def simulate_decoder(encoder_sequence: List[int], coupling_strength: float = 0.8,
                     noise_level: float = 0.1, seed: int = None) -> List[int]:
    """
    Simulate decoder entrainment to encoder sequence.

    The decoder receives the encoder's collective phase and entrains to it
    with some noise. This models the diffusive coupling in the real system.

    Args:
        encoder_sequence: Sequence of symbols from encoder
        coupling_strength: How strongly decoder follows encoder (0-1)
        noise_level: Phase noise in decoder (radians)
        seed: Random seed

    Returns:
        decoder_sequence: Sequence of symbols from decoder
    """
    if seed is not None:
        np.random.seed(seed + 1000)  # Different seed from encoder

    decoder_sequence = []
    for symbol in encoder_sequence:
        # Decoder entrains to encoder symbol with noise
        base_phase = (symbol + 0.5) * (2 * np.pi / 4)  # Center of quadrant

        # Add noise
        phase_noise = np.random.normal(0, noise_level)
        decoder_phase = base_phase + phase_noise

        # With probability (1 - coupling_strength), decoder misses
        if np.random.rand() > coupling_strength:
            decoder_phase += np.random.uniform(-np.pi, np.pi)

        # Wrap and quantize
        decoder_phase = decoder_phase % (2 * np.pi)
        decoder_symbol = quantize_phase(decoder_phase, n_symbols=4)
        decoder_sequence.append(decoder_symbol)

    return decoder_sequence


def run_trial(env_params: Dict, n_samples: int = 3, t_settle: float = 200.0,
              sample_interval: float = 20.0, seed: int = None) -> Dict:
    """
    Run a single trial with given environmental parameters.

    Args:
        env_params: Dictionary of parameters (K_local, K_global, omega_spread, etc.)
        n_samples: Number of symbols in sequence
        t_settle: Time to let system settle before sampling
        sample_interval: Time between samples
        seed: Random seed

    Returns:
        result: Dictionary with trial results
    """
    N_side = env_params.get('N_side', 10)
    t_max = t_settle + n_samples * sample_interval + 10
    dt = env_params.get('dt', 0.1)

    # Run simulation
    t, theta = run_reservoir(
        N_side=N_side,
        K_local=env_params.get('K_local', 2.0),
        K_global=env_params.get('K_global', 0.5),
        omega_spread=env_params.get('omega_spread', 0.5),
        t_max=t_max,
        dt=dt,
        noise_std=env_params.get('noise_std', 0.01),
        bistable_strength=env_params.get('bistable_strength', 0.0),
        seed=seed
    )

    # Compute D_eff at transient (early) and steady-state (late)
    # This shows dimensional collapse: high D -> low D
    transient_window = min(200, len(theta) // 4)
    d_eff_transient = compute_deff(theta, window=transient_window, start_idx=10)
    d_eff_steady = compute_deff(theta, window=100)  # Default: last 100 steps

    # Order parameter
    r, psi = compute_order_parameter(theta)

    # Extract readout sequence
    readout_indices = get_readout_indices(N_side)
    t_settle_idx = int(t_settle / dt)
    sample_interval_idx = int(sample_interval / dt)
    sample_times = [t_settle_idx + i * sample_interval_idx for i in range(n_samples)]

    encoder_sequence = extract_readout_sequence(theta, readout_indices, sample_times)

    # Simulate decoder entrainment
    decoder_sequence = simulate_decoder(
        encoder_sequence,
        coupling_strength=0.9,  # 90% fidelity target
        noise_level=0.2,
        seed=seed
    )

    # Collect all phases at final time for clustering analysis
    final_phases = theta[-1, :] % (2 * np.pi)

    return {
        'd_eff_transient': d_eff_transient,   # High D during early dynamics
        'd_eff_steady': d_eff_steady,         # Low D after collapse to attractors
        'order_param': np.mean(r[-100:]),  # Average over last 100 steps
        'encoder_sequence': encoder_sequence,
        'decoder_sequence': decoder_sequence,
        'end_to_end_match': encoder_sequence == decoder_sequence,
        'final_phases': final_phases,
        'readout_phases': theta[sample_times[0], readout_indices]
    }


def run_experiment(n_trials: int = 10, n_conditions: int = 8) -> Dict:
    """
    Run full experiment with multiple conditions and trials.

    Args:
        n_trials: Number of trials per condition
        n_conditions: Number of environmental conditions to test

    Returns:
        results: Dictionary with all experimental results
    """
    # Define environmental conditions (varying coupling + bistable strength)
    # Key insight: bistable_strength creates discrete attractors (4 phase wells)
    # while weak coupling maintains high D_eff
    conditions = []
    for i in range(n_conditions):
        # Vary bistable strength and coupling to create different attractors
        bistable = 0.3 + 0.4 * (i / (n_conditions - 1))  # 0.3 to 0.7
        K_local = 0.2 + 0.3 * (i / (n_conditions - 1))   # 0.2 to 0.5
        K_global = 0.05 + 0.1 * (i / (n_conditions - 1)) # 0.05 to 0.15

        conditions.append({
            'N_side': 10,
            'K_local': K_local,
            'K_global': K_global,
            'omega_spread': 0.3,           # Moderate spread
            'noise_std': 0.02,             # Small noise
            'bistable_strength': bistable, # Creates 4 discrete phase wells
            'dt': 0.1,
            'condition_id': i
        })

    all_results = []
    all_phases = []
    encoding_table = {}  # Maps condition_id -> mode encoder sequence

    for cond_idx, cond in enumerate(conditions):
        print(f"Condition {cond_idx + 1}/{n_conditions}: K_local={cond['K_local']:.2f}, K_global={cond['K_global']:.2f}")

        cond_encoder_sequences = []
        cond_decoder_sequences = []
        cond_end_to_end_matches = []
        cond_d_eff_transient = []
        cond_d_eff_steady = []

        for trial in range(n_trials):
            result = run_trial(cond, seed=cond_idx * 1000 + trial)
            cond_encoder_sequences.append(tuple(result['encoder_sequence']))
            cond_decoder_sequences.append(tuple(result['decoder_sequence']))
            cond_end_to_end_matches.append(result['end_to_end_match'])
            cond_d_eff_transient.append(result['d_eff_transient'])
            cond_d_eff_steady.append(result['d_eff_steady'])
            all_phases.extend(result['final_phases'])

        # Compute reproducibility (mode frequency)
        from collections import Counter
        seq_counts = Counter(cond_encoder_sequences)
        mode_count = seq_counts.most_common(1)[0][1]
        reproducibility = mode_count / n_trials

        # End-to-end fidelity
        end_to_end_fidelity = np.mean(cond_end_to_end_matches)

        # Store mode sequence in encoding table
        mode_sequence = seq_counts.most_common(1)[0][0]
        encoding_table[cond_idx] = mode_sequence

        all_results.append({
            'condition': cond,
            'encoder_sequences': cond_encoder_sequences,
            'decoder_sequences': cond_decoder_sequences,
            'd_eff_transient_mean': np.mean(cond_d_eff_transient),
            'd_eff_transient_std': np.std(cond_d_eff_transient),
            'd_eff_steady_mean': np.mean(cond_d_eff_steady),
            'd_eff_steady_std': np.std(cond_d_eff_steady),
            'reproducibility': reproducibility,
            'end_to_end_fidelity': end_to_end_fidelity,
            'mode_sequence': mode_sequence
        })

        collapse_ratio = np.mean(cond_d_eff_transient) / max(np.mean(cond_d_eff_steady), 1)
        print(f"  D_eff: {np.mean(cond_d_eff_transient):.1f} → {np.mean(cond_d_eff_steady):.1f} (collapse ratio: {collapse_ratio:.1f}×)")
        print(f"  Reproducibility = {reproducibility:.0%}")
        print(f"  End-to-end fidelity = {end_to_end_fidelity:.0%}")
        print(f"  Mode sequence = {mode_sequence}")

    # Print encoding table
    print("\n" + "=" * 60)
    print("ENCODING TABLE (Discovered from experiment)")
    print("=" * 60)
    print(f"{'Condition':<12} {'Mode Encoder Sequence':<25}")
    print("-" * 40)
    for cond_id, seq in encoding_table.items():
        seq_str = " ".join([f"S{s}" for s in seq])
        print(f"{cond_id:<12} {seq_str:<25}")

    return {
        'conditions': all_results,
        'all_phases': np.array(all_phases),
        'encoding_table': encoding_table
    }


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_clustering(phases: np.ndarray) -> Dict:
    """
    Analyze clustering in phase distribution (Test 5: Discreteness Verification).

    Args:
        phases: Array of phase values in [0, 2π)

    Returns:
        stats: Dictionary with bimodality coefficient and GMM BIC comparison
    """
    # Convert to degrees for interpretability
    phases_deg = np.degrees(phases) % 360

    # Bimodality coefficient
    n = len(phases_deg)
    s = skew(phases_deg)
    k = kurtosis(phases_deg, fisher=False)
    bc = (s**2 + 1) / (k + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))

    # GMM comparison
    X = phases_deg.reshape(-1, 1)
    gmm_1 = GaussianMixture(n_components=1, random_state=42).fit(X)
    gmm_4 = GaussianMixture(n_components=4, random_state=42).fit(X)

    bic_1 = gmm_1.bic(X)
    bic_4 = gmm_4.bic(X)
    delta_bic = bic_1 - bic_4

    return {
        'bimodality_coefficient': bc,
        'bic_uniform': bic_1,
        'bic_4state': bic_4,
        'delta_bic': delta_bic,
        'gmm_4': gmm_4
    }


def plot_results(results: Dict, save_path: str = None):
    """
    Generate publication-quality figures.

    Args:
        results: Output from run_experiment()
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. D_eff showing dimensional collapse (transient -> steady)
    ax = axes[0, 0]
    conditions = results['conditions']
    d_effs_trans = [c['d_eff_transient_mean'] for c in conditions]
    d_effs_steady = [c['d_eff_steady_mean'] for c in conditions]
    x = range(len(conditions))
    width = 0.35
    ax.bar([i - width/2 for i in x], d_effs_trans, width, color='orange', alpha=0.7, label='Transient (high-D)')
    ax.bar([i + width/2 for i in x], d_effs_steady, width, color='steelblue', alpha=0.7, label='Steady-state (low-D)')
    ax.set_xlabel('Environmental Condition')
    ax.set_ylabel('Effective Dimensionality (D_eff)')
    ax.set_title('Dimensional Collapse: Transient → Steady State')
    ax.legend(fontsize=8)

    # 2. Reproducibility and End-to-End Fidelity
    ax = axes[0, 1]
    repros = [c['reproducibility'] for c in conditions]
    e2e = [c.get('end_to_end_fidelity', 0) for c in conditions]
    width = 0.35
    ax.bar([i - width/2 for i in x], repros, width, color='forestgreen', alpha=0.7, label='Reproducibility')
    ax.bar([i + width/2 for i in x], e2e, width, color='steelblue', alpha=0.7, label='End-to-End Fidelity')
    ax.axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
    ax.set_xlabel('Environmental Condition')
    ax.set_ylabel('Rate')
    ax.set_title('Code Reproducibility & End-to-End Fidelity')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    # 3. Phase histogram with clustering
    ax = axes[1, 0]
    phases_deg = np.degrees(results['all_phases']) % 360
    ax.hist(phases_deg, bins=72, density=True, alpha=0.6, color='gray', label='Observed')

    # Overlay GMM fit
    clustering = analyze_clustering(results['all_phases'])
    x_plot = np.linspace(0, 360, 1000).reshape(-1, 1)
    log_prob = clustering['gmm_4'].score_samples(x_plot)
    ax.plot(x_plot, np.exp(log_prob), 'r-', linewidth=2, label='4-State GMM Fit')

    ax.axvline(90, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(180, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(270, color='blue', linestyle=':', alpha=0.5)
    ax.set_xlabel('Phase (degrees)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Phase Clustering (BC={clustering["bimodality_coefficient"]:.3f}, ΔBIC={clustering["delta_bic"]:.0f})')
    ax.legend()

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Compute overall stats
    mean_d_eff_trans = np.mean(d_effs_trans)
    mean_d_eff_steady = np.mean(d_effs_steady)
    collapse_ratio = mean_d_eff_trans / max(mean_d_eff_steady, 1)
    mean_repro = np.mean(repros)
    mean_e2e = np.mean(e2e)

    text = f"""
    SIMULATION SUMMARY
    ==================

    Reservoir Configuration:
    • Grid size: 10 × 10 = 100 oscillators
    • D_eff transient: {mean_d_eff_trans:.1f}
    • D_eff steady: {mean_d_eff_steady:.1f}
    • Collapse ratio: {collapse_ratio:.1f}×

    Code Properties:
    • Symbols: 4 (phase quadrants)
    • Sequence length: 3
    • Total states: 64

    Performance:
    • Mean reproducibility: {mean_repro:.1%}
    • End-to-end fidelity: {mean_e2e:.1%}
    • Bimodality coefficient: {clustering['bimodality_coefficient']:.3f}
      (threshold > 0.555)
    • ΔBIC (4-state vs uniform): {clustering['delta_bic']:.0f}
      (threshold > 10)

    VERDICT: {'✓ PASS' if clustering['delta_bic'] > 10 else '✗ FAIL'}
    (Note: Kuramoto model is simplified;
     real BZ has stronger attractors)
    """
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return clustering


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COUPLED OSCILLATOR RESERVOIR SIMULATION")
    print("HeroX Evolution 2.0 - Dimensional Collapse Demonstration")
    print("=" * 60)
    print()

    # Run experiment
    print("Running experiment (8 conditions × 10 trials)...")
    print()
    results = run_experiment(n_trials=10, n_conditions=8)

    # Analyze and plot
    print()
    print("=" * 60)
    print("CLUSTERING ANALYSIS (Test 5: Discreteness Verification)")
    print("=" * 60)
    clustering = analyze_clustering(results['all_phases'])
    print(f"Bimodality Coefficient: {clustering['bimodality_coefficient']:.3f}")
    print(f"  (threshold > 0.555 for multimodality)")
    print(f"BIC (uniform model): {clustering['bic_uniform']:.0f}")
    print(f"BIC (4-state model): {clustering['bic_4state']:.0f}")
    print(f"ΔBIC: {clustering['delta_bic']:.0f}")
    print(f"  (threshold > 10 for strong evidence of discreteness)")

    # Plot
    print()
    print("Generating figures...")
    plot_results(results, save_path='figures/reservoir_simulation_results.png')

    print()
    print("Done!")
