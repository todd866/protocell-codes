#!/usr/bin/env python3
"""
Coupled Compartment Code Generator v7 - Core Module
====================================================

This module contains all simulation logic without CLI dependencies.
Import this module to use the simulation programmatically.

Usage:
    from code_emergence_core import configure, EncoderArray, ReceiverArray
    configure('medium')  # Set scale before creating arrays
    encoder = EncoderArray()
    ...

Author: Ian Todd
Date: January 2026
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
from scipy.stats import spearmanr
import sys

# ============================================================================
# SCALE CONFIGURATION
# ============================================================================

SCALE_PARAMS = {
    "small": {
        "N_VESICLES": 19,      # 2-ring hex
        "N_INTERNAL": 64,
        "N_READOUT": 10,
        "N_TRIALS": 20,
    },
    "medium": {
        "N_VESICLES": 61,      # 4-ring hex
        "N_INTERNAL": 128,
        "N_READOUT": 30,
        "N_TRIALS": 20,
    },
    "large": {
        "N_VESICLES": 127,     # 6-ring hex
        "N_INTERNAL": 256,
        "N_READOUT": 50,
        "N_TRIALS": 20,
    },
    "massive": {
        "N_VESICLES": 169,     # 7-ring hex
        "N_INTERNAL": 512,
        "N_READOUT": 100,
        "N_TRIALS": 30,
    },
}

# Default scale - can be changed via configure()
_current_scale = "small"

# Module-level parameters (set by configure())
N_VESICLES = SCALE_PARAMS["small"]["N_VESICLES"]
N_INTERNAL = SCALE_PARAMS["small"]["N_INTERNAL"]
N_READOUT = SCALE_PARAMS["small"]["N_READOUT"]

# Reservoir parameters
SPECTRAL_RADIUS = 0.92
LEAK_RATE = 0.25
INPUT_SCALING = 1.0
FEEDBACK_SCALING = 0.2

# Coupling
COUPLING_STRENGTH = 0.15

# Mass-action substrate competition parameters
SUBSTRATE_POOL = 10.0
CONSUMPTION_RATE = 0.5
REPLENISHMENT_RATE = 0.3
HALF_SATURATION = 1.0
HILL_COEFF = 4.0

# Readout processing flags (for ablation studies)
USE_MEAN_CENTERING = True
READOUT_GAIN = 20.0
USE_CLIPPING = True
CLIP_BOUND = 1.0

# Ablation flags
INPUT_NOISE_LEVEL = 0.0      # Set to 0.2 for ±20% noise ablation
USE_RANDOM_TOPOLOGY = False  # Randomize neighbor connections
ABLATE_TEMPORAL_CYCLES = False  # Use single cycle instead of 4

# Environment
N_ENV_BITS = 5
N_CYCLES = 4
N_EPISODES = 100

# Topology caches (rebuilt on configure)
NEIGHBORS = {}
CENTER_IDX = []
EDGE_IDX = []
TOP_IDX = []
LEFT_IDX = []
RIGHT_IDX = []


def configure(scale: str = "small"):
    """
    Configure the simulation scale. Must be called before creating arrays.

    Args:
        scale: One of 'small', 'medium', 'large', 'massive'
    """
    global _current_scale, N_VESICLES, N_INTERNAL, N_READOUT
    global NEIGHBORS, CENTER_IDX, EDGE_IDX, TOP_IDX, LEFT_IDX, RIGHT_IDX

    if scale not in SCALE_PARAMS:
        raise ValueError(f"Unknown scale: {scale}. Choose from {list(SCALE_PARAMS.keys())}")

    _current_scale = scale
    params = SCALE_PARAMS[scale]
    N_VESICLES = params["N_VESICLES"]
    N_INTERNAL = params["N_INTERNAL"]
    N_READOUT = params["N_READOUT"]

    rebuild_topology()


def rebuild_topology():
    """Rebuild topology based on current settings (call after changing USE_RANDOM_TOPOLOGY)."""
    global NEIGHBORS, CENTER_IDX, EDGE_IDX, TOP_IDX, LEFT_IDX, RIGHT_IDX

    NEIGHBORS.clear()
    if USE_RANDOM_TOPOLOGY:
        NEIGHBORS.update(build_random_neighbors(N_VESICLES))
    else:
        NEIGHBORS.update(build_hexagonal_neighbors(N_VESICLES))

    CENTER_IDX.clear()
    EDGE_IDX.clear()
    c, e = compute_center_edge_indices(N_VESICLES)
    CENTER_IDX.extend(c)
    EDGE_IDX.extend(e)
    TOP_IDX.clear()
    LEFT_IDX.clear()
    RIGHT_IDX.clear()
    t, l, r = compute_directional_indices(N_VESICLES)
    TOP_IDX.extend(t)
    LEFT_IDX.extend(l)
    RIGHT_IDX.extend(r)


def build_random_neighbors(n_vesicles: int) -> Dict[int, List[int]]:
    """Build random neighbor graph with similar average degree to hex grid (~4-6)."""
    avg_degree = 5
    neighbors = {i: [] for i in range(n_vesicles)}

    for i in range(n_vesicles):
        # Random number of neighbors
        n_neighbors = np.random.randint(3, 7)
        candidates = [j for j in range(n_vesicles) if j != i and j not in neighbors[i]]
        if len(candidates) >= n_neighbors:
            chosen = np.random.choice(candidates, size=n_neighbors, replace=False)
            for j in chosen:
                if j not in neighbors[i]:
                    neighbors[i].append(j)
                if i not in neighbors[j]:
                    neighbors[j].append(i)

    return neighbors


def get_current_scale() -> str:
    """Return the current scale name."""
    return _current_scale


# ============================================================================
# TOPOLOGY
# ============================================================================

def hex_grid_coords(n_rings: int) -> List[Tuple[float, float]]:
    """Generate hexagonal grid coordinates for n_rings around center."""
    coords = [(0.0, 0.0)]
    for ring in range(1, n_rings + 1):
        for i in range(6):
            angle = i * np.pi / 3
            for j in range(ring):
                x = ring * np.cos(angle) - j * np.cos(angle + np.pi/3)
                y = ring * np.sin(angle) - j * np.sin(angle + np.pi/3)
                coords.append((x, y))
    return coords


def build_hexagonal_neighbors(n_vesicles: int) -> Dict[int, List[int]]:
    """Build neighbor dictionary for hexagonal grid of given size."""
    n_rings = 0
    total = 1
    while total < n_vesicles:
        n_rings += 1
        total += 6 * n_rings

    coords = hex_grid_coords(n_rings)[:n_vesicles]

    neighbors = {i: [] for i in range(n_vesicles)}
    for i in range(n_vesicles):
        for j in range(i + 1, n_vesicles):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1.2:
                neighbors[i].append(j)
                neighbors[j].append(i)

    return neighbors


def compute_center_edge_indices(n_vesicles: int) -> Tuple[List[int], List[int]]:
    """Compute center and edge indices for hexagonal grid."""
    n_rings = 0
    total = 1
    while total < n_vesicles:
        n_rings += 1
        total += 6 * n_rings

    coords = hex_grid_coords(n_rings)[:n_vesicles]

    radii = [np.sqrt(x*x + y*y) for x, y in coords]
    max_r = max(radii) if radii else 1
    threshold = max_r * 0.5

    center_idx = [i for i, r in enumerate(radii) if r <= threshold]
    edge_idx = [i for i, r in enumerate(radii) if r > threshold]

    return center_idx, edge_idx


def compute_directional_indices(n_vesicles: int) -> Tuple[List[int], List[int], List[int]]:
    """Compute top, left, right indices based on hex coordinates."""
    n_rings = 0
    total = 1
    while total < n_vesicles:
        n_rings += 1
        total += 6 * n_rings

    coords = hex_grid_coords(n_rings)[:n_vesicles]

    y_vals = [c[1] for c in coords]
    y_threshold_top = np.percentile(y_vals, 70)
    top_idx = [i for i, c in enumerate(coords) if c[1] >= y_threshold_top]

    x_vals = [c[0] for c in coords]
    x_threshold_left = np.percentile(x_vals, 25)
    left_idx = [i for i, c in enumerate(coords) if c[0] <= x_threshold_left]

    x_threshold_right = np.percentile(x_vals, 75)
    right_idx = [i for i, c in enumerate(coords) if c[0] >= x_threshold_right]

    return top_idx, left_idx, right_idx


# Initialize default topology
configure("small")


# ============================================================================
# VESICLE CLASS (MASS-ACTION SUBSTRATE COMPETITION)
# ============================================================================

class MassActionVesicle:
    """
    Single compartment with mass-action substrate competition.
    """

    def __init__(self, idx: int, shared_W: np.ndarray, input_dim: int):
        self.idx = idx
        self.state = np.zeros(N_INTERNAL)
        self.W = shared_W + np.random.randn(N_INTERNAL, N_INTERNAL) * 0.01
        self.W_in = np.random.randn(N_INTERNAL, input_dim) * INPUT_SCALING
        self.W_couple = np.random.randn(N_INTERNAL, N_READOUT) * COUPLING_STRENGTH
        self.W_out = np.random.randn(N_READOUT, N_INTERNAL) * 0.4
        self.W_fb = np.random.randn(N_INTERNAL, N_READOUT) * FEEDBACK_SCALING

        self.substrate = SUBSTRATE_POOL
        self.activity = np.zeros(N_READOUT)
        self.readout = np.zeros(N_READOUT)

    def step(self, stimulus: np.ndarray, neighbor_readouts: np.ndarray, dt: float = 0.1):
        """Update with mass-action substrate competition."""
        u = self.W_in @ stimulus
        u += self.W_couple @ neighbor_readouts
        u += self.W_fb @ self.readout

        pre = self.W @ self.state + u
        self.state = (1 - LEAK_RATE) * self.state + LEAK_RATE * np.tanh(pre)

        raw = self.W_out @ self.state
        self.activity = np.maximum(0, raw)

        saturation = (self.substrate ** HILL_COEFF) / (
            HALF_SATURATION ** HILL_COEFF + self.substrate ** HILL_COEFF
        )
        consumption = CONSUMPTION_RATE * np.sum(self.activity) * saturation
        replenishment = REPLENISHMENT_RATE * (SUBSTRATE_POOL - self.substrate)

        dS = replenishment - consumption
        self.substrate = np.clip(self.substrate + dt * dS, 0.01, SUBSTRATE_POOL)

        mm_factor = self.substrate / (HALF_SATURATION + self.substrate)

        activity_powered = self.activity ** HILL_COEFF
        total_powered = np.sum(activity_powered) + 1e-6
        allocation = activity_powered / total_powered

        if USE_MEAN_CENTERING:
            mean_alloc = np.mean(allocation)
            self.readout = (allocation - mean_alloc) * mm_factor * READOUT_GAIN
        else:
            self.readout = allocation * mm_factor * READOUT_GAIN

        if USE_CLIPPING:
            self.readout = np.clip(self.readout, -CLIP_BOUND, CLIP_BOUND)

    def reset(self):
        self.state = np.random.randn(N_INTERNAL) * 0.05
        self.substrate = SUBSTRATE_POOL
        self.activity = np.zeros(N_READOUT)
        self.readout = np.zeros(N_READOUT)


# ============================================================================
# ENCODER ARRAY
# ============================================================================

class EncoderArray:
    """Encoder colony with mass-action substrate competition."""

    def __init__(self):
        W_shared = self._create_reservoir_matrix()
        self.vesicles = [MassActionVesicle(i, W_shared, N_ENV_BITS) for i in range(N_VESICLES)]

    def _create_reservoir_matrix(self) -> np.ndarray:
        W = np.random.randn(N_INTERNAL, N_INTERNAL)
        mask = np.random.random((N_INTERNAL, N_INTERNAL)) < 0.25
        W *= mask
        eigenvalues = np.linalg.eigvals(W)
        W *= SPECTRAL_RADIUS / np.max(np.abs(eigenvalues))
        return W

    def reset(self):
        for v in self.vesicles:
            v.reset()

    def step(self, stimulus_field: np.ndarray):
        all_readouts = np.array([v.readout for v in self.vesicles])
        for i, v in enumerate(self.vesicles):
            neighbor_idx = NEIGHBORS[i]
            if neighbor_idx:
                neighbor_avg = np.mean(all_readouts[neighbor_idx], axis=0)
            else:
                neighbor_avg = np.zeros(N_READOUT)
            v.step(stimulus_field[i], neighbor_avg)

    def run_to_equilibrium(self, stimulus_field: np.ndarray, n_steps: int = 70) -> np.ndarray:
        for _ in range(n_steps):
            self.step(stimulus_field)

        accumulator = np.zeros((N_VESICLES, N_READOUT))
        for _ in range(20):
            self.step(stimulus_field)
            accumulator += np.array([v.readout for v in self.vesicles])

        return accumulator / 20.0

    def emit_code(self, pattern: np.ndarray) -> np.ndarray:
        """Spatial sampling (electrode placement)."""
        center_signal = np.mean(pattern[CENTER_IDX], axis=0)
        edge_signal = np.mean(pattern[EDGE_IDX], axis=0)
        return np.concatenate([center_signal, edge_signal])

    def get_substrate_state(self) -> np.ndarray:
        return np.array([v.substrate for v in self.vesicles])


# ============================================================================
# RECEIVER ARRAY
# ============================================================================

class ReceiverArray:
    """Receiver colony - only sees code signal, never environment."""

    def __init__(self):
        W_shared = self._create_reservoir_matrix()
        code_dim = 2 * N_READOUT
        self.vesicles = [MassActionVesicle(i, W_shared, code_dim) for i in range(N_VESICLES)]

    def _create_reservoir_matrix(self) -> np.ndarray:
        W = np.random.randn(N_INTERNAL, N_INTERNAL)
        mask = np.random.random((N_INTERNAL, N_INTERNAL)) < 0.25
        W *= mask
        eigenvalues = np.linalg.eigvals(W)
        W *= SPECTRAL_RADIUS / np.max(np.abs(eigenvalues))
        return W

    def reset(self):
        for v in self.vesicles:
            v.reset()

    def step(self, code_signal: np.ndarray):
        all_readouts = np.array([v.readout for v in self.vesicles])
        for i, v in enumerate(self.vesicles):
            neighbor_idx = NEIGHBORS[i]
            if neighbor_idx:
                neighbor_avg = np.mean(all_readouts[neighbor_idx], axis=0)
            else:
                neighbor_avg = np.zeros(N_READOUT)
            v.step(code_signal, neighbor_avg)

    def run_to_equilibrium(self, code_signal: np.ndarray, n_steps: int = 70) -> np.ndarray:
        for _ in range(n_steps):
            self.step(code_signal)

        accumulator = np.zeros((N_VESICLES, N_INTERNAL))
        for _ in range(20):
            self.step(code_signal)
            accumulator += np.array([v.state for v in self.vesicles])

        return accumulator / 20.0

    def get_attractor_signature(self, attractor_state: np.ndarray) -> np.ndarray:
        return attractor_state.flatten()

    def emit_code(self) -> np.ndarray:
        pattern = np.array([v.readout for v in self.vesicles])
        center_signal = np.mean(pattern[CENTER_IDX], axis=0)
        edge_signal = np.mean(pattern[EDGE_IDX], axis=0)
        return np.concatenate([center_signal, edge_signal])


# ============================================================================
# SYMBOL EXTRACTION (matches paper Code Definition box)
# ============================================================================

def extract_symbol(readout: np.ndarray) -> int:
    """Extract discrete symbol from readout vector (argmax)."""
    return int(np.argmax(readout))


def extract_symbol_from_pattern(pattern: np.ndarray) -> int:
    """Extract symbol from vesicle colony pattern."""
    colony_readout = np.mean(pattern, axis=0)
    return extract_symbol(colony_readout)


def extract_symbol_from_code(code: np.ndarray) -> int:
    """Extract symbol from emitted code signal."""
    center = code[:N_READOUT]
    edge = code[N_READOUT:]
    combined = (center + edge) / 2.0
    return extract_symbol(combined)


# ============================================================================
# ENVIRONMENT GENERATOR
# ============================================================================

def generate_stimulus_field(config_bits: Tuple[int, ...], cycle: int) -> np.ndarray:
    """Geometric constraints: center shielded, edge exposed."""
    base = np.array(config_bits, dtype=float) * 2 - 1

    # Temporal modulation (disabled if ABLATE_TEMPORAL_CYCLES)
    if not ABLATE_TEMPORAL_CYCLES:
        cycle_phase = cycle * np.pi / 2
        temporal_mod = 0.1 * np.sin(cycle_phase + np.arange(N_ENV_BITS) * np.pi / 3)
        base = base + temporal_mod

    field = np.zeros((N_VESICLES, N_ENV_BITS))
    for i in range(N_VESICLES):
        local = base.copy()

        if i in CENTER_IDX:
            local[0] *= 0.7
            local[2] *= 1.3
        else:
            local[0] *= 1.2
            local[2] *= 0.8

        if i in TOP_IDX:
            local[1] *= 1.2
        else:
            local[1] *= 0.85

        if i in LEFT_IDX:
            local[3] *= 1.15
            local[4] *= 0.9
        elif i in RIGHT_IDX:
            local[3] *= 0.85
            local[4] *= 1.1

        field[i] = local

    # Add input noise if enabled
    if INPUT_NOISE_LEVEL > 0:
        noise = np.random.uniform(-INPUT_NOISE_LEVEL, INPUT_NOISE_LEVEL, field.shape)
        field = field * (1 + noise)

    return field


# ============================================================================
# METRICS
# ============================================================================

def compute_effective_dimensionality(data: np.ndarray) -> float:
    """Participation ratio: (sum λ)² / sum(λ²)"""
    if len(data) < 2:
        return 0.0
    centered = data - np.mean(data, axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    total = np.sum(eigenvalues)
    if total < 1e-10:
        return 0.0
    return (total ** 2) / np.sum(eigenvalues ** 2)


def compute_confusion_matrix(true_labels: List[int], pred_labels: List[int], n_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(seed: int = 42, verbose: bool = True) -> Dict:
    """Run the main code emergence experiment."""
    np.random.seed(seed)

    if verbose:
        print("=" * 70)
        print("MASS-ACTION SUBSTRATE COMPETITION (v7)")
        print("=" * 70)
        print(f"\nScale: {_current_scale} ({N_VESICLES} vesicles × {N_INTERNAL}D)")
        print(f"Seed: {seed}")
        print(f"Environment: {N_ENV_BITS} bits = {2**N_ENV_BITS} states")
        print()

    encoder = EncoderArray()
    receiver = ReceiverArray()

    env_to_attractors = defaultdict(list)

    for ep in range(N_EPISODES):
        if verbose and ep % 25 == 0:
            print(f"  Episode {ep}/{N_EPISODES}...", file=sys.stderr)

        env_state = np.random.randint(0, 2**N_ENV_BITS)
        config_bits = tuple(int(x) for x in f"{env_state:0{N_ENV_BITS}b}")

        encoder.reset()
        receiver.reset()

        for cycle in range(N_CYCLES):
            stimulus_field = generate_stimulus_field(config_bits, cycle)
            encoder_pattern = encoder.run_to_equilibrium(stimulus_field)
            code_signal = encoder.emit_code(encoder_pattern)
            receiver_attractor = receiver.run_to_equilibrium(code_signal)

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
    separation_ratio = between_mean / within_mean if within_mean > 0 else float('inf')

    # Correlation
    env_list = list(env_states)
    n_envs = len(env_list)
    env_distances = []
    attractor_distances = []

    for i in range(n_envs):
        for j in range(i + 1, n_envs):
            bits_i = [int(x) for x in f"{env_list[i]:0{N_ENV_BITS}b}"]
            bits_j = [int(x) for x in f"{env_list[j]:0{N_ENV_BITS}b}"]
            env_dist = sum(a != b for a, b in zip(bits_i, bits_j))
            env_distances.append(env_dist)

            mean_i = np.mean(env_to_attractors[env_list[i]], axis=0)
            mean_j = np.mean(env_to_attractors[env_list[j]], axis=0)
            att_dist = np.linalg.norm(mean_i - mean_j)
            attractor_distances.append(att_dist)

    if len(env_distances) >= 3:
        corr, pval = spearmanr(env_distances, attractor_distances)
    else:
        corr = 0

    # Effective dimensionality
    all_encoder_outputs = []
    encoder.reset()
    for env_state in range(2**N_ENV_BITS):
        config_bits = tuple(int(x) for x in f"{env_state:0{N_ENV_BITS}b}")
        for cycle in range(N_CYCLES):
            stimulus_field = generate_stimulus_field(config_bits, cycle)
            pattern = encoder.run_to_equilibrium(stimulus_field)
            code = encoder.emit_code(pattern)
            all_encoder_outputs.append(code)

    encoder_outputs = np.array(all_encoder_outputs)
    d_eff = compute_effective_dimensionality(encoder_outputs)

    if verbose:
        print(f"\nResults:")
        print(f"  Separation ratio: {separation_ratio:.2f}x")
        print(f"  Env-Attractor correlation: {corr:.3f}")
        print(f"  Unique environments: {len(env_states)}")
        print(f"  Effective dimensionality (D_eff): {d_eff:.2f}")

    return {
        'seed': seed,
        'separation_ratio': separation_ratio,
        'correlation': corr,
        'n_environments': len(env_states),
        'within_mean': within_mean,
        'between_mean': between_mean,
        'd_eff': d_eff,
    }


def run_balanced_evaluation(n_repeats: int = 5, seed: int = 42, verbose: bool = True) -> Dict:
    """
    Balanced evaluation: all 32 environments × n_repeats.

    Returns metrics computed properly:
    - separation_attractor: in receiver attractor space
    - separation_interface: in transmitted code space (60D)
    - reproducibility: fraction of repeats producing same output
    - decoding_accuracy: from confusion matrix
    - d_eff: effective dimensionality
    - bimodality: fraction of readouts in saturated range
    """
    np.random.seed(seed)
    n_envs = 2 ** N_ENV_BITS
    n_cycles = 1 if ABLATE_TEMPORAL_CYCLES else N_CYCLES

    encoder = EncoderArray()
    receiver = ReceiverArray()

    # Storage for all codes
    env_to_interface_codes = {i: [] for i in range(n_envs)}
    env_to_attractor_codes = {i: [] for i in range(n_envs)}
    all_readouts = []

    for repeat in range(n_repeats):
        for env_idx in range(n_envs):
            config_bits = tuple(int(x) for x in f"{env_idx:0{N_ENV_BITS}b}")
            encoder.reset()
            receiver.reset()

            interface_codes = []
            for cycle in range(n_cycles):
                stimulus = generate_stimulus_field(config_bits, cycle)
                pattern = encoder.run_to_equilibrium(stimulus)
                code = encoder.emit_code(pattern)
                interface_codes.append(code.copy())

                # Collect readouts for bimodality
                for v in encoder.vesicles:
                    all_readouts.extend(v.readout.tolist())

                # Run receiver
                receiver.run_to_equilibrium(code)

            # Store full interface code (concatenated across cycles)
            full_interface = np.concatenate(interface_codes)
            env_to_interface_codes[env_idx].append(full_interface)

            # Store attractor signature
            attractor = np.concatenate([v.readout for v in receiver.vesicles])
            env_to_attractor_codes[env_idx].append(attractor)

    # Compute separation in INTERFACE space
    interface_within = []
    interface_between = []
    interface_centroids = {}
    for env in range(n_envs):
        codes = env_to_interface_codes[env]
        interface_centroids[env] = np.mean(codes, axis=0)
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                interface_within.append(np.linalg.norm(codes[i] - codes[j]))

    for i in range(n_envs):
        for j in range(i + 1, n_envs):
            interface_between.append(np.linalg.norm(interface_centroids[i] - interface_centroids[j]))

    sep_interface = np.mean(interface_between) / np.mean(interface_within) if interface_within else float('inf')

    # Compute separation in ATTRACTOR space
    attractor_within = []
    attractor_between = []
    attractor_centroids = {}
    for env in range(n_envs):
        codes = env_to_attractor_codes[env]
        attractor_centroids[env] = np.mean(codes, axis=0)
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                attractor_within.append(np.linalg.norm(codes[i] - codes[j]))

    for i in range(n_envs):
        for j in range(i + 1, n_envs):
            attractor_between.append(np.linalg.norm(attractor_centroids[i] - attractor_centroids[j]))

    sep_attractor = np.mean(attractor_between) / np.mean(attractor_within) if attractor_within else float('inf')

    # Compute decoding accuracy (nearest centroid in interface space)
    correct = 0
    total = 0
    for true_env in range(n_envs):
        for code in env_to_interface_codes[true_env]:
            dists = {e: np.linalg.norm(code - interface_centroids[e]) for e in range(n_envs)}
            pred_env = min(dists, key=dists.get)
            if pred_env == true_env:
                correct += 1
            total += 1

    decoding_accuracy = correct / total if total > 0 else 0

    # Compute reproducibility (using discretized codes via sign pattern)
    reproducibility_scores = []
    for env in range(n_envs):
        codes = env_to_interface_codes[env]
        if len(codes) >= 2:
            # Discretize to sign pattern
            patterns = [tuple(np.sign(c).astype(int).tolist()) for c in codes]
            # Most common pattern
            counter = Counter(patterns)
            most_common_count = counter.most_common(1)[0][1]
            reproducibility_scores.append(most_common_count / len(codes))

    reproducibility = np.mean(reproducibility_scores) if reproducibility_scores else 0

    # Compute bimodality
    readouts = np.array(all_readouts)
    bimodality = np.sum(np.abs(readouts) > 0.5) / len(readouts) if len(readouts) > 0 else 0

    # Compute D_eff
    all_codes = []
    for env in range(n_envs):
        all_codes.extend(env_to_interface_codes[env])
    d_eff = compute_effective_dimensionality(np.array(all_codes))

    # Correlation
    env_labels = []
    attractor_flat = []
    for env in range(n_envs):
        for att in env_to_attractor_codes[env]:
            env_labels.append(env)
            attractor_flat.append(np.linalg.norm(att))
    corr, _ = spearmanr(env_labels, attractor_flat) if len(env_labels) > 2 else (0, 1)

    if verbose:
        print(f"Balanced evaluation ({n_envs} envs × {n_repeats} repeats):")
        print(f"  Separation (interface): {sep_interface:.1f}×")
        print(f"  Separation (attractor): {sep_attractor:.1f}×")
        print(f"  Decoding accuracy: {100*decoding_accuracy:.1f}%")
        print(f"  Reproducibility: {100*reproducibility:.1f}%")
        print(f"  Bimodality: {100*bimodality:.1f}%")
        print(f"  D_eff: {d_eff:.2f}")
        print(f"  Correlation: {corr:.3f}")

    return {
        'separation_interface': sep_interface,
        'separation_attractor': sep_attractor,
        'decoding_accuracy': decoding_accuracy,
        'reproducibility': reproducibility,
        'bimodality': bimodality,
        'd_eff': d_eff,
        'correlation': corr,
        'n_envs': n_envs,
        'n_repeats': n_repeats,
    }


def run_systematic_trials(n_trials: int = 10, seed: int = 42,
                          return_confusion: bool = False) -> Dict:
    """
    Systematic trials with confusion matrix support.
    """
    print("=" * 70)
    print("SYSTEMATIC TRIALS (All 32 configs)")
    print("=" * 70)
    np.random.seed(seed)

    encoder = EncoderArray()
    receiver = ReceiverArray()

    config_to_enc_codes = {i: [] for i in range(2**N_ENV_BITS)}
    config_to_rec_codes = {i: [] for i in range(2**N_ENV_BITS)}
    config_to_enc_symbols = {i: [] for i in range(2**N_ENV_BITS)}
    config_to_rec_symbols = {i: [] for i in range(2**N_ENV_BITS)}

    print(f"Running {n_trials} trials per config...\n")

    for trial in range(n_trials):
        if trial % 5 == 0:
            print(f"  Trial {trial}/{n_trials}...", file=sys.stderr)

        for config_idx in range(2**N_ENV_BITS):
            config_bits = tuple(int(x) for x in f"{config_idx:0{N_ENV_BITS}b}")

            encoder.reset()
            receiver.reset()

            enc_codes = []
            rec_codes = []
            enc_symbols = []
            rec_symbols = []

            for cycle in range(N_CYCLES):
                stimulus_field = generate_stimulus_field(config_bits, cycle)
                enc_pattern = encoder.run_to_equilibrium(stimulus_field)
                enc_code = encoder.emit_code(enc_pattern)
                enc_codes.append(enc_code.copy())
                enc_symbols.append(extract_symbol_from_code(enc_code))

                receiver.run_to_equilibrium(enc_code)
                rec_code = receiver.emit_code()
                rec_codes.append(rec_code.copy())
                rec_symbols.append(extract_symbol_from_code(rec_code))

            config_to_enc_codes[config_idx].append(enc_codes)
            config_to_rec_codes[config_idx].append(rec_codes)
            config_to_enc_symbols[config_idx].append(tuple(enc_symbols))
            config_to_rec_symbols[config_idx].append(tuple(rec_symbols))

    # Encoder reproducibility
    print("\n" + "-" * 70)
    print("ENCODER REPRODUCIBILITY (argmax symbols)")
    print("-" * 70)

    repro_scores = []
    canonical_enc_symbols = {}

    for config_idx in range(2**N_ENV_BITS):
        symbol_seqs = config_to_enc_symbols[config_idx]
        seq_counts = Counter(symbol_seqs)
        most_common_seq, count = seq_counts.most_common(1)[0]
        repro = count / len(symbol_seqs)
        repro_scores.append(repro)
        canonical_enc_symbols[config_idx] = most_common_seq

    avg_repro = np.mean(repro_scores)
    min_repro = np.min(repro_scores)
    print(f"  Average: {100*avg_repro:.1f}%")
    print(f"  Minimum: {100*min_repro:.1f}%")
    print(f"  100% configs: {sum(r == 1.0 for r in repro_scores)}/32")

    unique_codes = set(canonical_enc_symbols.values())
    collisions = 32 - len(unique_codes)
    print(f"\n  Unique 4-symbol sequences: {len(unique_codes)}/32")
    if collisions > 0:
        print(f"  ⚠ {collisions} collisions detected")

    # Decoding with confusion matrix
    print("\n" + "-" * 70)
    print("DECODING ACCURACY (with confusion matrix)")
    print("-" * 70)

    true_labels = []
    pred_labels = []

    for true_config in range(2**N_ENV_BITS):
        for trial_idx, rec_codes in enumerate(config_to_rec_codes[true_config]):
            best_config = None
            best_dist = float('inf')

            for candidate_config in range(2**N_ENV_BITS):
                mean_enc = [np.mean([seq[c] for seq in config_to_enc_codes[candidate_config]], axis=0)
                            for c in range(N_CYCLES)]
                dist = sum(np.linalg.norm(rec_codes[c] - mean_enc[c]) for c in range(N_CYCLES))
                if dist < best_dist:
                    best_dist = dist
                    best_config = candidate_config

            true_labels.append(true_config)
            pred_labels.append(best_config)

    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    print(f"  Decoding accuracy: {100*accuracy:.1f}%")
    print(f"  (Chance = {100/32:.1f}%)")

    # Confusion matrix
    cm = compute_confusion_matrix(true_labels, pred_labels, 32)

    # Print summary stats from confusion matrix
    diagonal = np.diag(cm)
    per_class_accuracy = diagonal / np.maximum(cm.sum(axis=1), 1)
    print(f"  Per-class accuracy: {100*np.mean(per_class_accuracy):.1f}% mean, {100*np.min(per_class_accuracy):.1f}% min")

    # Off-diagonal errors
    off_diagonal = cm.sum() - diagonal.sum()
    print(f"  Total errors: {off_diagonal} / {cm.sum()}")

    result = {
        'avg_enc_reproducibility': avg_repro,
        'min_enc_reproducibility': min_repro,
        'collisions': collisions,
        'decoding_accuracy': accuracy,
        'per_class_accuracy_mean': np.mean(per_class_accuracy),
        'per_class_accuracy_min': np.min(per_class_accuracy),
    }

    if return_confusion:
        result['confusion_matrix'] = cm
        result['true_labels'] = true_labels
        result['pred_labels'] = pred_labels

    return result


def run_codebook(seed: int = 42) -> Dict[int, Tuple[int, ...]]:
    """Generate full encoding table."""
    print("=" * 70)
    print("ENCODING TABLE (Full Codebook)")
    print("=" * 70)
    np.random.seed(seed)

    encoder = EncoderArray()

    print(f"\n{'Config':<8} {'Binary':<8} {'S1':<6} {'S2':<6} {'S3':<6} {'S4':<6} {'Sequence':<16}")
    print("-" * 70)

    codebook = {}
    for config_idx in range(2**N_ENV_BITS):
        config_bits = tuple(int(x) for x in f"{config_idx:0{N_ENV_BITS}b}")
        encoder.reset()

        symbols = []
        for cycle in range(N_CYCLES):
            stimulus_field = generate_stimulus_field(config_bits, cycle)
            pattern = encoder.run_to_equilibrium(stimulus_field)
            code = encoder.emit_code(pattern)
            symbol = extract_symbol_from_code(code)
            symbols.append(symbol)

        binary_str = ''.join(str(b) for b in config_bits)
        seq_str = '-'.join(str(s) for s in symbols)
        codebook[config_idx] = tuple(symbols)
        print(f"{config_idx:<8} {binary_str:<8} {symbols[0]:<6} {symbols[1]:<6} {symbols[2]:<6} {symbols[3]:<6} {seq_str:<16}")

    unique_seqs = set(codebook.values())
    print(f"\nUnique 4-symbol sequences: {len(unique_seqs)}/32")
    print(f"Alphabet size: {N_READOUT} channels (0 to {N_READOUT-1})")

    return codebook
