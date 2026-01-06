"""
HIGH-DIMENSIONAL Oregonator Simulation for HeroX Evolution 2.0

This version addresses the ≥32 distinct states requirement by:
1. 8 membrane zones (4 sides + 4 corners) instead of 4
2. Each zone = independent symbol (parallel readout)
3. Lower coupling for more spatial heterogeneity
4. Single snapshot = 8 symbols = 4^8 = 65,536 theoretical states

Even with coarse-graining, this easily produces 32+ distinct codes.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# HIGH-D OREGONATOR MODEL
# =============================================================================

class OregonatorFieldHighD:
    """
    2D Oregonator with 8-ZONE MEMBRANE READOUT.

    Key changes from base version:
    - 8 membrane zones (4 sides + 4 corners)
    - Lower diffusion for more spatial heterogeneity
    - Stronger context input for more attractor diversity
    - Parallel readout: each zone is an independent symbol
    """

    def __init__(self,
                 nx: int = 60,
                 ny: int = 60,
                 dx: float = 1.0,
                 epsilon: float = 0.08,   # Slightly faster for more pattern variety
                 f: float = 1.2,          # Higher = more excitable
                 q: float = 0.008,        # Lower = more pattern formation
                 D_u: float = 0.6,        # REDUCED diffusion = less synchronization
                 D_v: float = 0.3,        # REDUCED = more spatial heterogeneity
                 membrane_width: int = 4,
                 seed: int = None):

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

        # Initialize with small noise
        self.u = 0.2 + 0.05 * np.random.rand(nx, ny)
        self.v = 0.1 + 0.05 * np.random.rand(nx, ny)

        # Membrane and interior masks
        self.membrane_mask = np.zeros((nx, ny), dtype=bool)
        self.membrane_mask[:membrane_width, :] = True
        self.membrane_mask[-membrane_width:, :] = True
        self.membrane_mask[:, :membrane_width] = True
        self.membrane_mask[:, -membrane_width:] = True
        self.interior_mask = ~self.membrane_mask

        self.history = []

    def reaction_u(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (1/self.epsilon) * (u - u**2 - self.f * v * (u - self.q) / (u + self.q))

    def reaction_v(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u - v

    def diffusion(self, field: np.ndarray, D: float) -> np.ndarray:
        return D * laplace(field, mode='nearest') / (self.dx**2)

    def apply_membrane_input(self, context_bits: List[int]):
        """
        Apply context through 8 membrane zones.
        Uses 8 bits of context for 8 independent receptor regions.
        """
        w = self.membrane_width
        nx, ny = self.nx, self.ny

        # Stronger inputs for more attractor diversity
        strength_u = 0.4
        strength_v = 0.25

        # 8 zones: 4 sides + 4 corners
        # Bits 0-3: sides (top, bottom, left, right)
        # Bits 4-7: corners (TL, TR, BL, BR)

        # Top side
        if context_bits[0]:
            self.u[:w, w:-w] += strength_u
            self.v[:w, w:-w] += strength_v

        # Bottom side
        if context_bits[1]:
            self.u[-w:, w:-w] += strength_u
            self.v[-w:, w:-w] += strength_v

        # Left side
        if context_bits[2]:
            self.u[w:-w, :w] += strength_u

        # Right side
        if context_bits[3]:
            self.u[w:-w, -w:] += strength_u

        # Top-left corner
        if context_bits[4]:
            self.u[:w, :w] += strength_u * 1.2

        # Top-right corner
        if context_bits[5]:
            self.u[:w, -w:] += strength_u * 1.2

        # Bottom-left corner
        if len(context_bits) > 6 and context_bits[6]:
            self.u[-w:, :w] += strength_u * 1.2

        # Bottom-right corner
        if len(context_bits) > 7 and context_bits[7]:
            self.u[-w:, -w:] += strength_u * 1.2

    def read_membrane_8zones(self) -> List[float]:
        """
        Read 8 membrane zones: 4 sides + 4 corners.
        Each zone gives an independent readout.
        """
        w = self.membrane_width
        nx, ny = self.nx, self.ny

        outputs = []

        # 4 sides (excluding corners)
        # Top side (center)
        u_zone = self.u[:w, w:-w]
        v_zone = self.v[:w, w:-w]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        # Bottom side (center)
        u_zone = self.u[-w:, w:-w]
        v_zone = self.v[-w:, w:-w]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        # Left side (center)
        u_zone = self.u[w:-w, :w]
        v_zone = self.v[w:-w, :w]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        # Right side (center)
        u_zone = self.u[w:-w, -w:]
        v_zone = self.v[w:-w, -w:]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        # 4 corners
        # Top-left
        u_zone = self.u[:w, :w]
        v_zone = self.v[:w, :w]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        # Top-right
        u_zone = self.u[:w, -w:]
        v_zone = self.v[:w, -w:]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        # Bottom-left
        u_zone = self.u[-w:, :w]
        v_zone = self.v[-w:, :w]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        # Bottom-right
        u_zone = self.u[-w:, -w:]
        v_zone = self.v[-w:, -w:]
        outputs.append(np.mean(u_zone) / (np.mean(v_zone) + 1e-10))

        return outputs

    def step(self, dt: float = 0.01):
        du_react = self.reaction_u(self.u, self.v)
        dv_react = self.reaction_v(self.u, self.v)
        du_diff = self.diffusion(self.u, self.D_u)
        dv_diff = self.diffusion(self.v, self.D_v)

        self.u += dt * (du_react + du_diff)
        self.v += dt * (dv_react + dv_diff)

        # Membrane damping
        membrane_damping = 0.05
        self.u[self.membrane_mask] *= (1 - membrane_damping * dt)
        self.v[self.membrane_mask] *= (1 - membrane_damping * dt)

        # Ensure positivity
        self.u = np.maximum(self.u, 1e-10)
        self.v = np.maximum(self.v, 1e-10)

    def run(self, t_max: float, dt: float = 0.01, record_interval: int = 100):
        n_steps = int(t_max / dt)
        self.history = []

        for i in range(n_steps):
            self.step(dt)
            if i % record_interval == 0:
                self.history.append((self.u.copy(), self.v.copy()))

        return self.history


# =============================================================================
# PARALLEL SYMBOL EXTRACTION (Each zone = independent symbol)
# =============================================================================

def zone_value_to_symbol(value: float, boundaries: List[float]) -> int:
    """
    Convert a zone value to a symbol using learned boundaries.
    This is data-driven discretization, not fixed quadrants.
    """
    for i, b in enumerate(boundaries):
        if value < b:
            return i
    return len(boundaries)


def learn_boundaries_from_data(all_zone_values: List[float], n_symbols: int = 4) -> List[float]:
    """
    Learn symbol boundaries from data using k-means clustering.
    This makes discretization EMERGENT, not pre-programmed.
    """
    values = np.array(all_zone_values).reshape(-1, 1)

    if len(values) < n_symbols:
        # Fallback to uniform
        vmin, vmax = np.min(values), np.max(values)
        return [vmin + (vmax - vmin) * (i + 1) / n_symbols for i in range(n_symbols - 1)]

    # K-means to find natural clusters
    kmeans = KMeans(n_clusters=n_symbols, random_state=42, n_init=10)
    kmeans.fit(values)

    # Boundaries are midpoints between cluster centers
    centers = sorted(kmeans.cluster_centers_.flatten())
    boundaries = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers) - 1)]

    return boundaries


def extract_8symbol_code(field: OregonatorFieldHighD,
                         zone_boundaries: List[List[float]]) -> Tuple[int, ...]:
    """
    Extract 8-symbol code from membrane readout.
    Each zone is an independent symbol.

    Returns tuple of 8 symbols, each in {0,1,2,3}.
    Total theoretical states: 4^8 = 65,536
    """
    outputs = field.read_membrane_8zones()

    symbols = []
    for i, value in enumerate(outputs):
        if zone_boundaries and i < len(zone_boundaries):
            symbol = zone_value_to_symbol(value, zone_boundaries[i])
        else:
            # Fallback: simple quartile
            symbol = min(3, int(value * 2))  # rough mapping
        symbols.append(symbol)

    return tuple(symbols)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_trial_highD(context_id: int,
                    nx: int = 60, ny: int = 60,
                    t_max: float = 60.0, dt: float = 0.01,
                    zone_boundaries: List[List[float]] = None,
                    n_zones: int = 4,
                    seed: int = None) -> Dict:
    """
    Run a single trial with parallel zone readout.
    n_zones: 4 (sides only) or 8 (sides + corners)
    """
    field = OregonatorFieldHighD(nx=nx, ny=ny, seed=seed)

    # Use 8 bits of context (256 input configurations)
    bits = [(context_id >> i) & 1 for i in range(8)]
    field.apply_membrane_input(bits)

    # Add small interior noise
    field.u[field.interior_mask] += 0.03 * np.random.rand(np.sum(field.interior_mask))

    # Run simulation
    history = field.run(t_max=t_max, dt=dt, record_interval=50)

    # Read final membrane state (all 8 zones)
    all_outputs = field.read_membrane_8zones()

    # Use only first n_zones
    final_outputs = all_outputs[:n_zones]

    # Extract n-symbol code
    if zone_boundaries and len(zone_boundaries) >= n_zones:
        symbols = []
        for i in range(n_zones):
            symbol = zone_value_to_symbol(final_outputs[i], zone_boundaries[i])
            symbols.append(symbol)
        code = tuple(symbols)
    else:
        # Simple fallback: quartile each zone
        code = tuple(min(3, int(v * 2)) for v in final_outputs)

    return {
        'context_id': context_id,
        'code': code,
        'raw_outputs': final_outputs,
        'final_u': field.u.copy(),
        'final_v': field.v.copy(),
    }


def run_full_experiment(n_contexts: int = 128,
                        trials_per_context: int = 5,
                        nx: int = 60, ny: int = 60,
                        t_max: float = 80.0,
                        n_zones: int = 4):
    """
    Run full experiment, learn boundaries, measure distinctness.

    n_zones: 4 or 8 - number of membrane zones to use
             4 zones = 4^4 = 256 states (more reproducible)
             8 zones = 4^8 = 65,536 states (more distinct)
    """
    theoretical_states = 4 ** n_zones
    print("=" * 70)
    print("HIGH-D OREGONATOR SIMULATION")
    print(f"{n_zones} membrane zones × 4 symbols = {theoretical_states:,} theoretical states")
    print("=" * 70)
    print(f"\nGrid: {nx}×{ny}, Contexts: {n_contexts}, Trials/context: {trials_per_context}")
    print()

    # Phase 1: Collect raw data to learn boundaries
    print("Phase 1: Collecting data to learn symbol boundaries...")
    all_zone_data = [[] for _ in range(n_zones)]

    for ctx in range(min(64, n_contexts)):  # Sample first 64 for learning
        result = run_trial_highD(ctx, nx=nx, ny=ny, t_max=t_max, n_zones=n_zones, seed=ctx)
        for i in range(n_zones):
            all_zone_data[i].append(result['raw_outputs'][i])

    # Learn boundaries for each zone
    print("Learning data-driven symbol boundaries...")
    zone_boundaries = []
    for i in range(n_zones):
        boundaries = learn_boundaries_from_data(all_zone_data[i], n_symbols=4)
        zone_boundaries.append(boundaries)
        print(f"  Zone {i}: boundaries at {[f'{b:.3f}' for b in boundaries]}")

    # Phase 2: Run full experiment with learned boundaries
    print(f"\nPhase 2: Running {n_contexts} contexts × {trials_per_context} trials...")

    results = {}
    all_codes = []

    for ctx in range(n_contexts):
        ctx_codes = []
        for trial in range(trials_per_context):
            seed = ctx * 1000 + trial
            result = run_trial_highD(ctx, nx=nx, ny=ny, t_max=t_max,
                                     zone_boundaries=zone_boundaries,
                                     n_zones=n_zones, seed=seed)
            ctx_codes.append(result['code'])
            all_codes.append(result['code'])

        # Mode code for this context
        mode_code = Counter(ctx_codes).most_common(1)[0][0]
        reproducibility = ctx_codes.count(mode_code) / len(ctx_codes)

        results[ctx] = {
            'mode_code': mode_code,
            'reproducibility': reproducibility,
            'all_codes': ctx_codes,
        }

        if ctx % 32 == 0:
            print(f"  Context {ctx}/{n_contexts}: code={mode_code}, repro={reproducibility:.0%}")

    # Phase 3: Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Count distinct codes
    all_mode_codes = [r['mode_code'] for r in results.values()]
    distinct_codes = len(set(all_mode_codes))

    # Reproducibility
    avg_repro = np.mean([r['reproducibility'] for r in results.values()])

    # Code distribution
    code_counts = Counter(all_mode_codes)

    print(f"\nDistinct codes realized: {distinct_codes}")
    print(f"  (Requirement: ≥32, Theoretical max: {theoretical_states:,})")
    print(f"Average reproducibility: {avg_repro:.1%}")
    print(f"\nMost common codes:")
    for code, count in code_counts.most_common(10):
        code_str = ''.join(str(s) for s in code)
        print(f"  {code_str}: {count} contexts")

    # Verdict
    print("\n" + "=" * 70)
    if distinct_codes >= 32:
        print(f"✓ PASS: {distinct_codes} distinct codes ≥ 32 required")
    else:
        print(f"✗ FAIL: {distinct_codes} distinct codes < 32 required")
    print("=" * 70)

    return results, zone_boundaries


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results, boundaries = run_full_experiment(
        n_contexts=256,
        trials_per_context=3,
        nx=60,
        ny=60,
        t_max=60.0
    )
