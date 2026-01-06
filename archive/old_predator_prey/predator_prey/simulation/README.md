# Predator-Prey Coherence Simulation

## Overview

Simulate the emergence of predator-prey dynamics from coherence constraints in protocell networks.

## Model Specification

### Core Parameters (Shared Across Papers)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Size exponent | γ | 1.5 | Superlinear cost in N (must be > 1) |
| Coherence exponent | ζ | 1.0 | Cost scaling in C |
| Cost coefficient | k | 0.1 | Overall cost magnitude |
| Coordination gain | a | 2.0 | Multiplier strength above threshold |
| Coordination power | m | 2.0 | Superlinearity of coordination benefit |
| Coordination threshold | C* | 0.3 | Where coordination kicks in |
| Coherence sensitivity | κ_C | 2.0 | Raid success sensitivity to C ratio |
| Size sensitivity | κ_N | 1.0 | Raid success sensitivity to N ratio |
| Extraction fraction | f | 0.2 | Resources extracted per successful raid |

### Fitness Function

```
W(N, C) = N · R̄ · φ(C) - k · N^γ · C^ζ
```

where:
```
φ(C) = 1 + a(C - C*)^m · 𝟙[C > C*]
```

### Extraction Operator

Raid success probability:
```
p_raid = σ(κ_C log(C_S/C_B) - κ_N log(N_B/N_S))
```

where σ(·) is the sigmoid function.

If raid succeeds:
```
ΔR_S = +f · R_B
ΔR_B = -f · R_B
```

### Selection Operator

Each generation:
1. Colonies with W < 0 die
2. Surviving colonies reproduce with probability ∝ max(W, 0)
3. Offspring inherit (N, C) with bounded mutation:
   - ΔN ~ N(0, σ_N²), reflected at boundaries [1, N_max]
   - ΔC ~ N(0, σ_C²), reflected at boundaries [0, 1]
4. Predator-prey encounters (local or random mixing)
5. Resource redistribution via extraction operator

### State Variables

Each colony i has:
- N_i ∈ [1, N_max]: number of compartments (size)
- C_i ∈ [0, 1]: coherence (Kuramoto order parameter)
- R_i: accumulated resources

## Simulation Protocol

### Phase 1: Baseline (No Predation)

1. Initialize M = 500 colonies with random (N, C) uniform in viable region
2. Run selection dynamics for T = 1000 generations
3. **Expected**: Population clusters at large-N/low-C (prey basin only)

### Phase 2: Introduce Coherence Advantage

1. Same setup, enable φ(C) coordination multiplier
2. Run selection dynamics
3. **Expected**: Two-cluster emergence begins

### Phase 3: Full Model with Extraction

1. Enable extraction operator (Eq. 10 in BioSystems paper)
2. Run for T = 2000 generations
3. **Expected**:
   - Two-attractor structure (predator + prey basins)
   - Population oscillations (LV-like)
   - Stable coexistence

### Phase 4: Robustness Sweep

Critical for reviewer credibility.

**Grid specification:**
- γ ∈ [1.1, 2.5] with 15 points
- ζ ∈ [0.5, 2.0] with 10 points
- 150 total configurations

**For each configuration:**
1. Run 5 random seeds
2. At t = 1000, classify outcome:
   - "two-attractor": distinct predator/prey clusters
   - "single-attractor": one dominant strategy
   - "collapse": population extinction
3. Record:
   - Fraction achieving two-attractor
   - Mean predator/prey population sizes
   - Oscillation period (if present)

**Expected result:** Two-attractor structure present for all γ > 1, robust to ζ variation.

## Figure Set

### Figure 1: Phase Diagram (BioSystems main figure)

Three panels showing population distribution in (N, C) space:
- (a) t = 0: Random initial distribution
- (b) t = 100: Clustering begins
- (c) t = 1000: Two-attractor structure

Overlays:
- Viability boundary (W = 0 contour)
- Metabolic ceiling curve
- Basin separation (dashed)

### Figure 2: Population Dynamics

- (a) Predator count N_pred(t) and prey count N_prey(t) over time
- (b) Phase portrait (N_pred vs N_prey) showing oscillation
- (c) Mean coherence per population

Caption note: "Oscillations are LV-like (phase-shifted) but mechanism differs: extraction via coherence differential, not consumption."

### Figure 3: Robustness Sweep

Heatmap in (γ, ζ) space:
- Color: fraction of runs achieving two-attractor structure
- Contour: γ = 1 boundary (critical threshold)
- Annotation: "Two-attractor structure present for all γ > 1"

### Figure 4 (IG paper): Curvature Heatmap

- (a) Gaussian curvature K(N, C) over viability region
- (b) Sign of curvature: negative (red) interior, non-negative (blue) boundaries
- (c) Geodesic distance d_g(γ) with log fit

## Implementation

### Language
Python 3.10+ with NumPy, SciPy, Matplotlib.

### Key Files

```
simulation/
├── README.md (this file)
├── config.yaml (parameters - generate from defaults)
├── model.py
│   ├── fitness(N, C, params) → W
│   ├── phi(C, params) → coordination multiplier
│   └── cost(N, C, params) → metabolic cost
├── selection.py
│   ├── select(population, params) → survivors
│   ├── reproduce(survivors, params) → offspring
│   └── mutate(colony, params) → mutated colony
├── extraction.py
│   ├── raid_probability(pred, prey, params) → p
│   └── extract(pred, prey, params) → (pred', prey')
├── simulation.py
│   ├── initialize(M, params) → population
│   ├── step(population, params) → population'
│   └── run(params) → trajectory
├── analysis.py
│   ├── classify_outcome(population) → label
│   ├── compute_oscillation_period(trajectory) → T
│   └── detect_clusters(population) → labels
├── curvature.py (for IG paper)
│   ├── metric_components(N, C, params) → g_ij
│   ├── gaussian_curvature(N, C, params) → K
│   └── geodesic_distance(P, B, params) → d_g
├── run_simulation.py
├── run_robustness_sweep.py
└── notebooks/
    ├── figure1_phase_diagram.ipynb
    ├── figure2_population_dynamics.ipynb
    ├── figure3_robustness_sweep.ipynb
    └── figure4_curvature_heatmap.ipynb
```

### Performance Targets

- Single run (500 colonies, 1000 generations): < 30 seconds
- Robustness sweep (150 configs × 5 seeds): < 2 hours
- Curvature computation (100 × 100 grid): < 10 seconds

### Reproducibility

- Fixed random seeds for all reported runs
- All parameters in `config.yaml`
- Jupyter notebooks generate figures from saved data
- Data files saved in `results/` (gitignored for size)

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pyyaml

# Run single simulation
python run_simulation.py --config config.yaml --seed 42 --output results/baseline.npz

# Run robustness sweep
python run_robustness_sweep.py --output results/robustness.npz

# Generate figures
jupyter notebook notebooks/figure1_phase_diagram.ipynb
```

## Validation Checklist

Before paper submission, verify:

- [ ] Two-attractor structure emerges from random initial conditions
- [ ] Population oscillations are phase-shifted (LV-like)
- [ ] γ > 1 is necessary and sufficient for two-attractor structure
- [ ] Results robust across ζ ∈ [0.5, 2.0]
- [ ] Curvature sign pattern matches Proposition 1 (IG paper)
- [ ] Geodesic distance scales as log(γ) (Proposition 2)
- [ ] All figures reproducible from saved data + notebooks
