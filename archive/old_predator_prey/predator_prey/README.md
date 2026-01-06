# Predator-Prey Dynamics from Coherence Constraints

**Ecology is not an add-on to life. Ecology is the stabilizer that makes abiogenesis possible.**

## Core Claim

Predator-prey dynamics emerge necessarily from a single constraint: **coherence is metabolically expensive and scales superlinearly with group size**.

This creates the three-phase evolutionary arc:

1. **Size regime**: Incoherent blobs compete on headcount N
2. **Coherence regime**: Small coordinated groups outcompete large ones via coordination multiplier
3. **Parasitism regime**: Coherence caps scale → extraction becomes the growth strategy

## The Key Insight

> "Predation is not teeth. Predation is asymmetric extraction enabled by coherence differentials."

High-coherence groups can't scale (metabolic ceiling), so they grow by extracting from large low-coherence groups. This isn't a biological add-on - it's a dynamical necessity.

## Why This Matters for Abiogenesis

The abiogenesis paper (Paper A) shows coherence can arise spontaneously. But reviewers can always ask: "Why doesn't everything just synchronize into one boring blob?"

This paper (Paper B) answers mechanistically:
- Coherence creates metabolic ceilings
- Ceilings prevent homogenization
- Differentiation → ecology → complexity ratchet

Paper B retroactively validates Paper A by showing the attractors are ecologically stable, not just mathematically cute.

## The Trade-off Function

Effective fitness:
```
W ~ N · R · φ(C)
```
where φ(C) is superlinear once coordination kicks in.

But coherence costs:
```
Cost ~ k · N^α · C^β  (α > 1)
```

This immediately gives:
- High C pushes toward small N
- Large N pushes toward lower C
- Small coherent groups can raid large weak groups

## Two-Attractor Ecology

- **Type B (big/weak)**: low metabolic rate, high carrying capacity, low coherence, good at occupying space
- **Type S (small/coherent)**: high metabolic rate, low carrying capacity, high coherence, good at targeted extraction

That's predator-prey without teeth.

## Simulation Results

**Validated on 2026-01-06**

The meta-oscillator model confirms the two-attractor ecology emerges from coherence constraints:

| Type | Mean N | Mean C | Count | Notes |
|------|--------|--------|-------|-------|
| **Predator** | 39 | 0.53 | ~45 (0.9%) | At coherence ceiling |
| **Prey** | 183 | 0.12 | ~4955 (99.1%) | Near max size |
| **Size Diff** | +144 | - | - | Prey 4.7× larger |

**Key findings:**
- Two-attractor structure is **robust** across all parameter variations
- Only N_ceiling (hard coherence constraint) affects size differentiation
- Extraction rate, encounter rate, coherence sensitivity all show invariant behavior
- Size difference: 143.9 ± 0.8 across 8 random seeds

**Parameter sweep results:**

| Parameter | Range | Effect on Size Diff |
|-----------|-------|---------------------|
| N_ceiling | 20-100 | 163 → 86 (linear) |
| encounter_rate | 0.01-0.2 | ~144 (invariant) |
| f (extraction) | 0.05-0.3 | ~144 (invariant) |
| kappa_C | 1-5 | ~144 (invariant) |

**Figures generated:**
- `fig1_phase_evolution.pdf` - Two-attractor emergence
- `fig2_dynamics.pdf` - Population dynamics
- `fig3_distributions.pdf` - Bimodal size/coherence distributions
- `fig4_ceiling_sweep.pdf` - Ceiling constraint shapes ecology

## Simulation Code

Located in `simulation/`:
- `overnight_run.py` - Main simulation with hard ceiling constraint
- `parameter_sweep.py` - Parameter robustness analysis
- `generate_paper_figures.py` - Publication-quality figures

**Key insight:** Soft cost penalties are insufficient for size differentiation. A **hard ceiling** (coherent oscillation breaks down above critical size) is necessary to create the two-attractor structure

## Paper Structure

### BioSystems version
- Biological framing: early life ecology, multicellularity transitions
- Emphasis on the complexity ratchet
- Connection to modern attention economies (the screen/toddler application)

### Information Geometry version
- Fisher geometry of coherence landscapes
- Curvature constraints on stable ecologies
- Geodesic separation between predator/prey strategies

## Target Venues

- **BioSystems**: Igamberdiev knows the program, biological framing
- **Information Geometry**: Pure math version, geodesic/curvature story

## Connection to Research Program

This paper extends:
- `biosystems/3_intelligence` - Observable Dimensionality Bound
- `60_heroX_evolution/biosystems` - Abiogenesis via coherence
- `60_heroX_evolution/ig` - Information geometry of evolution
- `math/15_code_formation` - Code emergence under constraint

## The Punchline

Ecology is a phase-space constraint. Predation is an emergent control strategy. Complexity is maintained because perfect coherence is metabolically unstable at scale.

This reframes:
- Early life
- Multicellularity
- Social specialization
- Attention economies

...as the same dynamical phenomenon.
