# Protocell Codes: From Chemistry to Ecology

**Core insight**: Codes emerge as coordination interfaces between coupled protocellular compartments. Metabolic ceilings on coherence prevent homogenization and generate predator-prey dynamics. Communication precedes information storage; ecology precedes biological complexity.

## Project Structure

```
60_heroX_evolution/
├── paper/                    # Discover Life submission (main paper)
│   └── main.tex             # "From Chemistry to Ecology" (~8 pages)
├── evo2/                     # HeroX Evolution 2.0 prize submission
│   └── prize_submission.tex  # Competition entry (~8 pages)
├── patent/                   # Australian provisional patent
│   └── australian_provisional.tex
├── preprints/                # Mathematical foundations (GitHub-only)
│   ├── manifold_expansion.tex    # Fisher rank increase under coupling
│   └── coherence_ecology.tex     # IG version of predator-prey
├── simulation/               # All simulation code
│   ├── code_emergence.py     # Code emergence (use --scale for size)
│   ├── predator_prey.py      # Predator-prey dynamics
│   └── results/              # Saved outputs
└── archive/                  # Old versions
```

## Key Results

### Code Emergence (61 vesicles, 128D)

| Metric | Result |
|--------|--------|
| Unique codes | 32/32 (no collisions) |
| Reproducibility | 98.4% |
| Separation ratio | 335,000x |
| Env-Attractor correlation | 0.72 |

### Scale Dependence

| Metric | Medium (61x128D) | Massive (169x512D) |
|--------|------------------|-------------------|
| Unique codes | 32/32 | 32/32 |
| D_eff | 6.3 | 16.7 |
| Reproducibility | 98.4% | 84.2% |

**Instability at scale is not a bug**---it's the metabolic ceiling that prevents homogenization.

### Predator-Prey Emergence

| Property | Predators | Prey |
|----------|-----------|------|
| Population | 1% (36) | 99% (4964) |
| Size N | 39 | 183 |
| Coherence C | 0.53 | 0.12 |

Size difference (144 +/- 0.8) robust across 8 random seeds.

## Running Simulations

```bash
# Code emergence (default: small scale, ~1 min)
python3 simulation/code_emergence.py

# Different scales
python3 simulation/code_emergence.py --scale=small    # 19×64D (~1 min)
python3 simulation/code_emergence.py --scale=medium   # 61×128D (~10 min)
python3 simulation/code_emergence.py --scale=large    # 127×256D (~30 min)
python3 simulation/code_emergence.py --scale=massive  # 169×512D (~2 hrs)

# Other options
python3 simulation/code_emergence.py --sweep          # 20-seed robustness
python3 simulation/code_emergence.py --coupling       # Manifold expansion test
python3 simulation/code_emergence.py --full           # Complete validation

# Overnight run (background)
nohup python3 simulation/code_emergence.py --scale=massive &

# Predator-prey ecology
python3 simulation/predator_prey.py
```

## Building Papers

```bash
# Discover Life paper
cd paper && pdflatex main.tex && pdflatex main.tex

# Evo2 prize submission
cd evo2 && pdflatex prize_submission.tex

# Patent
cd patent && pdflatex australian_provisional.tex
```

## Submission Targets

1. **Discover Life** (Springer) - Main paper combining codes + ecology
   - CAUL covered for USyd (free OA)
   - Discounted APC €1090 until Dec 2026 if cap hit
   - Formerly "Origins of Life and Evolution of Biospheres"
2. **HeroX Evolution 2.0** - Prize submission (after patent)
3. **IP Australia** - Provisional patent (~$130 AUD)

## Mathematical Foundations

Available as GitHub preprints:
- [Manifold Expansion](https://github.com/todd866/manifold-expansion) - Fisher rank increase under coupling
- [Tracking Complexity](https://github.com/todd866/tracking-complexity) - Curvature amplification in projections

## License

MIT License
