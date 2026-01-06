# Expanded Submission Workspace

**Paper:** "From Chemistry to Ecology: Codes and Predation Emerge from Coherence Constraints in Protocell Networks"

**Targets:** Discover Life (Springer journal) and/or Evolution 2.0 Prize (HeroX)

## Files

| File | Description |
|------|-------------|
| `main_expanded.tex` | Full manuscript (26 pages) |
| `main_expanded.pdf` | Compiled PDF |
| `demo_code_emergence.py` | Two mechanisms: substrate competition + Lewis games |
| `demo_basin_structure.py` | Basin analysis + p-adic validation |
| `overnight_sims.py` | Comprehensive overnight simulations |
| `cover_letter.md` | Generic cover letter template |
| `figures/` | Symlinks to paper/figures |
| `results/` | Output from overnight_sims.py |

## Key Results

### From demo_code_emergence.py

Two independent mechanisms produce codes:
- **Substrate competition:** Hill kinetics (h=4), winner-take-most
- **Lewis signaling:** Coordination pressure alone, no supervision

**Implication:** Code emergence is generic, not mechanism-specific.

### Overnight Simulation Results (Jan 7, 2026)

1. **Basin-robustness correlation:** r = 0.80 ± 0.08 (n=20 seeds)
   - Larger basins → more robust to noise
   - Matches genetic code structure (Leu/Ser/Arg have wide basins)

2. **P-adic validation:**
   - Closest codons (d=1/32) differ only at position 3: 100%
   - Code for same amino acid: 30/32 (94%)
   - Wobble structure emerges from 2-adic metric

3. **Scaling to 64 states:**
   - Coordination: 5.5% (vs 1.6% random baseline)
   - Code coverage: 39/64 codes used (61%)
   - Genetic code uses 21/64 (33%)

4. **Honest limitation:**
   - Model explains WHY degeneracy is adaptive
   - Does NOT explain why it concentrates at position 3
   - That requires tRNA/ribosome chemistry

## What's New (Compared to Original Paper)

| Original Paper | This Expansion Adds |
|----------------|---------------------|
| Substrate competition | Lewis signaling games (independent validation) |
| 32 codes, 98% accuracy | Basin structure analysis (degeneracy = robustness) |
| Predator-prey dynamics | P-adic formalism (Khrennikov connection) |
| — | Honest limitations section |
| — | Full OoL literature engagement (Kauffman, Eigen, etc.) |
| — | Softened claims throughout |

## Quick Start

```bash
# Run demos
python3 demo_code_emergence.py
python3 demo_basin_structure.py

# Run overnight simulations
nohup python3 -u overnight_sims.py > overnight_output.log 2>&1 &

# Compile paper
pdflatex main_expanded.tex && pdflatex main_expanded.tex
```

## Companion Papers

- [Manifold Expansion](https://github.com/todd866/manifold-expansion)
- [Tracking Complexity](https://github.com/todd866/tracking-complexity)

## Status

- ✅ Paper complete (26 pages)
- ✅ Lewis games + basin + p-adic sections added
- ✅ Demo scripts working
- ✅ Cover letter template ready
- ✅ Overnight simulations complete (Jan 7, 2026)
- ✅ Figures generated from overnight results
- ⏳ Ready for Discover Life submission
