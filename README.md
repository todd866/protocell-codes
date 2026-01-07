# Why Abiogenesis Experiments Produce Building Blocks But Not Codes: An Effective Dimensionality Threshold

**Core finding: Code emergence requires sufficient effective dimensionality (D_eff), not just many species. 50 species converging to 8 outputs → D_eff = 1.0, 57% accuracy. 15 species with orthogonal pathways → D_eff = 1.3, 83% accuracy.**

## Key Results

- **Species count ≠ effective dimensionality**: More species can mean *lower* D_eff
- **D_eff predicts code quality**: Participation ratio of output codes is the key variable
- **Timescale separation is essential**: Mixed timescales → 5× higher accuracy than uniform
- **Realistic chemistry converges**: The "asphalt problem" - products accumulate but don't create orthogonal channels
- **Formose reaction** may be viable if its autocatalytic loops create timescale separation

## Simulations

### Abstract chemistry (`chemistry_sim.py`)
Tests the D_eff hypothesis with random reaction networks:
```bash
python chemistry_sim.py --compare    # 15 vs 50 species comparison
python chemistry_sim.py --timescale  # Test timescale separation
python chemistry_sim.py --full       # Full 32-environment test
```

### Realistic prebiotic chemistry (`prebiotic_chemistry.py`)
Uses literature-derived rate constants spanning 5 orders of magnitude:
- **Fast** (formose aldol): k ~ 10² h⁻¹ (seconds)
- **Intermediate** (vesicle): k ~ 10⁻¹ h⁻¹ (hours)
- **Slow** (RNA ligation): k = 0.037 h⁻¹ (days)
- **Very slow** (peptide): k ~ 10⁻⁴ h⁻¹ (weeks)

```bash
python prebiotic_chemistry.py --quick    # 8-environment quick test
python prebiotic_chemistry.py --full     # 32-environment full test
python prebiotic_chemistry.py --ablation # Mixed vs uniform timescales
```

**Key finding**: Mixed timescales produce 5× higher accuracy than uniform timescales (31% vs 6%), even though both converge to stable products.

## Files

- `main.tex` / `main.pdf` - Paper (14 pages)
- `cover_letter.tex` / `cover_letter.pdf` - Cover letter
- `chemistry_sim.py` - Abstract mass-action ODE simulation
- `prebiotic_chemistry.py` - Realistic prebiotic chemistry simulation
- `generate_figure.py` - Generate figures from saved results
- `figures/fig_confusion_matrix.pdf` - Species comparison figure
- `figures/fig_timescale.pdf` - Timescale separation figure

## The Key Insight

Raw species count does not predict code quality. What matters is **effective dimensionality**: how many orthogonal directions the output codes actually span.

D_eff is measured as participation ratio:
```
D_eff = (Σλ)² / Σ(λ²)
```
where λ are eigenvalues of the output covariance matrix.

**Timescale separation helps** because slow reactions provide a scaffold on which fast dynamics can create orthogonal structure. This mirrors the brain's oscillatory hierarchy.

## Status

**In development.** Target: Discover Life (Springer) or BioSystems.

## Related

- `63_evox_expanded/` - Full version with Lewis games, basin structure, predator-prey
