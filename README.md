# Why Abiogenesis Experiments Produce Building Blocks But Not Codes: An Effective Dimensionality Threshold

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Seventy years of abiogenesis research have produced amino acids, nucleotides, lipid vesicles, and ribozymes — but no codes. Why?

**Answer:** Code emergence requires sufficient *effective dimensionality* (D_eff) — not merely many chemical species, but species whose dynamics span orthogonal information channels.

```
D_eff = (Σλ)² / Σ(λ²)    # Participation ratio of output eigenvalues
```

## Key Results

- **Species count ≠ effective dimensionality**: 50 species → D_eff = 1.0, 57% accuracy. 15 species with orthogonal pathways → D_eff = 1.3, 83% accuracy.
- **D_eff > 1 threshold**: Collapsed dynamics (D_eff = 1) achieve >80% accuracy in only 35% of cases; diverse dynamics (D_eff > 1) achieve it in 71%.
- **Mechanism**: Substrate competition discretizes continuous gradients into stable boundary codes.
- **Ablation finding**: Graded competition (h=1) works as well as cooperative binding (h≥2); competitive allocation matters, not winner-take-most amplification.

## Running Simulations

```bash
# Main simulation
python code/chemistry_sim.py --compare    # 15 vs 50 species comparison
python code/chemistry_sim.py --timescale  # Test timescale separation
python code/chemistry_sim.py --full       # Full 32-environment test

# Generate figures
python code/generate_figure.py
python code/generate_bimodality_figure.py
```

## Paper

**Title:** Why Abiogenesis Experiments Produce Building Blocks But Not Codes: An Effective Dimensionality Threshold

**Target Journal:** Discover Life (Springer)

**Status:** Submitted (January 8, 2026)

**Related:**
- [HeroX Evolution 2.0 Submission](https://github.com/todd866/evolution2-prize) - Prize competition entry

## Files

| File | Description |
|------|-------------|
| `main.tex` / `main.pdf` | Paper manuscript |
| `cover_letter.tex` | Journal cover letter |
| `code/chemistry_sim.py` | Core simulation (mass-action ODEs) |
| `code/generate_figure.py` | Figure generation |
| `code/ensemble_sweep.py` | Parameter sweeps |
| `figures/` | Output figures |

## Citation

```bibtex
@article{todd2026dimensionality,
  title={Why Abiogenesis Experiments Produce Building Blocks But Not Codes: An Effective Dimensionality Threshold},
  author={Todd, Ian},
  journal={Discover Life},
  year={2026},
  note={In preparation}
}
```

## License

MIT License
