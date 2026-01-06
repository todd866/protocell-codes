# Expanded Submission: Protocell Codes

**For GPT Review**

## Files to Review

| File | Description |
|------|-------------|
| `main_expanded.pdf` | The paper (26 pages) |
| `demo_code_emergence.py` | Two mechanisms: substrate competition + Lewis games |
| `demo_basin_structure.py` | Basin-robustness analysis + p-adic validation |
| `overnight_sims.py` | Full validation suite (scaling, statistics) |

## Paper Summary

**Title:** From Chemistry to Ecology: Codes and Predation Emerge from Coherence Constraints in Protocell Networks

**Target:** Discover Life (Springer, formerly Origins of Life)

**Core claim:** Symbolic codes and predator-prey ecology emerge from physics alone—no genetic machinery required. Two independent mechanisms (substrate competition, Lewis signaling games) produce equivalent codes, showing code emergence is generic.

## Key Results

| Finding | Quantification |
|---------|----------------|
| Code emergence | 32 environments → 32 distinguishable codes |
| Decoding accuracy | 98% (physics only) |
| Basin-robustness | r = 0.80 ± 0.08 (n=20 seeds) |
| P-adic validation | 30/32 closest pairs code same AA |
| Predator-prey | Emerges from coherence ceiling |

## Review Focus

1. **Scientific accuracy** — Are claims supported by the simulations?
2. **Clarity** — Is the mechanism clearly explained?
3. **Honest limitations** — Does the paper acknowledge what it doesn't explain?
4. **OoL literature engagement** — Does it connect to existing frameworks?

## What the Paper Does NOT Claim

- Does NOT explain why degeneracy concentrates at position 3 (wobble)
- Does NOT claim to have solved the origin of life
- Does NOT require specific chemistry—mechanism is generic

## Dev Files

Development artifacts (simulations, figures, logs) are in `dev/` subfolder.
