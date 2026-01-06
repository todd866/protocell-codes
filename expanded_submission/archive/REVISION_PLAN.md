# Revision Plan: GPT Review Response

## Priority 1: Must-Fix (Will Get Flagged)

### 1. Ceiling Calculation Error (Section 8.2)
**Problem:** N* = ρ(αvT)² gives 25,000, not 25-60
- With v=10μm/s, T=100s, ρ=0.1, α=0.5: (0.5×10×100)² × 0.1 = 25,000

**Fix options:**
- Use diffusion scaling (L²/D) instead of velocity (L/v)
- Or fix parameter values (ρ ~ 0.0001?)
- Or reframe what v represents

### 2. Abstract vs Body Mismatch
**Problem:** Abstract says r=0.89, body says r=0.80±0.08

**Fix:** Update abstract to r=0.80±0.08 (n=20)

---

## Priority 2: Major Clarifications

### A. Standardize "Code" Definition
**Problem:** 60-D per cycle vs 240-D codeword confusion

**Fix:** Add boxed definition:
> "In all reported decoding results, we decode using [X]-D vectors, computed as [method]."

Audit all mentions of code dimension.

### B. Coupling Role Clarification
**Problem:** κ=0 still gives 32/32 unique codes → undermines "coordination-first" framing

**Fix options:**
1. Add inter-compartment coordination metric (MI, phase-locking) that collapses at κ=0
2. Or reframe: "substrate competition yields local phenotypes; coupling stabilizes colony-level shared interfaces"

### C. Separation Ratio Inconsistency
**Problem:** 335,000× (attractor space) vs 343× (interface) - different metrics?

**Fix:**
- Rename: "Separation_attractor" vs "Separation_interface"
- Define both as formulas
- Report absolute distances too

### D. Mean-Centering Ablation
**Problem:** Analysis trick appearing as causal component

**Fix:** Move to "analysis robustness checks" or justify as modeling receiver adaptation

### E. Lewis Game Framing
**Problem:** "Mechanism-independent" is too strong - Lewis games have explicit fitness

**Fix:** Reframe as "independent validation in an abstract signaling-game framework"

### F. Predation Terminology
**Problem:** "Predation" may trigger pushback

**Fix:** Consider "raider-aggregator" or "coherent extractors vs incoherent bulk"
Define operationally: "net resource transfer via asymmetric destabilization"

---

## Priority 3: Strengthening Additions

### Add 64-State Lewis Scaling Figure
- Already have data from overnight sims
- Shows 5.5% coordination, 39/64 codes (61% coverage)
- Preempts "Lewis games only tested at 4-8 states" critique

### Add P-adic Baseline Comparison
- Compare to Hamming distance (1-nt difference)
- Or randomized nucleotide encodings
- Protects against "cherry-picked encoding" criticism

### Add Confusion Matrix
- Show where 87% same-stats receiver fails
- Are errors structured (e.g., temporal bit)?

### Consider CRN Sanity Check
- Small catalytic network with saturating kinetics
- Would defuse "neural net toy" criticism
- Lower priority - significant effort

---

## Priority 4: Small Fixes

1. **Eq. cross-reference:** Section 8.2 cites Eq. 2 but means Eq. 7
2. **Decoding pipeline:** Add 6-10 line algorithm box
3. **"Bimodality 89% saturated":** Define threshold and distribution
4. **generate_figures.py:** Don't hard-code seed index 7
5. **Align Eq.1 notation:** `+ε` vs `+1` for unbound substrate

---

## Suggested Revision Order

1. Fix ceiling calculation (critical math error)
2. Sync abstract/body numbers
3. Add boxed "code" definition + audit
4. Clarify coupling role (add metric or reframe)
5. Fix separation ratio naming
6. Add 64-state Lewis figure (already have data)
7. Add p-adic baseline
8. Small fixes

---

## Code Changes Needed

- [ ] `generate_figures.py`: Select seed by closest-to-mean r, not hardcoded
- [ ] Add Lewis 64-state figure to paper
- [ ] Add p-adic baseline comparison (Hamming vs 2-adic)
- [ ] Consider compressed Lewis game (64 states → 21 signals)
