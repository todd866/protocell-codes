# Experimental Protocol: BZ Oscillator Code Emergence

## Overview

This protocol demonstrates spontaneous code emergence from non-living chemistry using coupled Belousov-Zhabotinsky (BZ) oscillators in a microfluidic format. The system implements a **Dimensional Funnel** architecture: a high-dimensional reservoir (100 droplets, D_eff ~ 200-300) collapses onto a low-dimensional readout (4 symbols), physically demonstrating dimensional collapse as the mechanism of code emergence.

## Materials

### Reagents

| Reagent | Concentration | Source | Purity |
|---------|---------------|--------|--------|
| Malonic acid | 0.2 M | Sigma-Aldrich | ≥99% |
| Sodium bromate | 0.3 M | Fisher Scientific | Reagent grade |
| Sulfuric acid | 0.3 M | Fisher Scientific | Reagent grade |
| Ferroin indicator | 0.001 M | Sigma-Aldrich | 25 mM stock |
| Cerium ammonium nitrate | 0.001 M (optional) | Sigma-Aldrich | ≥98% |
| Fluorinated oil (FC-40) | Carrier phase | 3M | - |
| Pico-Surf surfactant | 2% v/v | Sphere Fluidics | - |

### Equipment

- Microfluidic chip (custom PDMS, soft lithography)
- Syringe pumps (2-4 channels)
- Temperature controller (Peltier stage, ±0.1°C)
- Light source (white LED, controllable intensity)
- Camera (USB microscope or DSLR, ≥30 fps)
- Image analysis software (ImageJ/Python)

## Microfluidic Chip Design: The Dimensional Funnel

### Architecture

To demonstrate *Dimensional Collapse* (the emergence of low-D order from high-D dynamics), the chip utilizes a **Reservoir-Projection** architecture:

- **The Reservoir (Encoder):** A macroscopic field of ~100 coupled BZ droplets. This provides the high dimensionality (D_free >> 1) required for nonergodicity defense.
- **The Bottleneck:** Diffusive coupling that forces the reservoir's state to project onto a smaller subspace.
- **The Readout (Interface):** 4 probe droplets at the reservoir corners that sample the collective field.
- **The Decoder:** A second 4-node array entrained by the readout.

### Geometry

```
┌──────────────────────────────────────────────────────────────────────────┐
│  HIGH-DIMENSIONAL RESERVOIR (Encoder)          DECODER                   │
│  (10 × 10 grid, N = 100 droplets)              (4 entrained nodes)       │
│                                                                          │
│  [R1]○ ○ ○ ○ ○ ○ ○ ○ ○ [R2]                                             │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○           Coupling      ┌──┐                      │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○            Link         │D1│                      │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○           ═══════>      ├──┤                      │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○                         │D2│                      │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○          (Diffusive     ├──┤                      │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○           junction)     │D3│                      │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○                         ├──┤                      │
│    ○ ○ ○ ○ ○ ○ ○ ○ ○ ○                         │D4│                      │
│  [R3]○ ○ ○ ○ ○ ○ ○ ○ ○ [R4]                    └──┘                      │
│                                                                          │
│  D_eff ~ 200-300                              4 readout states           │
│  (High-D Dynamics)                            (Low-D Code)               │
└──────────────────────────────────────────────────────────────────────────┘

[R1-R4]: Corner readout nodes (project reservoir state to 4 symbols)
[D1-D4]: Decoder nodes (entrain to readout, verify end-to-end transmission)
```

### Specifications

- **Reservoir:** 10 × 10 grid (100 droplets), ~5mm × 5mm area
- **Droplet diameter:** 300-500 μm
- **Droplet volume:** 10-100 nL
- **Inter-droplet spacing:** 100-200 μm (enables strong internal coupling)
- **Readout nodes:** 4 corner droplets (positions [0,0], [0,9], [9,0], [9,9])
- **Coupling channel to decoder:** 50-100 μm width (weak, directional)

### Why 100 Droplets?

The high-dimensional reservoir is not overkill—it is essential:

1. **D_eff ~ 200-300**: Each droplet contributes ~3 dynamical variables (phase, amplitude, local coupling). This provides the massive state space needed for nonergodicity defense.
2. **Error correction through physics**: If one droplet fluctuates, the other 99 pull it back. This is how biology maintains stable codes in warm, wet environments.
3. **"Simple code from complex dynamics"**: The 4-symbol readout emerges from the 100-droplet reservoir, demonstrating dimensional collapse. Judges see a simple code; the physics ensures it's robust.

### Fabrication

1. Design mask in CAD software (10×10 droplet chambers + 4 decoder chambers)
2. SU-8 master fabrication (standard photolithography)
3. PDMS casting (10:1 base:curing agent)
4. Cure 65°C for 2 hours
5. Plasma bond to glass slide
6. **Surface treatment:** Fluorinated oil with Pico-Surf (2%) to prevent droplet merging while allowing diffusive coupling of intermediates (Br⁻)

## Experimental Procedure

### Day 1: Preparation

1. **Prepare BZ stock solutions**
   - Dissolve malonic acid in deionized water
   - Prepare bromate solution
   - Dilute sulfuric acid
   - Store ferroin stock in dark

2. **Calibrate equipment**
   - Check temperature controller accuracy
   - Calibrate light intensity
   - Test camera focus and frame rate

3. **Prime microfluidic chip**
   - Flush with fluorinated oil + surfactant
   - Check for leaks
   - Verify droplet generation

### Day 2-3: Data Collection

#### For each of 64 environmental contexts:

**Step 1: Set control parameters**

| Bit | Parameter | OFF (0) | ON (1) |
|-----|-----------|---------|--------|
| 1 | Light | 0 lux | 1000 lux |
| 2 | Temperature | 20°C | 30°C |
| 3 | Catalyst | Ferroin only | Ferroin + Ce |
| 4 | Coupling | 50 μm channel | 100 μm channel |
| 5 | Initial phase | Synchronized | Random |
| 6 | Stirring | OFF | ON (convective) |

**Step 2: Generate droplets**
- Flow rate: 0.5-2 μL/min
- Oil:aqueous ratio: 2:1 to 5:1
- Wait for stable 100-droplet array

**Step 3: Initialize system**
- Allow thermal equilibration: 10 minutes
- Record baseline (no oscillation yet)

**Step 4: Start oscillations**
- Add catalyst/acid to trigger BZ reaction
- Record video for 60-90 minutes
- Typical oscillation period: 30-120 seconds

**Step 5: Dimensional Projection (Phase Detection)**

Unlike a toy model, we do not read every droplet. We demonstrate that the *collective* system projects a stable code onto the *readout* nodes.

1. **Global Order Parameter (R):**
   - Calculate the Kuramoto Order Parameter of the entire 100-droplet reservoir:
     ```
     R = |1/N × Σ exp(i×θⱼ)|
     ```
   - Wait until R > 0.7 (system has "decided" on an attractor)

2. **Readout Sampling:**
   - Measure the phase θ ONLY of the 4 Readout Droplets (R1, R2, R3, R4)
   - Compute the **Mean Readout Phase**: θ̄ = angle(Σⱼ exp(i×θⱼ))

3. **Symbol Quantization (The Digital Step):**
   - Apply the quadrant rule to θ̄:
     - S₀: 0° ≤ θ̄ < 90°
     - S₁: 90° ≤ θ̄ < 180°
     - S₂: 180° ≤ θ̄ < 270°
     - S₃: 270° ≤ θ̄ < 360°

4. **Sequence Recording:**
   - Sample at 3 time points (t, t+T, t+2T where T = oscillation period)
   - Record 3-symbol encoder sequence (e.g., "S₂S₀S₃")

**Step 6: Decoder Verification (End-to-End)**

**Critical: The decoder is PHYSICALLY COUPLED to the encoder, not just observed.**

The 4 Readout nodes (R1-R4) are connected to the 4 Decoder nodes (D1-D4) via diffusive microfluidic channels. This is not a computational step—the encoder's chemical state physically drives the decoder's dynamics through:
- **Diffusive coupling**: Br⁻ intermediates diffuse through the 50-100 μm channel
- **Light projection** (alternative): The readout region's color is optically projected onto the decoder droplets, entraining their phase

Measurement protocol:
- Measure phase of decoder nodes (D1-D4) at same time points as encoder readout
- Apply same gradient-based quantization (compass direction of chemical center of mass)
- Record 3-symbol decoder sequence
- **Pass criterion:** Encoder sequence = Decoder sequence (≥90% trials)

This verifies **end-to-end bit preservation**: the decoder receives and reproduces the same 3-symbol sequence (6 bits) as the encoder produced. The "message" is carried by physical chemistry, not by a computer.

**Step 7: Repeat**
- Reset system (flush and regenerate droplets)
- Run 10 trials minimum per configuration

### Analysis

**Encoding table construction:**

```python
encoding_table = {}
for config in range(64):
    sequences = get_encoder_sequences(config)  # List of observed sequences
    mode_sequence = mode(sequences)             # Most common sequence
    encoding_table[config] = mode_sequence
```

**End-to-end fidelity:**

```python
fidelity = []
for trial in all_trials:
    encoder_seq = trial.encoder_sequence
    decoder_seq = trial.decoder_sequence
    match = (encoder_seq == decoder_seq)
    fidelity.append(match)

end_to_end_accuracy = mean(fidelity)  # Target: > 90%
```

**Confusion matrix:**

```python
confusion = np.zeros((64, 64))
for trial in all_trials:
    true_config = trial.input_config
    observed_sequence = trial.encoder_sequence
    predicted_config = decode(observed_sequence, encoding_table)
    confusion[true_config, predicted_config] += 1
```

**Metrics:**

1. Reproducibility = Σᵢ max(histogram(sequences_i)) / Σᵢ n_trials_i (target: >90%)
2. End-to-end fidelity = encoder→decoder match rate (target: >90%)
3. Distinguishability = diagonal dominance of confusion matrix (target: >80%)
4. Mutual information = I(Input; Output) (target: >3 bits)

## Objective Correctness Tests

### Test 1: Reproducibility
Same environmental context → same encoder sequence (>90% of trials)

### Test 2: Distinguishability
Different contexts → different sequences (confusion matrix diagonal >80%)

### Test 3: End-to-End Bit Preservation
Encoder sequence = Decoder sequence (>90% of trials)
This proves 6 bits (3 symbols × 2 bits) transmitted through the system.

### Test 4: Robustness
±5% perturbation in control parameters → output unchanged (>90%)

### Test 5: Natural Discreteness (Anti-"Binned Analog" Test)
- Collect phase histogram across all trials
- Calculate bimodality coefficient (BC > 0.555 for multimodality)
- Fit 1-component vs 4-component GMM, compare BIC
- **Pass:** ΔBIC > 10 proves natural clustering, not arbitrary binning

## Expected Results

### Positive outcome (code emerges)

- Reproducibility > 90%
- End-to-end fidelity > 90%
- Distinguishability (accuracy) > 80%
- ΔBIC > 10 (natural clustering)
- Mutual information > 3 bits

### Negative outcome (no code)

- Random sequences
- No encoder-decoder correlation
- Confusion matrix uniform
- ΔBIC < 10 (uniform phase distribution)

## Troubleshooting

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| No oscillation | Wrong pH | Adjust acid concentration |
| Irregular oscillation | Temperature drift | Improve thermal control |
| No coupling | Channel blocked | Increase channel width |
| Too much coupling | All synchronized instantly | Reduce channel width |
| Poor phase detection | Low contrast | Increase ferroin concentration |
| Low reproducibility | Noise too high | Reduce vibration, improve thermal stability |

## Safety

- Sulfuric acid: Corrosive. Use PPE (gloves, goggles)
- Bromate: Oxidizer. Keep away from organics
- Work in fume hood
- Dispose as hazardous waste

## Controls

1. **Water blank**: No BZ chemistry → No distinguishable states
2. **Uncoupled reservoir**: No internal coupling → Low D_eff, no stable attractors
3. **No decoder coupling**: Decoder isolated → Random decoder sequences (proves coupling is necessary)

## Data Archiving

- Raw video files: MP4/AVI, labeled by configuration
- Phase trajectories: CSV with timestamps
- Encoding tables: JSON format
- Confusion matrices: NumPy arrays
- Laboratory notebook: Timestamped PDF

## Timeline

- Week 1: Chip fabrication (100-droplet reservoir design)
- Week 2-3: Systematic data collection (64 configs × 10 trials)
- Week 4: Analysis and table construction
- Total: ~1 month to complete

## Citation

If this protocol leads to results, cite:

Todd, I. (2026). Code Emergence Through Dimensional Collapse. HeroX Evolution 2.0 Prize Submission.
