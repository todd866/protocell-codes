# HeroX Evolution 2.0 Prize Submission

**Core Insight**: Codes emerge as coordination interfaces between coupled protocellular compartments. Communication precedes information storage. The genetic code is the compression of meaning that first existed in distributed coupling fields.

## Folder Structure

```
60_heroX_evolution/
├── prize/                    # HeroX prize submission (~5 pages)
│   ├── prize_submission.tex
│   └── prize_submission.pdf
├── biosystems/               # BioSystems paper (~15 pages)
│   ├── biosystems_submission.tex
│   └── biosystems_submission.pdf
├── ig/                       # Information Geometry companion (~9 pages)
│   ├── constraint_exchange.tex
│   └── constraint_exchange.pdf
├── patent/                   # Australian provisional (25 claims)
│   ├── australian_provisional.tex
│   └── australian_provisional.pdf
├── simulation/               # Python code
│   ├── simulate.py           # Main simulation
│   ├── overnight_run.py      # Batch runner
│   └── results.npy           # Saved metrics
├── figures/
├── articles/
└── archive/
```

## Key Results (61 Vesicles, 128D)

| Metric | Value |
|--------|-------|
| Unique mappings | 32/32 (no collisions) |
| Reproducibility | 100% |
| Separation ratio | 335,361× |
| Env-Attractor correlation | 0.72 |
| Decoder accuracy | 100% (physics-only) |

## The Mechanism

**Substrate competition** (lateral inhibition) discretizes continuous dynamics:
- Output channels compete for finite metabolic resources
- Hill kinetics + Michaelis-Menten saturation
- The allocation formula is QSSA for competitive binding
- 89% of outputs saturated → emergent digitality

## Building

```bash
# Prize submission
cd prize && pdflatex prize_submission.tex

# BioSystems paper
cd biosystems && pdflatex biosystems_submission.tex

# Patent
cd patent && pdflatex australian_provisional.tex

# Simulations
python3 simulation/simulate.py          # Single run
python3 simulation/simulate.py --sweep  # 20-seed test
nohup python3 simulation/overnight_run.py &  # Background batch
```

## Submission Priority

1. **Patent** → IP Australia (~$130 AUD) - establishes priority date
2. **Prize** → HeroX (can be submitted anytime after patent)
3. **Discover Life** (formerly Origins of Life) → Full theoretical treatment

## Status (Jan 2026)

- [x] Main paper complete (`paper/main.tex`)
- [x] Cover letter complete
- [x] All simulations reproducible
- [x] Prize submission ready
- [ ] Submit to Discover Life

## License

MIT License

---

## See Also

**This project is part of the Evo2.0 cluster. See:**
- `../../CLUSTER_STATUS_EVO2.md` — Cluster status tracking (HeroX + Araudia + Social Intel)
- `../../PROJECT_INDEX.md` — Full research program inventory
- `../../CLAUDE.md` — Workflow documentation and journal strategy
- `../62_araudia_integration/` — Araudia integration (stochastic validation, winner margins)
- `../../biosystems/61_social_intelligence/` — Social Intelligence paper (why codes require coordination)
