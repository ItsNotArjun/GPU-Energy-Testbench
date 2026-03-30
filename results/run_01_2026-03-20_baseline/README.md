# Run 01 — Baseline sweep — 2026-03-20

## Hardware
- GPU: RTX 5000 Ada (sm_89, 16GB GDDR6)
- Clocks: locked to base frequency

## Settings
| Parameter       | Load          | Store         |
|----------------|---------------|---------------|
| Stride (bytes) | 128           | 32            |
| Mode           | Pointer chase | Grid-stride   |
| Arch flag      | sm_89         | sm_89         |
| Max size       | 24576 MB      | 24576 MB      |

## Key findings
- Load and store use different strides — comparison is not fair
- Store DRAM bandwidth collapses above 8GB (1362 → 485 GB/s)
- Load energy per bit at DRAM: ~130 pJ/bit (inflated by stride=128 amplification)
- Store energy per bit at DRAM: ~6-27 pJ/bit (coalesced, efficient)
- At L1 (64kB): load ~4.9 pJ/bit, store ~5.4 pJ/bit — roughly comparable

## What needs to change for next run
- Fix stride to be identical for both benchmarks (use 32 for both)
- Fix ld.cu and st.cu so stride actually controls the access pattern
- Fix measure_power.py to subtract static power baseline
- See CHANGELOG.md [Unreleased] section for full list
