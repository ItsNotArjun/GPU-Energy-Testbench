# Changelog

## [Unreleased]
### Changed
- Restructured repo: source files → `benchmarks/`, results → `results/run_NN_date_desc/`
- Removed compiled binaries from git tracking
- Added `.gitignore` to prevent future binary commits
- Added `benchmarks/Makefile` for one-command compilation

## [0.1.0] — 2026-03-20
### Added
- Initial replication of Delestrac et al. 2024 methodology on RTX 5000 Ada (sm_89)
- `ld.cu`: pointer-chasing load benchmark (Fisher-Yates random permutation, stride ignored)
- `st.cu`: grid-stride store benchmark (stride ignored in kernel)
- `run_sweep.py`: size sweep across L1/L2/DRAM ranges
- `measure_power.py`: NVML power measurement, total energy per bit
- First results in `results/run_01_2026-03-20_baseline/`
### Known Issues (to fix in next iteration)
- Stride argument is dead code in both CUDA kernels
- measure_power.py does not subtract static (idle) power baseline
- Load uses stride=128, store uses stride=32 — incomparable access patterns
