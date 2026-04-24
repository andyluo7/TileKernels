# TileKernels ROCm Test Results — MI300X (gfx942)

**TileLang:** v0.1.9+rocm | **Docker:** tilelang_rocm:latest | **Date:** 2026-04-24

## Results

| Module | Test File | Passed | Failed | Skipped | Notes |
|--------|-----------|--------|--------|---------|-------|
| **transpose** | test_transpose.py | 66 | 46 | 112 | All failures are FP8 E4M3 (NaN output) |
| **engram** | all | 8 | 30 | 38 | Needs investigation |
| **moe** | test_topk_gate.py | 20 | 0 | 20 | ✅ All non-skipped pass |
| **moe** | remaining 10 files | TBD | TBD | TBD | Some hang (JIT timeout) |
| **quant** | all | TBD | TBD | ~1600+ skip | Most skipped (SM90/SM100) |
| **mhc** | all | ~40 | ~6 | TBD | Partial results from earlier run |

## Known Issues

1. **FP8 E4M3 NaN** — All FP8 transpose tests produce NaN. Likely TileLang ROCm FP8 codegen issue.
2. **SM90/SM100 arch skips** — Many tests skip due to compute capability checks (SM90=Hopper, SM100=Blackwell). Need gfx942/gfx950 equivalents.
3. **JIT compile timeout** — First compilation of Triton kernels on ROCm takes 30-120s per kernel, causing test timeouts.

## Summary (partial)
- **94+ passed** (66 transpose + 20 moe + 8 engram)
- **76+ failed** (46 transpose FP8 + 30 engram)
- **170+ skipped** (112 transpose + 20 moe + 38 engram)
- MoE, quant, mhc: incomplete due to JIT timeouts
