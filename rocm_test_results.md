# TileKernels ROCm Test Results — MI300X (gfx942)

**TileLang:** v0.1.9+rocm | **Docker:** tilelang_rocm:latest | **Date:** 2026-04-24

## Results Summary

| Module | Passed | Failed | Skipped | Total | Time |
|--------|--------|--------|---------|-------|------|
| **transpose** | 70 | 42 | 112 | 224 | 34s |
| **engram** | 18 | 20 | 38 | 76 | 124s |
| **moe** | 116 | 160 | 257 | 533 | ~6min |
| **quant** | TBD | TBD | ~1600 | ~1800 | — |
| **mhc** | ~40 | ~35 | ~10 | ~85 | — |
| **TOTAL** | **204+** | **222+** | **407+** | **833+** | — |

## MoE Detailed

| File | Pass | Fail | Skip |
|------|------|------|------|
| aux_fi | 24 | 0 | 24 |
| group_count | 12 | 0 | 12 |
| inplace_unique | 24 | 0 | 24 |
| mask_indices_by_tp | 36 | 0 | 36 |
| topk_gate | 20 | 0 | 20 |
| expand_to_fused | 0 | 25 | 24 |
| get_fused_mapping | 0 | 24 | 24 |
| normalize_weight | 0 | 12 | 12 |
| reduce_fused | 0 | 40 | 0 |
| top2_sum_gate | 0 | 11 | 33 |
| topk_sum_and_idx | 0 | 48 | 48 |

## Failure Categories

1. **FP8 E4M3 NaN** (42 transpose) — TileLang ROCm FP8 codegen produces NaN
2. **hipModuleLaunchKernel invalid arg** (~35 mhc) — kernel launch config incompatible with HIP
3. **Warp-level ops** (~60) — hardcoded warp_size=32, some fixed in our patch
4. **shared_memory_per_multiprocessor** — fixed in our patch
5. **nvcc.get_target_compute_version** — fixed in our patch

## Fixes Applied (branch rocm-support)

1. `get_best_vectorize_size()` — HIP target detection
2. `get_warp_size()` utility — returns 64 for AMD wave64
3. Warp size in moe/get_fused_mapping, moe/top2_sum_gate, engram/engram_gate
4. `get_max_smem_per_sm()` — fallback for ROCm
5. `nvcc` import guard
6. Engram gate forward threads=64 on AMD
