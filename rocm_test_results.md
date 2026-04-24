# TileKernels ROCm Test Results — MI300X (gfx942)

**TileLang:** v0.1.9+rocm | **Docker:** tilelang_rocm:latest | **Date:** 2026-04-24

## Results Summary

| Module | Passed | Failed | Skipped | Notes |
|--------|--------|--------|---------|-------|
| **transpose** | 70 | 42 | 112 | FP8 E4M3 NaN (TileLang upstream) |
| **engram** | 18 | 20 | 38 | shared_memory fix helped (+10); gate kernel needs more work |
| **moe** | 116 | 160 | 257 | 5/11 files fully pass; sync_warp fix applied; some kernels hang during JIT |
| **mhc** | ~40 | ~35 | ~10 | hipModuleLaunchKernel invalid arg (TileLang upstream) |
| **quant** | TBD | TBD | ~1600 | Not yet tested with fixes |
| **TOTAL** | **244+** | **257+** | **417+** | |

## MoE Detailed (with sync_warp fix)

| File | Pass | Fail | Skip | Root Cause |
|------|------|------|------|------------|
| aux_fi ✅ | 24 | 0 | 24 | — |
| group_count ✅ | 12 | 0 | 12 | — |
| inplace_unique ✅ | 24 | 0 | 24 | — |
| mask_indices_by_tp ✅ | 36 | 0 | 36 | — |
| topk_gate ✅ | 20 | 0 | 20 | — |
| expand_to_fused ❌ | 0 | 25 | 24 | JIT compile hangs (>10min) |
| get_fused_mapping ❌ | 0 | 24 | 24 | warp-level ops + sync_warp |
| normalize_weight ❌ | 0 | 12 | 12 | Numerical correctness |
| reduce_fused ❌ | 0 | 40 | 0 | sync_warp / warp ops |
| top2_sum_gate ❌ | 0 | 11 | 33 | warp-level ops |
| topk_sum_and_idx ❌ | 0 | 48 | 48 | warp-level ops |

## Fixes Applied (8 commits on branch rocm-support)

1. **get_best_vectorize_size()** — HIP target detection, no nvcc crash
2. **get_warp_size() utility** — returns 64 for AMD wave64
3. **Warp size fixes** — moe/get_fused_mapping, moe/top2_sum_gate, engram/engram_gate
4. **get_max_smem_per_sm()** — fallback for missing ROCm attribute
5. **nvcc import guard** — handles ROCm-only builds
6. **Engram gate forward** — threads=64 on AMD, vec_size=4
7. **T.sync_warp() → T.sync_threads()** — portable warp sync
8. **is_hip_target() utility** — platform detection helper

## Upstream TileLang Issues (need fixes in tile-ai/tilelang)

1. **FP8 E4M3 NaN** — ROCm codegen produces NaN for float8_e4m3fn dtype
2. **hipModuleLaunchKernel invalid argument** — some kernel launch configs fail on HIP
3. **T.sync_warp() unresolved** — no HIP equivalent in codegen
4. **Slow JIT compilation** — 30-120s per kernel on ROCm (vs <5s on CUDA)
5. **Complex kernel JIT hangs** — expand_to_fused takes >10min to compile
