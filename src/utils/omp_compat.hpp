#pragma once
//
// Portable OpenMP macros for MSVC vs GCC/Clang.
//
// MSVC only implements OpenMP 2.0:
//   - loop index must be a signed integral type
//   - no collapse(), no simd
//
// Usage:
//   OMP_PARALLEL_FOR            – simple parallel for
//   OMP_PARALLEL_FOR_SIMD       – parallel for simd  (falls back to parallel for on MSVC)
//   OMP_PARALLEL_FOR_COLLAPSE2  – parallel for collapse(2)  (falls back to parallel for on MSVC)
//   OMP_SIMD                    – #pragma omp simd  (no-op on MSVC)
//   OMP_FOR_NOWAIT              – #pragma omp for nowait
//   OMP_LOOP_T                  – loop index type  (ptrdiff_t on MSVC, size_t elsewhere)
//   OMP_CAST(n)                 – cast to OMP_LOOP_T
//

#include <cstddef>

#ifdef _MSC_VER
// MSVC: OpenMP 2.0 only – signed index, no collapse/simd
#  define OMP_PARALLEL_FOR              _Pragma("omp parallel for")
#  define OMP_PARALLEL_FOR_SCHED(s)     _Pragma("omp parallel for schedule(static)")
#  define OMP_PARALLEL_FOR_SIMD         _Pragma("omp parallel for")
#  define OMP_PARALLEL_FOR_SIMD_SCHED(s) _Pragma("omp parallel for schedule(static)")
#  define OMP_PARALLEL_FOR_COLLAPSE2    _Pragma("omp parallel for")
#  define OMP_SIMD                      /* no-op */
#  define OMP_FOR_NOWAIT                _Pragma("omp for nowait")
   using omp_idx_t = ptrdiff_t;
#  define OMP_CAST(n) static_cast<ptrdiff_t>(n)
#else
// GCC / Clang: full OpenMP 4.5+ support
#  define OMP_PARALLEL_FOR              _Pragma("omp parallel for")
#  define OMP_PARALLEL_FOR_SCHED(s)     _Pragma("omp parallel for simd schedule(static)")
#  define OMP_PARALLEL_FOR_SIMD         _Pragma("omp parallel for simd")
#  define OMP_PARALLEL_FOR_SIMD_SCHED(s) _Pragma("omp parallel for simd schedule(static)")
#  define OMP_PARALLEL_FOR_COLLAPSE2    _Pragma("omp parallel for collapse(2)")
#  define OMP_SIMD                      _Pragma("omp simd")
#  define OMP_FOR_NOWAIT                _Pragma("omp for nowait")
   using omp_idx_t = size_t;
#  define OMP_CAST(n) (n)
#endif
