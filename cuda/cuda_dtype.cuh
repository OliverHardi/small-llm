#pragma once

#include <cstdint>
#include <cuda_bf16.h>

namespace cuda {

__device__ __forceinline__ float ld(const float* p, size_t i)          { return p[i]; }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, size_t i)  { return __bfloat162float(p[i]); }
__device__ __forceinline__ void  st(float* p, size_t i, float v)          { p[i] = v; }
__device__ __forceinline__ void  st(__nv_bfloat16* p, size_t i, float v)  { p[i] = __float2bfloat16(v); }

inline const __nv_bfloat16* as_bf16(const uint16_t* p) { return reinterpret_cast<const __nv_bfloat16*>(p); }
inline       __nv_bfloat16* as_bf16(uint16_t* p)       { return reinterpret_cast<__nv_bfloat16*>(p); }

} // namespace cuda
