#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

// Causal flash attention on contiguous bf16 device tensors laid out [B, H, T, D],
// backed by cuDNN's SDPA graphs. Build one object per (B, H, T, D) and reuse it every step:
// graph construction is slow, execution is fast.
//
// The cudnn-frontend headers are only included in cudnn_attention.cpp (pimpl), so nothing
// else in the project pays their compile time.
class CudnnAttention {
public:
    CudnnAttention(int64_t B, int64_t H, int64_t T, int64_t D);
    ~CudnnAttention();

    // q, k, v, o: bf16 [B, H, T, D]
    // stats:      fp32 [B, H, T, 1], the softmax statistics the backward pass needs
    void forward(const void* q, const void* k, const void* v,
                 void* o, void* stats, cudaStream_t stream);

    // q, k, v, o, d_o, dq, dk, dv: bf16 [B, H, T, D];  stats: fp32 [B, H, T, 1]
    void backward(const void* q, const void* k, const void* v, const void* o,
                  const void* d_o, const void* stats,
                  void* dq, void* dk, void* dv, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};