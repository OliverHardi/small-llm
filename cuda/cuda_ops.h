#pragma once
// Plain C++ interface to the CUDA code: raw device pointers and sizes, no CUDA types.
// ops/*.cpp includes this and calls cuda::foo(...) when a tensor is on the GPU.
// Everything runs on the default stream and is asynchronous unless noted.
//
// bf16 values are passed as uint16_t* (the raw bits); the .cu file reinterprets them.

#include <cstddef>
#include <cstdint>

namespace cuda {

// ---- memory (cuda_memory.cu) -------------------------------------------------------
void* device_alloc(size_t bytes);                       // uninitialized!
void  device_free(void* ptr);                           // null-safe
void  device_zero(void* ptr, size_t bytes);
void  copy_h2d(void* dst_device, const void* src_host, size_t bytes);
void  copy_d2h(void* dst_host, const void* src_device, size_t bytes);   // synchronizes
void  copy_d2d(void* dst_device, const void* src_device, size_t bytes);
void  synchronize();




// ---- elementwise (cuda_ops.cu) -----------------------------------------------------
void fill(float* x, float value, size_t n);
void add(const float* a, const float* b, float* out, size_t n);
void multiply(const float* a, const float* b, float* out, size_t n);
void multiply_scalar(const float* a, float s, float* out, size_t n);
void scale_inplace(float* x, float s, size_t n);
void accumulate(float* dst, const float* src, size_t n);                 // dst += src
void silu(const float* x, float* out, size_t n);
void silu_backward(const float* x, const float* grad_out, float* out, size_t n);

void cast_f32_to_bf16(const float* src, uint16_t* dst, size_t n);
void cast_bf16_to_f32(const uint16_t* src, float* dst, size_t n);

// Same math as ops::adamw_step. bias1 = 1 - beta1^t, bias2 = 1 - beta2^t.
void adamw_step(float* param, const float* grad, float* m, float* v, size_t n,
                float lr, float beta1, float beta2, float eps,
                float weight_decay, float bias1, float bias2);




// ---- data movement (cuda_ops.cu) ---------------------------------------------------
// Token ids are stored as floats, like the CPU tensors.
void embedding(const float* table, const float* ids, float* out,
               size_t num_tokens, size_t d_model);                       // out [N, D]
void embedding_backward(const float* ids, const float* grad_out, float* grad_table,
                        size_t num_tokens, size_t d_model);              // grad_table += (atomic)

void split_heads(const float* x, float* out, size_t B, size_t T, size_t H, size_t hd);  // [B,T,H*hd] -> [B,H,T,hd]
void merge_heads(const float* x, float* out, size_t B, size_t T, size_t H, size_t hd);  // [B,H,T,hd] -> [B,T,H*hd]
void transpose(const float* x, float* out, size_t rows, size_t cols);                   // [R,C] -> [C,R]

// x is [groups, T, head_dim], consecutive pairs (2i, 2i+1) rotated, like ops::rope.
// sign = +1 forward, -1 backward.
void rope(const float* x, float* out, size_t groups, size_t T, size_t head_dim,
          float theta, float sign);




// ---- matmul, cuBLAS fp32 (cuda_matmul.cu) -----------------------------------------
// All row-major. M, K, N are the logical dimensions of the product.
void matmul(const float* a, const float* b, float* c, size_t M, size_t K, size_t N);     // a[M,K]   @ b[K,N]   -> c[M,N]
void matmul_bt(const float* a, const float* b, float* c, size_t M, size_t K, size_t N);  // a[M,K]   @ b[N,K]^T -> c[M,N]
void matmul_at(const float* a, const float* b, float* c, size_t M, size_t K, size_t N);  // a[K,M]^T @ b[K,N]   -> c[M,N]
// batch independent products: a[batch,M,K] @ b[batch,K,N] -> c[batch,M,N]
void batched_matmul(const float* a, const float* b, float* c, size_t batch, size_t M, size_t K, size_t N);




// ---- row-wise kernels (cuda_reduce.cu) ---------------------------------------------
// rows = numel / D. y = w * x / sqrt(mean(x^2) + eps)
void rmsnorm(const float* x, const float* w, float* out, size_t rows, size_t D, float eps);
// writes dx; ACCUMULATES into w_grad (atomic, so tiny run-to-run differences are expected)
void rmsnorm_backward(const float* x, const float* w, const float* grad_out,
                      float* dx, float* w_grad, size_t rows, size_t D, float eps);
 
// Softmax over the last axis. causal: rows are [.., T, T] and row r is query position r % cols;
// entries with j > position come out exactly 0.
void softmax(const float* x, float* out, size_t rows, size_t cols, bool causal);
void softmax_backward(const float* probs, const float* grad_out, float* dx, size_t rows, size_t cols);
 
// Mean cross-entropy. targets are float class ids. Returns the loss on the host (SYNCHRONIZES).
float cross_entropy(const float* logits, const float* targets, size_t rows, size_t vocab);
void  cross_entropy_backward(const float* logits, const float* targets, float* grad,
                             size_t rows, size_t vocab);
 
// Sum of squares of n floats, accumulated in double. Returns on the host (SYNCHRONIZES).
float sum_squares(const float* x, size_t n);

} // namespace cuda