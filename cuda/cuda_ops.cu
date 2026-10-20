#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cuda_bf16.h>

namespace cuda {

// =====================================================================================
// Helpers
// =====================================================================================
__global__ void print_build_info() {
#ifdef __CUDA_ARCH__
    printf("CUDA device architecture: %d\n", __CUDA_ARCH__);
#endif

#ifdef __CUDA_DEBUG__
    printf("CUDA debug compilation enabled\n");
#endif
}

void print_device_build_info() {
    print_build_info<<<1, 1>>>();
    CUDA_LAUNCH_CHECK();
}

cudaEvent_t timer_start;
cudaEvent_t timer_stop;
bool timer_initialized = false;

void initCudaTimer() {
    if (!timer_initialized) {
        cudaEventCreate(&timer_start);
        cudaEventCreate(&timer_stop);
        timer_initialized = true;
    }
}

void startCudaTimer() {
    initCudaTimer();
    cudaEventRecord(timer_start);
}

float stopCudaTimer() {
    cudaEventRecord(timer_stop);
    cudaEventSynchronize(timer_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, timer_start, timer_stop);
    return ms;
}

// =====================================================================================
// Elementwise: one thread per element. `i` is size_t so >2^31 elements can't overflow.
// =====================================================================================

#define ELEMENT_INDEX(n)                                                        \
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; \
    if (i >= (n)) return;

#define LAUNCH(kernel, n, ...)                                                  \
    do {                                                                        \
        if ((n) == 0) break;                                                    \
        kernel<<<blocks_for(n), kBlockSize>>>(__VA_ARGS__);                     \
        CUDA_LAUNCH_CHECK();                                                    \
    } while (0)

__global__ void fill_kernel(float* x, float value, size_t n) {
    ELEMENT_INDEX(n)
    x[i] = value;
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
    ELEMENT_INDEX(n)
    out[i] = a[i] + b[i];
}

__global__ void multiply_kernel(const float* a, const float* b, float* out, size_t n) {
    ELEMENT_INDEX(n)
    out[i] = a[i] * b[i];
}

__global__ void multiply_scalar_kernel(const float* a, float s, float* out, size_t n) {
    ELEMENT_INDEX(n)
    out[i] = a[i] * s;
}

__global__ void scale_inplace_kernel(float* x, float s, size_t n) {
    ELEMENT_INDEX(n)
    x[i] *= s;
}

__global__ void accumulate_kernel(float* dst, const float* src, size_t n) {
    ELEMENT_INDEX(n)
    dst[i] += src[i];
}

__global__ void silu_kernel(const float* x, float* out, size_t n) {
    ELEMENT_INDEX(n)
    const float v = x[i];
    out[i] = v / (1.0f + expf(-v));
}

__global__ void silu_backward_kernel(const float* x, const float* g, float* out, size_t n) {
    ELEMENT_INDEX(n)
    const float v = x[i];
    const float s = 1.0f / (1.0f + expf(-v));
    out[i] = g[i] * s * (1.0f + v * (1.0f - s));
}

__global__ void f32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, size_t n) {
    ELEMENT_INDEX(n)
    dst[i] = __float2bfloat16(src[i]);   // round to nearest even
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* src, float* dst, size_t n) {
    ELEMENT_INDEX(n)
    dst[i] = __bfloat162float(src[i]);
}

__global__ void adamw_kernel(float* param, const float* grad, float* m, float* v, size_t n,
                             float lr, float beta1, float beta2, float eps,
                             float weight_decay, float bias1, float bias2) {
    ELEMENT_INDEX(n)
    const float g = grad[i];
    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    const float m_hat = mi / bias1;
    const float v_hat = vi / bias2;
    param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i]);
}

void fill(float* x, float value, size_t n)                  { LAUNCH(fill_kernel, n, x, value, n); }
void add(const float* a, const float* b, float* out, size_t n)      { LAUNCH(add_kernel, n, a, b, out, n); }
void multiply(const float* a, const float* b, float* out, size_t n) { LAUNCH(multiply_kernel, n, a, b, out, n); }
void multiply_scalar(const float* a, float s, float* out, size_t n) { LAUNCH(multiply_scalar_kernel, n, a, s, out, n); }
void scale_inplace(float* x, float s, size_t n)             { LAUNCH(scale_inplace_kernel, n, x, s, n); }
void accumulate(float* dst, const float* src, size_t n)     { LAUNCH(accumulate_kernel, n, dst, src, n); }
void silu(const float* x, float* out, size_t n)             { LAUNCH(silu_kernel, n, x, out, n); }
void silu_backward(const float* x, const float* g, float* out, size_t n) {
    LAUNCH(silu_backward_kernel, n, x, g, out, n);
}

void cast_f32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    LAUNCH(f32_to_bf16_kernel, n, src, reinterpret_cast<__nv_bfloat16*>(dst), n);
}
void cast_bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
    LAUNCH(bf16_to_f32_kernel, n, reinterpret_cast<const __nv_bfloat16*>(src), dst, n);
}

void adamw_step(float* param, const float* grad, float* m, float* v, size_t n,
                float lr, float beta1, float beta2, float eps,
                float weight_decay, float bias1, float bias2) {
    LAUNCH(adamw_kernel, n, param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bias1, bias2);
}

// =====================================================================================
// Data movement
// =====================================================================================

// One thread per output element; consecutive threads write consecutive addresses.
__global__ void embedding_kernel(const float* table, const float* ids, float* out,
                                 size_t total, size_t d_model) {
    ELEMENT_INDEX(total)
    const size_t token = i / d_model;
    const size_t d = i % d_model;
    const size_t row = static_cast<size_t>(ids[token]);
    out[i] = table[row * d_model + d];
}

// Several tokens can share a vocabulary row, so the accumulate must be atomic.
// (Float atomics make the summation order nondeterministic: results can differ in the
// last bits from run to run and from the CPU. Compare with a tolerance.)
__global__ void embedding_backward_kernel(const float* ids, const float* grad_out,
                                          float* grad_table, size_t total, size_t d_model) {
    ELEMENT_INDEX(total)
    const size_t token = i / d_model;
    const size_t d = i % d_model;
    const size_t row = static_cast<size_t>(ids[token]);
    atomicAdd(&grad_table[row * d_model + d], grad_out[i]);
}

void embedding(const float* table, const float* ids, float* out, size_t num_tokens, size_t d_model) {
    const size_t total = num_tokens * d_model;
    LAUNCH(embedding_kernel, total, table, ids, out, total, d_model);
}

void embedding_backward(const float* ids, const float* grad_out, float* grad_table,
                        size_t num_tokens, size_t d_model) {
    const size_t total = num_tokens * d_model;
    LAUNCH(embedding_backward_kernel, total, ids, grad_out, grad_table, total, d_model);
}

// Indexed by the OUTPUT layout, so writes are coalesced.
// [B, T, H*hd] -> [B, H, T, hd]
__global__ void split_heads_kernel(const float* x, float* out, size_t total,
                                   size_t T, size_t H, size_t hd) {
    ELEMENT_INDEX(total)
    const size_t d = i % hd;
    const size_t t = (i / hd) % T;
    const size_t h = (i / (hd * T)) % H;
    const size_t b = i / (hd * T * H);
    out[i] = x[(b * T + t) * (H * hd) + h * hd + d];
}

// [B, H, T, hd] -> [B, T, H*hd]
__global__ void merge_heads_kernel(const float* x, float* out, size_t total,
                                   size_t T, size_t H, size_t hd) {
    ELEMENT_INDEX(total)
    const size_t d = i % hd;
    const size_t h = (i / hd) % H;
    const size_t t = (i / (hd * H)) % T;
    const size_t b = i / (hd * H * T);
    out[i] = x[((b * H + h) * T + t) * hd + d];
}

// Naive version (reads are strided). Fine for now; a shared-memory tiled transpose is
// the standard optimization if profiling ever shows it matters.
__global__ void transpose_kernel(const float* x, float* out, size_t rows, size_t cols) {
    ELEMENT_INDEX(rows * cols)
    // Output is [cols, rows]; i = out_row * rows + out_col, and out[r][c] = x[c][r].
    const size_t out_row = i / rows;
    const size_t out_col = i % rows;
    out[i] = x[out_col * cols + out_row];
}

// [batch, rows, cols] -> [batch, cols, rows]. Used for q @ k^T style products in attention.
__global__ void transpose_batched_kernel(const float* x, float* out, size_t total,
                                         size_t rows, size_t cols) {
    ELEMENT_INDEX(total)
    const size_t matrix = i / (rows * cols);
    const size_t within = i % (rows * cols);
    const size_t out_row = within / rows;     // output matrix is [cols, rows]
    const size_t out_col = within % rows;
    out[i] = x[matrix * rows * cols + out_col * cols + out_row];
}

void split_heads(const float* x, float* out, size_t B, size_t T, size_t H, size_t hd) {
    const size_t total = B * T * H * hd;
    LAUNCH(split_heads_kernel, total, x, out, total, T, H, hd);
}

void merge_heads(const float* x, float* out, size_t B, size_t T, size_t H, size_t hd) {
    const size_t total = B * T * H * hd;
    LAUNCH(merge_heads_kernel, total, x, out, total, T, H, hd);
}

void transpose(const float* x, float* out, size_t rows, size_t cols) {
    const size_t total = rows * cols;
    LAUNCH(transpose_kernel, total, x, out, rows, cols);
}

void transpose_batched(const float* x, float* out, size_t batch, size_t rows, size_t cols) {
    const size_t total = batch * rows * cols;
    LAUNCH(transpose_batched_kernel, total, x, out, total, rows, cols);
}

// One thread per (group, position, pair). Same math as the CPU table version:
//   angle = pos * theta^(-2i / head_dim);  (x1, x2) -> (x1 c - x2 s, x1 s + x2 c)
// Do NOT compile with --use_fast_math: __sinf/__cosf lose accuracy at large angles.
__global__ void rope_kernel(const float* x, float* out, size_t total_pairs,
                            size_t T, size_t head_dim, float theta, float sign) {
    ELEMENT_INDEX(total_pairs)
    const size_t pairs = head_dim / 2;
    const size_t pair = i % pairs;
    const size_t pos = (i / pairs) % T;

    const float frequency = 1.0f / powf(theta, static_cast<float>(2 * pair) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * frequency;
    const float c = cosf(angle);
    const float s = sign * sinf(angle);

    const size_t base = i * 2;   // pair i starts at element 2*i of the flattened tensor
    const float x1 = x[base];
    const float x2 = x[base + 1];
    out[base]     = x1 * c - x2 * s;
    out[base + 1] = x1 * s + x2 * c;
}

void rope(const float* x, float* out, size_t groups, size_t T, size_t head_dim,
          float theta, float sign) {
    const size_t total_pairs = groups * T * (head_dim / 2);
    LAUNCH(rope_kernel, total_pairs, x, out, total_pairs, T, head_dim, theta, sign);
}

} // namespace cuda
