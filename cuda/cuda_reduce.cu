#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cfloat>
#include <cmath>

#include "cuda/cuda_dtype.cuh"

// Row-wise kernels
// one block per row, 256 threads stride across the row
// then combine the threads' partial results with a block reduction

namespace cuda {

namespace {

constexpr unsigned kFullMask = 0xffffffffu;

template <typename T>
__device__ T warp_sum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(kFullMask, v, offset);
    return v;
}

__device__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(kFullMask, v, offset));
    return v;
}

// Result is returned to EVERY thread in the block.
template <typename T>
__device__ T block_sum(T v) {
    __shared__ T shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    v = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();

    const T result = shared[0];
    __syncthreads();
    return result;
}

__device__ float block_max(float v) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    v = warp_max(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    v = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();

    const float result = shared[0];
    __syncthreads();
    return result;
}

#define ROW_LAUNCH(kernel, rows, ...)                                           \
    do {                                                                        \
        if ((rows) == 0) break;                                                 \
        kernel<<<static_cast<unsigned>(rows), kBlockSize>>>(__VA_ARGS__);       \
        CUDA_LAUNCH_CHECK();                                                    \
    } while (0)

// ---- RMSNorm -----------------------------------------------------------------------

template <typename EO>
__global__ void rmsnorm_kernel(const float* x, const float* w, EO* out, size_t D, float eps) {
    const size_t base = blockIdx.x * D;

    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += blockDim.x) sum_sq += x[base + i] * x[base + i];
    sum_sq = block_sum(sum_sq);

    const float inv_rms = 1.0f / sqrtf(sum_sq / static_cast<float>(D) + eps);
    for (size_t i = threadIdx.x; i < D; i += blockDim.x) st(out, base + i, w[i] * x[base + i] * inv_rms);
}

// dx_i = r * w_i g_i - (r^3 / D) * x_i * sum_j(w_j g_j x_j);  dw_i += g_i x_i r
template <typename EG>
__global__ void rmsnorm_backward_kernel(const float* x, const float* w, const EG* g,
                                        float* dx, float* w_grad, size_t D, float eps) {
    const size_t base = blockIdx.x * D;

    float sum_sq = 0.0f, dot = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += blockDim.x) {
        const float xi = x[base + i];
        sum_sq += xi * xi;
        dot += w[i] * ld(g, base + i) * xi;
    }
    sum_sq = block_sum(sum_sq);
    dot = block_sum(dot);

    const float inv_rms = 1.0f / sqrtf(sum_sq / static_cast<float>(D) + eps);
    const float coeff = inv_rms * inv_rms * inv_rms * dot / static_cast<float>(D);

    for (size_t i = threadIdx.x; i < D; i += blockDim.x) {
        const float xi = x[base + i];
        const float gi = ld(g, base + i);
        dx[base + i] = inv_rms * w[i] * gi - coeff * xi;
        // rows * D atomics potentially slow
        atomicAdd(&w_grad[i], gi * xi * inv_rms);
    }
}

// ---- softmax -----------------------------------------------------------------------

__global__ void softmax_kernel(const float* x, float* out, size_t cols, bool causal) {
    const size_t base = blockIdx.x * cols;
    const size_t valid = causal ? (blockIdx.x % cols) + 1 : cols;   // query position = row % T

    float max_value = -FLT_MAX;
    for (size_t j = threadIdx.x; j < valid; j += blockDim.x) max_value = fmaxf(max_value, x[base + j]);
    max_value = block_max(max_value);

    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < valid; j += blockDim.x) {
        const float e = expf(x[base + j] - max_value);
        out[base + j] = e;
        sum += e;
    }
    sum = block_sum(sum);

    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        out[base + j] = (j < valid) ? out[base + j] / sum : 0.0f;
    }
}

// dx = p * (g - sum(g * p))
__global__ void softmax_backward_kernel(const float* p, const float* g, float* dx, size_t cols) {
    const size_t base = blockIdx.x * cols;

    float dot = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) dot += g[base + j] * p[base + j];
    dot = block_sum(dot);

    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) dx[base + j] = p[base + j] * (g[base + j] - dot);
}

// ---- cross entropy -----------------------------------------------------------------

// per row loss: log-sum-exp(logits) - logits[target].
__global__ void cross_entropy_kernel(const float* logits, const float* targets,
                                     float* row_loss, size_t vocab) {
    const size_t base = blockIdx.x * vocab;

    float max_logit = -FLT_MAX;
    for (size_t j = threadIdx.x; j < vocab; j += blockDim.x) max_logit = fmaxf(max_logit, logits[base + j]);
    max_logit = block_max(max_logit);

    float sum_exp = 0.0f;
    for (size_t j = threadIdx.x; j < vocab; j += blockDim.x) sum_exp += expf(logits[base + j] - max_logit);
    sum_exp = block_sum(sum_exp);

    if (threadIdx.x == 0) {
        const size_t target = static_cast<size_t>(targets[blockIdx.x]);
        row_loss[blockIdx.x] = (max_logit + logf(sum_exp)) - logits[base + target];
    }
}

__global__ void mean_kernel(const float* values, size_t n, float* out) {
    double total = 0.0;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) total += values[i];
    total = block_sum(total);
    if (threadIdx.x == 0) *out = static_cast<float>(total / static_cast<double>(n));
}

// grad = (softmax(logits) - onehot(target)) / N
template <typename EG>
__global__ void cross_entropy_backward_kernel(const float* logits, const float* targets,
                                              EG* grad, size_t vocab, float inv_n) {
    const size_t base = blockIdx.x * vocab;
    const size_t target = static_cast<size_t>(targets[blockIdx.x]);

    float max_logit = -FLT_MAX;
    for (size_t j = threadIdx.x; j < vocab; j += blockDim.x) max_logit = fmaxf(max_logit, logits[base + j]);
    max_logit = block_max(max_logit);

    float sum_exp = 0.0f;
    for (size_t j = threadIdx.x; j < vocab; j += blockDim.x) sum_exp += expf(logits[base + j] - max_logit);
    sum_exp = block_sum(sum_exp);

    for (size_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        const float p = expf(logits[base + j] - max_logit) / sum_exp;
        st(grad, base + j, p * inv_n - (j == target ? inv_n : 0.0f));
    }
}

// ---- sum of squares ----------------------------------------------------------------

constexpr unsigned kMaxPartials = 1024;

__global__ void sum_squares_partial_kernel(const float* x, size_t n, double* partials) {
    double local = 0.0;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += stride) {
        local += static_cast<double>(x[i]) * x[i];
    }
    local = block_sum(local);
    if (threadIdx.x == 0) partials[blockIdx.x] = local;
}

__global__ void sum_squares_accum_kernel(const float* x, size_t n, double* acc) {
    double local = 0.0;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += stride) {
        local += static_cast<double>(x[i]) * x[i];
    }
    local = block_sum(local);
    if (threadIdx.x == 0) atomicAdd(acc, local);
}

double* accumulator() {                    // one persistent device double, never freed
    static double* acc = nullptr;
    if (!acc) CUDA_CHECK(cudaMalloc(&acc, sizeof(double)));
    return acc;
}

__global__ void sum_partials_kernel(const double* partials, unsigned count, double* out) {
    double total = 0.0;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) total += partials[i];
    total = block_sum(total);
    if (threadIdx.x == 0) *out = total;
}

template <typename T>
T* scratch(size_t count, T*& ptr, size_t& capacity) {
    if (count > capacity) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        capacity = count;
    }
    return ptr;
}

} // namespace

// =====================================================================================

void rmsnorm(const float* x, const float* w, float* out, size_t rows, size_t D, float eps) {
    ROW_LAUNCH(rmsnorm_kernel<float>, rows, x, w, out, D, eps);
}

void rmsnorm_backward(const float* x, const float* w, const float* grad_out,
                      float* dx, float* w_grad, size_t rows, size_t D, float eps) {
    ROW_LAUNCH(rmsnorm_backward_kernel<float>, rows, x, w, grad_out, dx, w_grad, D, eps);
}

void softmax(const float* x, float* out, size_t rows, size_t cols, bool causal) {
    ROW_LAUNCH(softmax_kernel, rows, x, out, cols, causal);
}

void softmax_backward(const float* probs, const float* grad_out, float* dx, size_t rows, size_t cols) {
    ROW_LAUNCH(softmax_backward_kernel, rows, probs, grad_out, dx, cols);
}

float cross_entropy(const float* logits, const float* targets, size_t rows, size_t vocab) {
    static float* row_loss = nullptr;     static size_t row_loss_cap = 0;
    static float* result = nullptr;       static size_t result_cap = 0;

    float* losses = scratch(rows, row_loss, row_loss_cap);
    float* out = scratch<float>(1, result, result_cap);

    ROW_LAUNCH(cross_entropy_kernel, rows, logits, targets, losses, vocab);
    mean_kernel<<<1, kBlockSize>>>(losses, rows, out);
    CUDA_LAUNCH_CHECK();

    float host = 0.0f;
    copy_d2h(&host, out, sizeof(float));   // synchronizes
    return host;
}

void cross_entropy_backward(const float* logits, const float* targets, float* grad,
                            size_t rows, size_t vocab) {
    const float inv_n = 1.0f / static_cast<float>(rows);
    ROW_LAUNCH(cross_entropy_backward_kernel<float>, rows, logits, targets, grad, vocab, inv_n);
}

// ---- bf16 variants ----

void rmsnorm(const float* x, const float* w, uint16_t* out, size_t rows, size_t D, float eps) {
    ROW_LAUNCH(rmsnorm_kernel<__nv_bfloat16>, rows, x, w, as_bf16(out), D, eps);
}

void rmsnorm_backward(const float* x, const float* w, const uint16_t* grad_out,
                      float* dx, float* w_grad, size_t rows, size_t D, float eps) {
    ROW_LAUNCH(rmsnorm_backward_kernel<__nv_bfloat16>, rows, x, w, as_bf16(grad_out), dx, w_grad, D, eps);
}

void cross_entropy_backward(const float* logits, const float* targets, uint16_t* grad,
                            size_t rows, size_t vocab) {
    const float inv_n = 1.0f / static_cast<float>(rows);
    ROW_LAUNCH(cross_entropy_backward_kernel<__nv_bfloat16>, rows, logits, targets, as_bf16(grad), vocab, inv_n);
}

float sum_squares(const float* x, size_t n) {
    static double* partials = nullptr;    static size_t partials_cap = 0;
    static double* result = nullptr;      static size_t result_cap = 0;

    if (n == 0) return 0.0f;
    double* p = scratch<double>(kMaxPartials, partials, partials_cap);
    double* out = scratch<double>(1, result, result_cap);

    unsigned blocks = blocks_for(n);
    if (blocks > kMaxPartials) blocks = kMaxPartials;

    sum_squares_partial_kernel<<<blocks, kBlockSize>>>(x, n, p);
    CUDA_LAUNCH_CHECK();
    sum_partials_kernel<<<1, kBlockSize>>>(p, blocks, out);
    CUDA_LAUNCH_CHECK();

    double host = 0.0;
    copy_d2h(&host, out, sizeof(double));   // synchronizes
    return static_cast<float>(host);
}

void sum_squares_reset() { CUDA_CHECK(cudaMemsetAsync(accumulator(), 0, sizeof(double))); }

void sum_squares_add(const float* x, size_t n) {          // no sync
    if (n == 0) return;
    unsigned blocks = blocks_for(n);
    if (blocks > kMaxPartials) blocks = kMaxPartials;
    sum_squares_accum_kernel<<<blocks, kBlockSize>>>(x, n, accumulator());
    CUDA_LAUNCH_CHECK();
}

double sum_squares_read() {                               // the one synchronization
    double host = 0.0;
    copy_d2h(&host, accumulator(), sizeof(double));
    return host;
}

} // namespace cuda
