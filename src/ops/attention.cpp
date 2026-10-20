#include "ops/attention.h"
 
#include <cmath>
 
#include "ops/elementwise.h"
#include "ops/matmul.h"
#include "ops/softmax.h"

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

#ifdef USE_CUDNN
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include "cuda/cudnn_attention.h"
#endif
 
namespace {

// [B, H, A, C] -> [B, H, C, A]
Tensor transpose_last_two(const Tensor& x) {
    const size_t B = x.dim(0), H = x.dim(1), A = x.dim(2), C = x.dim(3);

    #ifdef USE_CUDA
        if (x.is_cuda()) {
            Tensor out({B, H, C, A}, Device::CUDA);
            cuda::transpose_batched(x.data(), out.data(), B * H, A, C);
            return out;
        }
    #endif
 
    Tensor out({B, H, C, A});
    for (size_t b = 0; b < B; ++b)
        for (size_t h = 0; h < H; ++h)
            for (size_t i = 0; i < A; ++i)
                for (size_t j = 0; j < C; ++j)
                    out(b, h, j, i) = x(b, h, i, j);
    return out;
}

Tensor attention_probs(const Tensor& q, const Tensor& k, bool causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.dim(3)));
    Tensor scores = ops::batched_matmul(q, transpose_last_two(k));
    scores = ops::multiply(scores, scale);
    return ops::softmax(scores, -1, causal);
}

#ifdef USE_CUDNN
bool g_cudnn_enabled = true;

// bf16 device scratch, returned to the allocator pool when it goes out of scope.
struct Bf16Buffer {
    void* ptr;
    explicit Bf16Buffer(size_t n) : ptr(cuda::device_alloc(n * sizeof(uint16_t))) {}
    ~Bf16Buffer() { cuda::device_free(ptr); }
    Bf16Buffer(const Bf16Buffer&) = delete;
    Bf16Buffer& operator=(const Bf16Buffer&) = delete;
    uint16_t* bf16() const { return static_cast<uint16_t*>(ptr); }
};

// One cuDNN graph set per (B, H, T, D); building is slow, so it happens once.
// Leaked on purpose: destroying cuDNN objects during static destruction at exit is not safe.
CudnnAttention& cudnn_for(size_t B, size_t H, size_t T, size_t D) {
    static auto* cache = new std::map<std::tuple<size_t, size_t, size_t, size_t>, std::unique_ptr<CudnnAttention>>;
    auto& slot = (*cache)[{B, H, T, D}];
    if (!slot) slot = std::make_unique<CudnnAttention>(B, H, T, D);
    return *slot;
}

bool use_cudnn(const Tensor& q, bool causal) {
    return g_cudnn_enabled && causal && q.is_cuda() && q.dim(3) % 8 == 0 && q.dim(3) <= 128;
}
#endif

} // namespace

void ops::set_cudnn_enabled(bool enabled) {
#ifdef USE_CUDNN
    g_cudnn_enabled = enabled;
#else
    (void)enabled;
#endif
}

// [B, T, D] -> [B, H, T, hd]
Tensor ops::split_heads(const Tensor& x, size_t num_heads) {
    const size_t B = x.dim(0), T = x.dim(1), D = x.dim(2);
    const size_t hd = D / num_heads;

    #ifdef USE_CUDA
        if (x.is_cuda()) {
            Tensor out({B, num_heads, T, hd}, Device::CUDA);
            cuda::split_heads(x.data(), out.data(), B, T, num_heads, hd);
            return out;
        }
    #endif
 
    Tensor out({B, num_heads, T, hd});
    for (size_t b = 0; b < B; ++b)
        for (size_t h = 0; h < num_heads; ++h)
            for (size_t t = 0; t < T; ++t)
                for (size_t d = 0; d < hd; ++d)
                    out(b, h, t, d) = x(b, t, h * hd + d);
    return out;
}

// [B, H, T, hd] -> [B, T, D]
Tensor ops::merge_heads(const Tensor& x) {
    const size_t B = x.dim(0), H = x.dim(1), T = x.dim(2), hd = x.dim(3);

    #ifdef USE_CUDA
        if (x.is_cuda()) {
            Tensor out({B, T, H * hd}, Device::CUDA);
            cuda::merge_heads(x.data(), out.data(), B, T, H, hd);
            return out;
        }
    #endif
 
    Tensor out({B, T, H * hd});
    for (size_t b = 0; b < B; ++b)
        for (size_t h = 0; h < H; ++h)
            for (size_t t = 0; t < T; ++t)
                for (size_t d = 0; d < hd; ++d)
                    out(b, t, h * hd + d) = x(b, h, t, d);
    return out;
}

// CPU forward. `stats` (the log-sum-exp a flash backward needs) is left empty here:
// the CPU backward recomputes P from q and k instead.
ops::AttentionOutput ops::attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {

    #ifdef USE_CUDNN
    if (use_cudnn(q, causal)) {
        const size_t B = q.dim(0), H = q.dim(1), T = q.dim(2), D = q.dim(3), n = q.numel();
        Bf16Buffer qb(n), kb(n), vb(n), ob(n);
        cuda::cast_f32_to_bf16(q.data(), qb.bf16(), n);
        cuda::cast_f32_to_bf16(k.data(), kb.bf16(), n);
        cuda::cast_f32_to_bf16(v.data(), vb.bf16(), n);

        AttentionOutput result;
        result.stats = Tensor({B, H, T, 1}, Device::CUDA);          // fp32 softmax stats for the backward
        cudnn_for(B, H, T, D).forward(qb.ptr, kb.ptr, vb.ptr, ob.ptr, result.stats.data(), 0);

        result.out = Tensor({B, H, T, D}, Device::CUDA);
        cuda::cast_bf16_to_f32(ob.bf16(), result.out.data(), n);
        return result;
    }
    #endif

    Tensor probs = attention_probs(q, k, causal);
 
    AttentionOutput result;
    result.out = ops::batched_matmul(probs, v);
    return result;
}

// CPU backward. P is recomputed from q and k, so only the inputs need to be kept
// (this is the "recompute" strategy; a flash kernel does the same with the saved stats).
// `out` and `stats` are unused here but part of the interface so a GPU version can use them.
ops::AttentionGrads ops::attention_backward(const Tensor& q, const Tensor& k, const Tensor& v,
                                            const Tensor& out, const Tensor& stats,
                                            const Tensor& grad_out, bool causal) {

    #ifdef USE_CUDNN
        if (use_cudnn(q, causal) && stats.numel() > 0) {     // empty stats = forward ran unfused
            const size_t B = q.dim(0), H = q.dim(1), T = q.dim(2), D = q.dim(3), n = q.numel();
            Bf16Buffer qb(n), kb(n), vb(n), ob(n), gob(n), dqb(n), dkb(n), dvb(n);
            cuda::cast_f32_to_bf16(q.data(), qb.bf16(), n);
            cuda::cast_f32_to_bf16(k.data(), kb.bf16(), n);
            cuda::cast_f32_to_bf16(v.data(), vb.bf16(), n);
            cuda::cast_f32_to_bf16(out.data(), ob.bf16(), n);    // exact: out came from bf16 in the first place
            cuda::cast_f32_to_bf16(grad_out.data(), gob.bf16(), n);

            cudnn_for(B, H, T, D).backward(qb.ptr, kb.ptr, vb.ptr, ob.ptr, gob.ptr, stats.data(),
                                        dqb.ptr, dkb.ptr, dvb.ptr, 0);

            AttentionGrads grads;
            grads.dq = Tensor(q.shape(), Device::CUDA);
            grads.dk = Tensor(q.shape(), Device::CUDA);
            grads.dv = Tensor(q.shape(), Device::CUDA);
            cuda::cast_bf16_to_f32(dqb.bf16(), grads.dq.data(), n);
            cuda::cast_bf16_to_f32(dkb.bf16(), grads.dk.data(), n);
            cuda::cast_bf16_to_f32(dvb.bf16(), grads.dv.data(), n);
            return grads;
        }
    #endif
    
    (void)out;
    (void)stats;

    const float scale = 1.0f / std::sqrt(static_cast<float>(q.dim(3)));

    Tensor probs = attention_probs(q, k, causal);

    // out = P V
    Tensor d_probs = ops::batched_matmul(grad_out, transpose_last_two(v));
    AttentionGrads grads;
    grads.dv = ops::batched_matmul(transpose_last_two(probs), grad_out);

    // P = softmax(S), S = scale * Q K^T
    Tensor d_scores = ops::softmax_backward(probs, d_probs);
    ops::scale_inplace(d_scores, scale);

    grads.dq = ops::batched_matmul(d_scores, k);
    grads.dk = ops::batched_matmul(transpose_last_two(d_scores), q);
    return grads;
}