#include "ops/attention.h"
 
#include <cmath>
 
#include "ops/elementwise.h"
#include "ops/matmul.h"
#include "ops/softmax.h"

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif
 
namespace {

// [B, H, A, C] -> [B, H, C, A]
Tensor transpose_last_two(const Tensor& x) {
    const size_t B = x.dim(0), H = x.dim(1), A = x.dim(2), C = x.dim(3);
 
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

} // namespace


// [B, T, D] -> [B, H, T, hd]
Tensor ops::split_heads(const Tensor& x, size_t num_heads) {
    const size_t B = x.dim(0), T = x.dim(1), D = x.dim(2);
    const size_t hd = D / num_heads;
 
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