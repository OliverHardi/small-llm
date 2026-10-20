#pragma once

#include "tensor.h"

namespace ops {

void set_cudnn_enabled(bool enabled);

struct AttentionOutput {
    Tensor out;     // [B, H, T, hd]
    Tensor stats;   // log-sum-exp per row [B, H, T]; for flash attention backwards pass later
};

struct AttentionGrads {
    Tensor dq, dk, dv;
};

// q, k, v: [B, H, T, hd], after RoPE.
AttentionOutput attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal);

AttentionGrads attention_backward(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& out, const Tensor& stats,
                                  const Tensor& grad_out, bool causal);

Tensor split_heads(const Tensor& x, size_t num_heads);   // [B, T, D] -> [B, H, T, hd]
Tensor merge_heads(const Tensor& x);                     // [B, H, T, hd] -> [B, T, D]

}