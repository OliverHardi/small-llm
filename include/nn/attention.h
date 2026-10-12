#pragma once

#include "config.h"
#include "tensor.h"


// Shapes:  B = batch, T = sequence length, D = d_model, H = num_heads, hd = head_dim
class Attention {
public:

    explicit Attention(const ModelConfig& config);

    // x: [B, T, D] -> [B, T, D]
    Tensor forward(const Tensor& x);

    Tensor backward(const Tensor& grad_output);

    Tensor& q_weights()   { return q_weights_; }
    Tensor& k_weights()   { return k_weights_; }
    Tensor& v_weights()   { return v_weights_; }
    Tensor& out_weights() { return out_weights_; }

    Tensor& q_grad()   { return q_grad_; }
    Tensor& k_grad()   { return k_grad_; }
    Tensor& v_grad()   { return v_grad_; }
    Tensor& out_grad() { return out_grad_; }

private:
    // weights [D, D]
    Tensor q_weights_;
    Tensor k_weights_;
    Tensor v_weights_;
    Tensor out_weights_;

    // gradients
    Tensor q_grad_;
    Tensor k_grad_;
    Tensor v_grad_;
    Tensor out_grad_;

    // forward cache for backward pass
    Tensor input_flat_;     // [B*T, D]
    Tensor q_heads_;        // [B, H, T, hd]    after rope
    Tensor k_heads_;        // [B, H, T, hd]    after rope
    Tensor v_heads_;        // [B, H, T, hd]
    Tensor attn_out_;       // [B, H, T, hd]    attention output before the head merge
    Tensor attn_stats_;     // log-sum-exp      empty on the CPU path
    Tensor context_flat_;   // [B*T, D]         heads merged

    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    float  rope_theta_;
};