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

    // call after the weights are on CUDA with bf16 mode
    void enable_bf16(bool on);
    bool bf16() const { return bf16_; }

    Tensor& q_weights()   { return q_weights_; }
    Tensor& k_weights()   { return k_weights_; }
    Tensor& v_weights()   { return v_weights_; }
    Tensor& out_weights() { return out_weights_; }

    Tensor& q_grad()   { return q_grad_; }
    Tensor& k_grad()   { return k_grad_; }
    Tensor& v_grad()   { return v_grad_; }
    Tensor& out_grad() { return out_grad_; }

    Tensor& q_weights16()   { return q_w16_; }
    Tensor& k_weights16()   { return k_w16_; }
    Tensor& v_weights16()   { return v_w16_; }
    Tensor& out_weights16() { return out_w16_; }

private:
    Tensor forward_bf16(const Tensor& x);
    Tensor backward_bf16(const Tensor& grad_output);

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

    // bf16 shadows of the weights
    Tensor q_w16_;
    Tensor k_w16_;
    Tensor v_w16_;
    Tensor out_w16_;

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
    bool   bf16_ = false;
};