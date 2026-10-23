#pragma once

#include "config.h"
#include "tensor.h"
#include "nn/attention.h"
#include "nn/swiglu.h"

// The residual stream is always fp32
class TransformerBlock {
public:
    explicit TransformerBlock(const ModelConfig& config);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    void enable_bf16(bool on) {
        bf16_ = on;
        attention_.enable_bf16(on);
        swiglu_.enable_bf16(on);
    }

    inline Attention& attention() { return attention_; }
    inline SwiGLU& swiglu() { return swiglu_; }

    inline Tensor& attention_norm_weight() { return attention_norm_weight_; }
    inline Tensor& ffn_norm_weight() { return ffn_norm_weight_; }

    inline Tensor& attention_norm_grad() { return attention_norm_grad_; }
    inline Tensor& ffn_norm_grad() { return ffn_norm_grad_; }

private:
    Attention attention_;
    SwiGLU swiglu_;

    // RMSNorm weights (fp32 in both modes)
    Tensor attention_norm_weight_;
    Tensor ffn_norm_weight_;

    // RMSNorm gradients
    Tensor attention_norm_grad_;
    Tensor ffn_norm_grad_;

    float kRmsNormEps_ = 1e-5f;
    bool bf16_ = false;

    // forward cache for backward pass
    Tensor input_;
    Tensor attention_norm_output_;
    Tensor attention_output_;
    Tensor residual1_;
    Tensor ffn_norm_output_;
    Tensor ffn_output_;
};