#include "nn/transformer_block.h"

#include <stdexcept>

#include "ops/elementwise.h"
#include "ops/rmsnorm.h"

TransformerBlock::TransformerBlock(const ModelConfig& config)
    : attention_(config),
      swiglu_(config),
      attention_norm_weight_(std::vector<size_t>{config.d_model}, 1.0f),
      ffn_norm_weight_(std::vector<size_t>{config.d_model}, 1.0f),
      attention_norm_grad_(std::vector<size_t>{config.d_model}, 0.0f),
      ffn_norm_grad_(std::vector<size_t>{config.d_model}, 0.0f),
      kRmsNormEps_(config.kRmsNormEps)
{
}

Tensor TransformerBlock::forward(const Tensor& x) {
    input_ = x.clone();

    // attention
    Tensor attn_out = attention_.forward(ops::rmsnorm(x, attention_norm_weight_, kRmsNormEps_));
    residual1_ = ops::add(x, attn_out);

    // ffn
    Tensor ffn_out = swiglu_.forward(ops::rmsnorm(residual1_, ffn_norm_weight_, kRmsNormEps_));
    return ops::add(residual1_, ffn_out);
}

Tensor TransformerBlock::backward(const Tensor& grad_output) {
    // out = h + ffn(norm(h))
    Tensor d_ffn_norm_out = swiglu_.backward(grad_output);
    Tensor d_h = ops::add(
        grad_output,
        ops::rmsnorm_backward(
            residual1_,
            ffn_norm_weight_,
            d_ffn_norm_out,
            ffn_norm_grad_,
            kRmsNormEps_
        )
    );

    // h = x + attn(norm(x))
    Tensor d_attn_norm_out = attention_.backward(d_h);
    return ops::add(
        d_h,
        ops::rmsnorm_backward(
            input_,
            attention_norm_weight_,
            d_attn_norm_out,
            attention_norm_grad_,
            kRmsNormEps_
        )
    );
}