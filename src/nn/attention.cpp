#include "nn/attention.h"

#include "ops/elementwise.h"
#include "ops/matmul.h"
#include "ops/rope.h"
#include "ops/attention.h"

#include <stdexcept>

namespace {
 
// x_flat [B*T, D] @ w [D, D] -> [B, T, D]
Tensor project(const Tensor& x_flat, const Tensor& w, size_t B, size_t T) {
    return ops::matmul(x_flat, w).reshaped({B, T, w.dim(1)});
}
 
} // namespace

Attention::Attention(const ModelConfig& config)
      : q_weights_({config.d_model, config.d_model}),
        k_weights_({config.d_model, config.d_model}),
        v_weights_({config.d_model, config.d_model}),
        out_weights_({config.d_model, config.d_model}),
 
        q_grad_({config.d_model, config.d_model}, 0.0f),
        k_grad_({config.d_model, config.d_model}, 0.0f),
        v_grad_({config.d_model, config.d_model}, 0.0f),
        out_grad_({config.d_model, config.d_model}, 0.0f),
 
        d_model_(config.d_model),
        num_heads_(config.num_heads),
        head_dim_(config.head_dim()),
        rope_theta_(config.rope_theta)
{
    if (d_model_ != num_heads_ * head_dim_) {
        throw std::invalid_argument("Attention: d_model must equal num_heads * head_dim");
    }
 
    const float std_in  = 0.02f;
    const float std_out = 0.02f / std::sqrt(2.0f * static_cast<float>(config.num_layers));
 
    q_weights_.randomNormal(0.0f, std_in);
    k_weights_.randomNormal(0.0f, std_in);
    v_weights_.randomNormal(0.0f, std_in);
    out_weights_.randomNormal(0.0f, std_out);
}

void Attention::enable_bf16(bool on) {
    bf16_ = on;
    if (on) {
        if (!q_weights_.is_cuda()) throw std::logic_error("Attention::enable_bf16: move the model to CUDA first");
        q_w16_   = Tensor(q_weights_.shape(),   Device::CUDA, DType::BF16);
        k_w16_   = Tensor(k_weights_.shape(),   Device::CUDA, DType::BF16);
        v_w16_   = Tensor(v_weights_.shape(),   Device::CUDA, DType::BF16);
        out_w16_ = Tensor(out_weights_.shape(), Device::CUDA, DType::BF16);
    } else {
        q_w16_ = k_w16_ = v_w16_ = out_w16_ = Tensor();
    }
}

Tensor Attention::forward(const Tensor& x) {
    if (bf16_) return forward_bf16(x);
    if (x.ndim() != 3 || x.dim(2) != d_model_) {
        throw std::invalid_argument("Attention::forward expects [B, T, d_model]");
    }
 
    const size_t B = x.dim(0);
    const size_t T = x.dim(1);
 
    input_flat_ = x.reshaped({B * T, d_model_});
 
    Tensor q = project(input_flat_, q_weights_, B, T);
    Tensor k = project(input_flat_, k_weights_, B, T);
    Tensor v = project(input_flat_, v_weights_, B, T);
 
    // split into heads; rotate (RoPE) on q and k heads only
    q_heads_ = ops::rope(ops::split_heads(q, num_heads_), rope_theta_);
    k_heads_ = ops::rope(ops::split_heads(k, num_heads_), rope_theta_);
    v_heads_ = ops::split_heads(v, num_heads_);
 
    // fused causal attention: softmax(Q K^T / sqrt(hd)) V
    ops::AttentionOutput result = ops::attention(q_heads_, k_heads_, v_heads_, true);
    attn_out_ = result.out;
    attn_stats_ = result.stats;
 
    // merge heads + output projection
    context_flat_ = ops::merge_heads(attn_out_).reshaped({B * T, d_model_});
 
    return ops::matmul(context_flat_, out_weights_).reshaped({B, T, d_model_});
}

Tensor Attention::backward(const Tensor& grad_output) {
    if (bf16_) return backward_bf16(grad_output);
    if (grad_output.ndim() != 3 || grad_output.dim(2) != d_model_) {
        throw std::invalid_argument("Attention::backward expects [B, T, d_model]");
    }
 
    const size_t B = grad_output.dim(0);
    const size_t T = grad_output.dim(1);
 
    Tensor grad_flat = grad_output.reshaped({B * T, d_model_});
 
    // output projection: out = context @ Wo
    out_grad_ = ops::add(out_grad_, ops::matmul_at(context_flat_, grad_flat));
    Tensor d_context = ops::split_heads(
        ops::matmul_bt(grad_flat, out_weights_).reshaped({B, T, d_model_}), num_heads_
    );
 
    // fused attention backward
    ops::AttentionGrads g = ops::attention_backward(
        q_heads_, k_heads_, v_heads_, attn_out_, attn_stats_, d_context, true
    );
 
    // Undo RoPE on q and k
    Tensor d_q = ops::rope_backward(g.dq, rope_theta_);
    Tensor d_k = ops::rope_backward(g.dk, rope_theta_);
 
    // merge heads back to [B*T, D]
    Tensor dq_flat = ops::merge_heads(d_q).reshaped({B * T, d_model_});
    Tensor dk_flat = ops::merge_heads(d_k).reshaped({B * T, d_model_});
    Tensor dv_flat = ops::merge_heads(g.dv).reshaped({B * T, d_model_});
 
    // weight gradients: dW = X^T @ dY
    q_grad_ = ops::add(q_grad_, ops::matmul_at(input_flat_, dq_flat));
    k_grad_ = ops::add(k_grad_, ops::matmul_at(input_flat_, dk_flat));
    v_grad_ = ops::add(v_grad_, ops::matmul_at(input_flat_, dv_flat));
 
    // input gradient: dX = dY @ W^T, summed over each qkv
    Tensor dx = ops::matmul_bt(dq_flat, q_weights_);
    dx = ops::add(dx, ops::matmul_bt(dk_flat, k_weights_));
    dx = ops::add(dx, ops::matmul_bt(dv_flat, v_weights_));
 
    return dx.reshaped({B, T, d_model_});
}

// ---- bf16 path ---------------------------------------------------------------------
// bf16 forward version
Tensor Attention::forward_bf16(const Tensor& x) {
    if (x.ndim() != 3 || x.dim(2) != d_model_ || !x.is_bf16()) {
        throw std::invalid_argument("Attention::forward (bf16) expects bf16 [B, T, d_model]");
    }

    const size_t B = x.dim(0);
    const size_t T = x.dim(1);

    input_flat_ = x.reshaped({B * T, d_model_});

    Tensor q = project(input_flat_, q_w16_, B, T);
    Tensor k = project(input_flat_, k_w16_, B, T);
    Tensor v = project(input_flat_, v_w16_, B, T);

    q_heads_ = ops::rope(ops::split_heads(q, num_heads_), rope_theta_);
    k_heads_ = ops::rope(ops::split_heads(k, num_heads_), rope_theta_);
    v_heads_ = ops::split_heads(v, num_heads_);

    ops::AttentionOutput result = ops::attention(q_heads_, k_heads_, v_heads_, true);   // cuDNN
    attn_out_ = result.out;
    attn_stats_ = result.stats;

    context_flat_ = ops::merge_heads(attn_out_).reshaped({B * T, d_model_});
    return ops::matmul(context_flat_, out_w16_).reshaped({B, T, d_model_});
}

// bf16 backward version
Tensor Attention::backward_bf16(const Tensor& grad_output) {
    if (grad_output.ndim() != 3 || grad_output.dim(2) != d_model_) {
        throw std::invalid_argument("Attention::backward expects [B, T, d_model]");
    }

    const size_t B = grad_output.dim(0);
    const size_t T = grad_output.dim(1);

    Tensor grad_flat = grad_output.to_dtype(DType::BF16).reshaped({B * T, d_model_});

    ops::matmul_at_accumulate(context_flat_, grad_flat, out_grad_);
    Tensor d_context = ops::split_heads(
        ops::matmul_bt(grad_flat, out_w16_).reshaped({B, T, d_model_}), num_heads_
    );

    ops::AttentionGrads g = ops::attention_backward(
        q_heads_, k_heads_, v_heads_, attn_out_, attn_stats_, d_context, true
    );

    Tensor d_q = ops::rope_backward(g.dq, rope_theta_);
    Tensor d_k = ops::rope_backward(g.dk, rope_theta_);

    Tensor dq_flat = ops::merge_heads(d_q).reshaped({B * T, d_model_});
    Tensor dk_flat = ops::merge_heads(d_k).reshaped({B * T, d_model_});
    Tensor dv_flat = ops::merge_heads(g.dv).reshaped({B * T, d_model_});

    ops::matmul_at_accumulate(input_flat_, dq_flat, q_grad_);
    ops::matmul_at_accumulate(input_flat_, dk_flat, k_grad_);
    ops::matmul_at_accumulate(input_flat_, dv_flat, v_grad_);

    Tensor dx = ops::matmul_bt(dq_flat, q_w16_, DType::F32);
    ops::matmul_bt_accumulate(dk_flat, k_w16_, dx);
    ops::matmul_bt_accumulate(dv_flat, v_w16_, dx);

    return dx.reshaped({B, T, d_model_});
}