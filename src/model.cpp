#include "model.h"

#include <stdexcept>
#include <cstdio>
 
#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/matmul.h"
#include "ops/rmsnorm.h"

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

Model::Model(const ModelConfig& config)
    : config_(config),
      embedding_({config.vocab_size, config.d_model}),
      embedding_grad_({config.vocab_size, config.d_model}, 0.0f),
      final_norm_weight_(std::vector<size_t>{config.d_model}, 1.0f),
      final_norm_grad_(std::vector<size_t>{config.d_model}, 0.0f),
      kRmsNormEps_(config.kRmsNormEps)
{
    embedding_.randomNormal(0.0f, 0.02f);
 
    blocks_.reserve(config.num_layers);   // keeps pointers from parameters() stable
    for (size_t i = 0; i < config.num_layers; ++i) {
        blocks_.emplace_back(config);
    }
}

Tensor Model::forward(const std::vector<uint32_t>& token_ids, size_t batch_size, size_t seq_len) {
    if (token_ids.size() != batch_size * seq_len) {
        throw std::invalid_argument("Model::forward: token_ids.size() must equal batch_size * seq_len");
    }
    if (seq_len > config_.context_length) {
        throw std::invalid_argument("Model::forward: seq_len exceeds context_length");
    }
 
    const size_t d_model = config_.d_model;
    const size_t vocab = config_.vocab_size;
 
    token_ids_ = token_ids;
    batch_size_ = batch_size;
    seq_len_ = seq_len;
 
    // Embedding lookup
    Tensor h = ops::embedding(token_ids_, embedding_).reshaped({batch_size, seq_len, d_model});
 
    // transformer
    for (TransformerBlock& block : blocks_) {
        h = block.forward(h);
    }
 
    // final norm + tied LM head
    final_norm_input_ = h;
    head_input_ = ops::rmsnorm(h, final_norm_weight_, kRmsNormEps_,
                               bf16_ ? DType::BF16 : DType::F32).reshaped({batch_size * seq_len, d_model});
    // bf16: bf16 operands, fp32 logits (cross-entropy math wants full precision)
    Tensor logits = bf16_ ? ops::matmul_bt(head_input_, embedding16_, DType::F32)
                          : ops::matmul_bt(head_input_, embedding_);   // [B*T, V]
    return logits.reshaped({batch_size, seq_len, vocab});
}


void Model::backward(const Tensor& d_logits) {
    const size_t d_model = config_.d_model;
    const size_t vocab = config_.vocab_size;
    const size_t rows = batch_size_ * seq_len_;
 
    if (d_logits.numel() != rows * vocab) {
        throw std::invalid_argument("Model::backward: d_logits must be [B, T, vocab_size]");
    }
 
    // logits = H @ E^T:  dE += dLogits^T @ H,   dH = dLogits @ E
    Tensor d_head_in;
    if (bf16_) {
        Tensor d_flat = d_logits.to_dtype(DType::BF16).reshaped({rows, vocab});   // [N, V], no-op if already bf16
        ops::matmul_at_accumulate(d_flat, head_input_, embedding_grad_);          // straight into the fp32 grad
        d_head_in = ops::matmul(d_flat, embedding16_).reshaped({batch_size_, seq_len_, d_model});   // bf16
    } else {
        Tensor d_flat = d_logits.reshaped({rows, vocab});   // [N, V]
        embedding_grad_ = ops::add(embedding_grad_, ops::matmul_at(d_flat, head_input_));
        d_head_in = ops::matmul(d_flat, embedding_).reshaped({batch_size_, seq_len_, d_model});
    }
 
    // final norm
    Tensor d_h = ops::rmsnorm_backward(
        final_norm_input_, final_norm_weight_, d_head_in, final_norm_grad_, kRmsNormEps_);
 
    // reverse transformer blocks
    for (size_t i = blocks_.size(); i-- > 0;) {
        d_h = blocks_[i].backward(d_h);
    }
 
    // embedding lookup
    ops::embedding_backward(token_ids_, d_h.reshaped({rows, d_model}), embedding_grad_);
}


void Model::zero_grad() {
    for (Param& p : parameters()) {
        p.grad->zero();
    }
}
 
std::vector<Param> Model::parameters() {
    std::vector<Param> params;
    auto shadow = [this](Tensor& t) -> Tensor* { return bf16_ ? &t : nullptr; };   // bf16 copy, only in bf16 mode
 
    params.push_back({"embedding", &embedding_, &embedding_grad_, true, bf16_ ? &embedding16_ : nullptr});
 
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const std::string prefix = "blocks." + std::to_string(i) + ".";
        TransformerBlock& b = blocks_[i];
        Attention& attn = b.attention();
        SwiGLU& ffn = b.swiglu();
 
        params.push_back({prefix + "attn_norm", &b.attention_norm_weight(), &b.attention_norm_grad(), false});
        params.push_back({prefix + "attn.q",    &attn.q_weights(),   &attn.q_grad(),   true, shadow(attn.q_weights16())});
        params.push_back({prefix + "attn.k",    &attn.k_weights(),   &attn.k_grad(),   true, shadow(attn.k_weights16())});
        params.push_back({prefix + "attn.v",    &attn.v_weights(),   &attn.v_grad(),   true, shadow(attn.v_weights16())});
        params.push_back({prefix + "attn.out",  &attn.out_weights(), &attn.out_grad(), true, shadow(attn.out_weights16())});
        params.push_back({prefix + "ffn_norm",  &b.ffn_norm_weight(), &b.ffn_norm_grad(), false});
        params.push_back({prefix + "ffn.gate",  &ffn.gate_weights(), &ffn.gate_grad(), true, shadow(ffn.gate_weights16())});
        params.push_back({prefix + "ffn.up",    &ffn.up_weights(),   &ffn.up_grad(),   true, shadow(ffn.up_weights16())});
        params.push_back({prefix + "ffn.down",  &ffn.down_weights(), &ffn.down_grad(), true, shadow(ffn.down_weights16())});
    }
 
    params.push_back({"final_norm", &final_norm_weight_, &final_norm_grad_, false});
    return params;
}

void Model::to(Device device) {
    for (const Param& p : parameters()) {
        *p.value = p.value->to(device);
        *p.grad  = p.grad->to(device);
    }
}
// ---- bf16 mode ---------------------------------------------------------------------
// Matmul weights get a bf16 shadow of the fp32 master (norm weights stay fp32 and have none).
// Call after model.to(CUDA). After every optimizer step and every checkpoint load, call
// refresh_bf16_weights() so the shadows follow the masters.
void Model::enable_bf16(bool on) {
    if (on && !embedding_.is_cuda()) throw std::logic_error("Model::enable_bf16: move the model to CUDA first");
    bf16_ = on;
    for (TransformerBlock& b : blocks_) b.enable_bf16(on);
    embedding16_ = on ? Tensor(embedding_.shape(), Device::CUDA, DType::BF16) : Tensor();
    if (on) refresh_bf16_weights();
}

void Model::refresh_bf16_weights() {
#ifdef USE_CUDA
    if (!bf16_) return;
    for (const Param& p : parameters()) {
        if (p.shadow) cuda::cast_f32_to_bf16(p.value->data(), p.shadow->bf16_data(), p.value->numel());
    }
#endif
}
