#include "model.h"

#include <stdexcept>
#include <cstdio>
 
#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/matmul.h"
#include "ops/rmsnorm.h"

#define FSTEP(msg) do { std::printf("    [fwd %s]\n", msg); std::fflush(stdout); } while (0)

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
    FSTEP("embedding");
    Tensor h = ops::embedding(token_ids_, embedding_).reshaped({batch_size, seq_len, d_model});
 
    // transformer
    size_t block_index = 0;
    for (TransformerBlock& block : blocks_) {
        std::printf("    [fwd block %zu]\n", block_index++); std::fflush(stdout);
        h = block.forward(h);
    }
 
    // final norm + tied LM head
    FSTEP("final norm");
    final_norm_input_ = h;
    head_input_ = ops::rmsnorm(h, final_norm_weight_, kRmsNormEps_).reshaped({batch_size * seq_len, d_model});
    FSTEP("lm head");
    Tensor logits = ops::matmul_bt(head_input_, embedding_);   // [B*T, V]
    return logits.reshaped({batch_size, seq_len, vocab});
}


void Model::backward(const Tensor& d_logits) {
    const size_t d_model = config_.d_model;
    const size_t vocab = config_.vocab_size;
    const size_t rows = batch_size_ * seq_len_;
 
    if (d_logits.numel() != rows * vocab) {
        throw std::invalid_argument("Model::backward: d_logits must be [B, T, vocab_size]");
    }
 
    Tensor d_flat = d_logits.reshaped({rows, vocab});   // [N, V]
 
    // logits = H @ E^T:  dE += dLogits^T @ H,   dH = dLogits @ E
    embedding_grad_ = ops::add(embedding_grad_, ops::matmul_at(d_flat, head_input_));
    Tensor d_head_in = ops::matmul(d_flat, embedding_).reshaped({batch_size_, seq_len_, d_model});
 
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
 
    params.push_back({"embedding", &embedding_, &embedding_grad_, true});
 
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const std::string prefix = "blocks." + std::to_string(i) + ".";
        TransformerBlock& b = blocks_[i];
        Attention& attn = b.attention();
        SwiGLU& ffn = b.swiglu();
 
        params.push_back({prefix + "attn_norm", &b.attention_norm_weight(), &b.attention_norm_grad(), false});
        params.push_back({prefix + "attn.q",    &attn.q_weights(),   &attn.q_grad(),   true});
        params.push_back({prefix + "attn.k",    &attn.k_weights(),   &attn.k_grad(),   true});
        params.push_back({prefix + "attn.v",    &attn.v_weights(),   &attn.v_grad(),   true});
        params.push_back({prefix + "attn.out",  &attn.out_weights(), &attn.out_grad(), true});
        params.push_back({prefix + "ffn_norm",  &b.ffn_norm_weight(), &b.ffn_norm_grad(), false});
        params.push_back({prefix + "ffn.gate",  &ffn.gate_weights(), &ffn.gate_grad(), true});
        params.push_back({prefix + "ffn.up",    &ffn.up_weights(),   &ffn.up_grad(),   true});
        params.push_back({prefix + "ffn.down",  &ffn.down_weights(), &ffn.down_grad(), true});
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