#pragma once

#include <cstdint>
#include <string>
#include <vector>
 
#include "config.h"
#include "tensor.h"
#include "nn/transformer_block.h"

// trainable tensor + gradient
struct Param {
    std::string name;
    Tensor* value;
    Tensor* grad;
    bool decay;   // apply weight decay? true for matrices, false for norm weights
    Tensor* shadow = nullptr;   // bf16 copy of `value`
};


// token ids -> embedding -> N x TransformerBlock -> final RMSNorm -> tied LM head -> logits
// (tied)
class Model {
public:
    explicit Model(const ModelConfig& config);
 
    // token_ids [B, T] -> logits [B, T, vocab_size]   (logits are fp32 in both modes)
    Tensor forward(const std::vector<uint32_t>& token_ids, size_t batch_size, size_t seq_len);

    // backward pass. d_logits may be fp32 or bf16.
    void backward(const Tensor& d_logits);
 
    void zero_grad();
 
    // every trainable tensor (in order)
    std::vector<Param> parameters();

    // the config of the model
    const ModelConfig& config() const { return config_; }

    void to(Device device);

    // ---- bf16 mode ----
    // Call after to(CUDA)
    // call refresh_bf16_weights() after every optimizer step and checkpoint load
    void enable_bf16(bool on);
    bool bf16() const { return bf16_; }
    void refresh_bf16_weights();

    DType logits_grad_dtype() const { return bf16_ ? DType::BF16 : DType::F32; }

private:
    ModelConfig config_;
 
    Tensor embedding_;        // [vocab_size, d_model], also the language modeling head
    Tensor embedding_grad_;
    Tensor embedding16_;      // bf16 shadow

    std::vector<TransformerBlock> blocks_;

    Tensor final_norm_weight_;   // [d_model]
    Tensor final_norm_grad_;

    float kRmsNormEps_ = 1e-5f;
    bool bf16_ = false;

    // forward cache for backward pass
    std::vector<uint32_t> token_ids_;
    size_t batch_size_ = 0;
    size_t seq_len_ = 0;
    Tensor final_norm_input_;    // [B, T, D], output of the last block (fp32)
    Tensor head_input_;          // [B*T, D], output of the final norm (bf16 in bf16 mode)
};