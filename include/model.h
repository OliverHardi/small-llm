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
};


// token ids -> embedding -> N x TransformerBlock -> final RMSNorm -> tied LM head -> logits
// (tied)
class Model {
public:
    explicit Model(const ModelConfig& config);
 
    // token_ids [B, T] -> logits [B, T, vocab_size]
    Tensor forward(const std::vector<uint32_t>& token_ids, size_t batch_size, size_t seq_len);

    // backward pass
    void backward(const Tensor& d_logits);
 
    void zero_grad();
 
    // every trainable tensor (in order)
    std::vector<Param> parameters();

    // the config of the model
    const ModelConfig& config() const { return config_; }

    void to(Device device);
private:
    ModelConfig config_;
 
    Tensor embedding_;        // [vocab_size, d_model], also the language modeling head
    Tensor embedding_grad_;

    std::vector<TransformerBlock> blocks_;

    Tensor final_norm_weight_;   // [d_model]
    Tensor final_norm_grad_;

    float kRmsNormEps_ = 1e-5f;

    // forward cache for backward pass
    std::vector<uint32_t> token_ids_;
    size_t batch_size_ = 0;
    size_t seq_len_ = 0;
    Tensor final_norm_input_;    // [B, T, D], output of the last block
    Tensor head_input_;          // [B*T, D], output of the final norm
};