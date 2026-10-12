#pragma once

#include <cstddef>
#include <cstdint>

struct ModelConfig {
/*
    base: 256
    added: 32506
    end of sequence: 1
    reserved: 5
    
    total: 32768 (2^15)
*/
    size_t vocab_size = 32768;
    size_t context_length = 2048;

    size_t num_layers = 16;
    size_t d_model = 640;

    size_t num_heads = 10;
    size_t head_dim() const {
        return d_model / num_heads;
    }

    size_t ffn_hidden = 1728;

    float rope_theta = 10000.0f;

    size_t totalParams() const {
        size_t embedding_params = vocab_size * d_model; // tied
        // per layer:
        size_t attention_params = 4 * d_model * d_model;
        size_t ffn_params = 3 * d_model * ffn_hidden;
        size_t norm_params = 2 * d_model;

        size_t layer_params = attention_params + ffn_params + norm_params;

        size_t final_norm_params = d_model;

        return embedding_params + num_layers * layer_params + final_norm_params;
    }

    uint32_t seed = 1234;

    float kRmsNormEps = 1e-5f;
};

struct TrainingConfig {
    size_t batch_size = 2;

    float learning_rate = 3e-4f;
    float weight_decay = 0.1f;

};