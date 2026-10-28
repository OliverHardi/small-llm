#pragma once

#include "model.h"
#include "tokenizer.h"

#include <string>
#include <cstdint>

struct SamplingOptions {
    float temperature = 0.7f;          // <= 0 means greedy
    float top_p = 0.8f;
    float repetition_penalty = 1.1f;   // 1.0 = off
    uint64_t seed = 1234;              // fixed seed -> comparable samples across checkpoints
    int64_t stop_token = -1;           // extra stop token (e.g. <|end|> after SFT); EOS always stops
};

class Inference {
public:
    Inference(Model& model, Tokenizer& tokenizer);

    std::string generate(const std::string& prompt, size_t max_new_tokens, SamplingOptions opt = {});
    std::string generateTokens(std::vector<uint32_t> tokens, size_t max_new_tokens, SamplingOptions opt = {});
    std::string generateChat(const std::string& user_text, size_t max_new_tokens, SamplingOptions opt = {});

private:
    Model& model_;
    Tokenizer& tokenizer_;
};