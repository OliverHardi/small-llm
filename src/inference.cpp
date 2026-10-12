#include "inference.h"

#include "ops/softmax.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

Inference::Inference(Model& model, Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

// greedy decoding
std::string Inference::generate(const std::string& prompt, size_t max_new_tokens) {
    std::vector<uint32_t> tokens = tokenizer_.encode(prompt);

    if (tokens.empty()) {
        throw std::invalid_argument("Inference::generate: prompt encoded to zero tokens");
    }

    const size_t context_length = model_.config().context_length;

    for (size_t i = 0; i < max_new_tokens; ++i) {
        // max context length tokens
        const size_t start = tokens.size() > context_length ? tokens.size() - context_length : 0;
        const std::vector<uint32_t> window(tokens.begin() + start, tokens.end());

        const Tensor logits = model_.forward(window, 1, window.size());   // [1, T, vocab]

        tokens.push_back(ops::argmax(logits).back());   // argmax of the last position
    }

    return tokenizer_.decode(tokens);
}