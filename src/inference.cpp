#include "inference.h"

#include "ops/softmax.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

Inference::Inference(Model& model, Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

std::string Inference::generate(
    const std::string& prompt,
    size_t max_new_tokens
) {
    std::vector<uint32_t> tokens = tokenizer_.encode(prompt);

    if (tokens.empty()) {
        throw std::invalid_argument(
            "Inference::generate: prompt encoded to zero tokens"
        );
    }

    const size_t context_length = model_.config().context_length;

    if (context_length == 0) {
        throw std::runtime_error(
            "Inference::generate: context length must be greater than zero"
        );
    }

    for (size_t step = 0; step < max_new_tokens; ++step) {
        const size_t start =
            tokens.size() > context_length
                ? tokens.size() - context_length
                : 0;

        const std::vector<uint32_t> window(
            tokens.begin() + start,
            tokens.end()
        );

        Tensor logits = model_.forward(
            window,
            1,
            window.size()
        );

        // Transfer logits to CPU for inspection.
        Tensor cpu_logits = logits.to(Device::CPU);

        if (cpu_logits.ndim() != 3 ||
            cpu_logits.dim(0) != 1 ||
            cpu_logits.dim(1) == 0 ||
            cpu_logits.dim(2) == 0) {
            throw std::runtime_error(
                "Inference::generate: unexpected logits shape"
            );
        }

        const size_t T = cpu_logits.dim(1);
        const size_t vocab_size = cpu_logits.dim(2);
        const float* data = cpu_logits.data();

        // Only inspect the final sequence position.
        const size_t offset = (T - 1) * vocab_size;

        // Greedy decoding: select the token with the highest logit.
        size_t best_id = 0;
        float best_value = -std::numeric_limits<float>::infinity();

        for (size_t v = 0; v < vocab_size; ++v) {
            const float value = data[offset + v];

            if (std::isfinite(value) && value > best_value) {
                best_value = value;
                best_id = v;
            }
        }

        if (!std::isfinite(best_value)) {
            throw std::runtime_error(
                "Inference::generate: all logits are non-finite"
            );
        }

        // Append the next token without printing it yet.
        tokens.push_back(static_cast<uint32_t>(best_id));
    }

    // Decode and print the complete prompt + generated continuation once.

    const std::string result = tokenizer_.decode(tokens);
    std::cout << result << '\n';
    std:: cout << "------------------------------------------------------------ \n";


    return result;
}