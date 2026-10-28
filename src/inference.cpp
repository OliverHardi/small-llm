#include "inference.h"

#include "ops/softmax.h"
#include "data/sft.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>



Inference::Inference(Model& model, Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

std::string Inference::generateTokens(std::vector<uint32_t> tokens, size_t max_new_tokens, SamplingOptions opt) {

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

    std::mt19937_64 rng(opt.seed);

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
        std::vector<float> l(data + offset, data + offset + vocab_size);

        // repetition penalty: once per distinct token already in the sequence
        if (opt.repetition_penalty != 1.0f) {
            std::vector<char> seen(vocab_size, 0);
            for (uint32_t t : tokens) {
                if (t < vocab_size && !seen[t]) {
                    seen[t] = 1;
                    l[t] = l[t] > 0 ? l[t] / opt.repetition_penalty
                                    : l[t] * opt.repetition_penalty;
                }
            }
        }

        size_t next = 0;
        if (opt.temperature <= 0.0f) {
            float best = -std::numeric_limits<float>::infinity();
            for (size_t v = 0; v < vocab_size; ++v)
                if (std::isfinite(l[v]) && l[v] > best) { best = l[v]; next = v; }
        } else {
            float mx = -std::numeric_limits<float>::infinity();
            for (float x : l) if (std::isfinite(x) && x > mx) mx = x;
            if (!std::isfinite(mx))
                throw std::runtime_error("Inference::generate: all logits are non-finite");

                    std::vector<float> scores(data + offset, data + offset + vocab_size);  // raw logits for the last position

        // repetition_penalty: make tokens that already appeared less likely (each distinct token once)
        if (opt.repetition_penalty != 1.0f) {
            std::vector<bool> already_penalized(vocab_size, false);
            for (uint32_t id : tokens) {
                if (id >= vocab_size || already_penalized[id]) continue;
                already_penalized[id] = true;
                scores[id] = scores[id] > 0 ? scores[id] / opt.repetition_penalty
                                            : scores[id] * opt.repetition_penalty;
            }
        }

        uint32_t next_token = 0;

        if (opt.temperature <= 0.0f) {
            // temperature = 0 means greedy: take the highest score
            next_token = static_cast<uint32_t>(
                std::max_element(scores.begin(), scores.end()) - scores.begin());
        } else {
            const float max_score = *std::max_element(scores.begin(), scores.end());
            if (!std::isfinite(max_score))
                throw std::runtime_error("Inference::generate: all logits are non-finite");

            // softmax: turn scores into probabilities (subtracting max_score keeps exp() from overflowing)
            std::vector<std::pair<double, uint32_t>> candidates;   // (probability, token id)
            candidates.reserve(vocab_size);
            double total = 0.0;
            for (size_t id = 0; id < vocab_size; ++id) {
                // temperature: dividing the scores by it makes the distribution sharper (<1) or flatter (>1)
                const double weight = std::exp((scores[id] - max_score) / opt.temperature);
                candidates.push_back({weight, static_cast<uint32_t>(id)});
                total += weight;
            }
            for (auto& candidate : candidates) candidate.first /= total;

            // most likely tokens first
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

            // top_p: keep adding the most likely tokens until their probabilities add up to top_p
            double cumulative = 0.0;
            size_t kept = 0;

            while (kept < candidates.size() && cumulative < opt.top_p) {
                cumulative += candidates[kept].first;
                ++kept;
            }
            candidates.resize(kept);

            // draw one token from the kept ones (the weights don't need to sum to 1 here)
            std::vector<double> kept_probabilities;
            for (const auto& candidate : candidates) kept_probabilities.push_back(candidate.first);
            std::discrete_distribution<size_t> pick(kept_probabilities.begin(), kept_probabilities.end());
            next_token = candidates[pick(rng)].second;
            // std::cout << "next_token: " << tokenizer_.decode({next_token}) << "  (probability: " << candidates[pick(rng)].first << ")\n";
        }

        tokens.push_back(next_token);
        if (next_token == tokenizer_.getEosToken() ||
            static_cast<int64_t>(next_token) == opt.stop_token) break;

        }
    }

    // Decode and print the complete prompt + generated continuation once.

    const std::string result = tokenizer_.decode(tokens);
    std::cout << result << '\n';
    std:: cout << "------------------------------------------------------------ \n";


    return result;
}

std::string Inference::generate(const std::string& prompt, size_t max_new_tokens, SamplingOptions opt) {
    return generateTokens(tokenizer_.encode(prompt), max_new_tokens, opt);
}

// Wraps the text in the chat template and stops at <|end|>.
std::string Inference::generateChat(const std::string& user_text, size_t max_new_tokens, SamplingOptions opt) {
    opt.stop_token = chatTokens(tokenizer_).end;
    return generateTokens(encodeChatPrompt(tokenizer_, user_text), max_new_tokens, opt);
}