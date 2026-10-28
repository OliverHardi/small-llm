#pragma once

// Supervised fine-tuning support
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"
#include "tokenizer.h"

struct ChatTokens {
    uint32_t user, assistant, end, eos;
};


inline ChatTokens chatTokens(const Tokenizer& tokenizer) {
    const uint32_t eos = tokenizer.getEosToken();
    return {eos + 1, eos + 2, eos + 3, eos};
}

inline std::vector<uint32_t> encodeChatPrompt(const Tokenizer& tokenizer, const std::string& user_text) {
    const ChatTokens ids = chatTokens(tokenizer);
    std::vector<uint32_t> tokens{ids.user};
    for (uint32_t id : tokenizer.encode(user_text)) tokens.push_back(id);
    tokens.push_back(ids.end);
    tokens.push_back(ids.assistant);
    return tokens;
}

// chat.bin record
inline void prepareSft(const std::string& chat_path, const std::string& out_path, const Tokenizer& tokenizer) {
    const ChatTokens ids = chatTokens(tokenizer);
    std::cout << "chat tokens: user=" << ids.user << " assistant=" << ids.assistant
              << " end=" << ids.end << " eos=" << ids.eos << "\n";

    std::ifstream in(chat_path, std::ios::binary);
    if (!in) throw std::runtime_error("prepareSft: cannot open " + chat_path);

    std::vector<uint32_t> tokens;
    std::vector<uint8_t> mask;
    auto push = [&](uint32_t id, bool is_target) {
        tokens.push_back(id);
        mask.push_back(is_target ? 1 : 0);
    };

    uint64_t dialogues = 0;
    uint32_t message_count = 0;
    while (in.read(reinterpret_cast<char*>(&message_count), sizeof(message_count))) {
        for (uint32_t m = 0; m < message_count; ++m) {
            uint8_t role = 0;
            uint32_t byte_length = 0;
            in.read(reinterpret_cast<char*>(&role), sizeof(role));
            in.read(reinterpret_cast<char*>(&byte_length), sizeof(byte_length));
            std::string text(byte_length, '\0');
            in.read(text.data(), byte_length);
            if (!in) throw std::runtime_error("prepareSft: truncated chat file");

            const bool is_assistant = (role == 1);
            push(is_assistant ? ids.assistant : ids.user, false);                 // role marker
            for (uint32_t id : tokenizer.encode(text)) push(id, is_assistant);    // text
            push(ids.end, is_assistant);                                          // end of turn
        }
        push(ids.eos, false);   // separates dialogues
        if (++dialogues % 5000 == 0) std::cout << "\rdialogues: " << dialogues << std::flush;
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("prepareSft: cannot open " + out_path);
    const uint64_t n = tokens.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(tokens.data()), n * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(mask.data()), n);
    if (!out) throw std::runtime_error("prepareSft: write failed");

    const uint64_t targets = std::accumulate(mask.begin(), mask.end(), uint64_t{0});
    std::cout << "\ndialogues: " << dialogues << "  tokens: " << n << "  training targets: " << targets
              << " (" << (n ? 100.0 * targets / n : 0.0) << "%)\n";
}

struct SftBatch {
    std::vector<uint32_t> inputs;   // [batch_size * T]
    Tensor targets;                 // [batch_size, T], token id as float, -1 = no loss at this position
    size_t batch_size = 0;
    size_t T = 0;
    size_t num_valid = 0;           // positions that have a target (pass to ops::cross_entropy)
};

// Packs whole dialogues into windows of T+1 tokens
class SftLoader {
public:
    SftLoader(const std::string& path, uint32_t eos, size_t holdout_dialogues, bool validation,
              size_t batch_size, size_t T, uint64_t seed = 1)
        : batch_size_(batch_size), T_(T), pad_(eos), shuffle_(!validation), rng_(seed) {

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("SftLoader: cannot open " + path);

        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        tokens_.resize(n);
        mask_.resize(n);
        in.read(reinterpret_cast<char*>(tokens_.data()), n * sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(mask_.data()), n);
        if (!in) throw std::runtime_error("SftLoader: truncated " + path);

        // every dialogue ends with EOS (and EOS appears nowhere else)
        std::vector<size_t> starts{0};
        for (size_t i = 0; i < n; ++i) {
            if (tokens_[i] == eos) starts.push_back(i + 1);
        }
        const size_t total = starts.size() - 1;
        if (holdout_dialogues >= total) throw std::runtime_error("SftLoader: holdout is larger than the dataset");

        const size_t first = validation ? total - holdout_dialogues : 0;
        const size_t last = validation ? total : total - holdout_dialogues;
        for (size_t d = first; d < last; ++d) dialogues_.push_back({starts[d], starts[d + 1]});

        repack();
    }

    size_t numDialogues() const { return dialogues_.size(); }
    size_t batchesPerEpoch() const { return windows_.size() / batch_size_; }
    size_t epoch() const { return epoch_; }

    void reset() { cursor_ = 0; }

    SftBatch nextBatch() {
        SftBatch batch;
        batch.batch_size = batch_size_;
        batch.T = T_;
        batch.inputs.assign(batch_size_ * T_, pad_);
        batch.targets = Tensor(std::vector<size_t>{batch_size_, T_});
        float* targets = batch.targets.data();
        std::fill(targets, targets + batch_size_ * T_, -1.0f);

        for (size_t row = 0; row < batch_size_; ++row) {
            if (cursor_ >= windows_.size()) {   // epoch finished: new shuffle
                ++epoch_;
                repack();
            }

            std::vector<uint32_t> window_tokens;
            std::vector<uint8_t> window_mask;
            for (size_t d : windows_[cursor_]) {
                const auto [begin, end] = dialogues_[d];
                const size_t room = (T_ + 1) - window_tokens.size();
                const size_t length = std::min(end - begin, room);   // only clips a dialogue longer than a window
                window_tokens.insert(window_tokens.end(), tokens_.begin() + begin, tokens_.begin() + begin + length);
                window_mask.insert(window_mask.end(), mask_.begin() + begin, mask_.begin() + begin + length);
            }
            ++cursor_;

            for (size_t i = 0; i < T_ && i < window_tokens.size(); ++i) {
                batch.inputs[row * T_ + i] = window_tokens[i];
                if (i + 1 < window_tokens.size() && window_mask[i + 1]) {
                    targets[row * T_ + i] = static_cast<float>(window_tokens[i + 1]);
                    ++batch.num_valid;
                }
            }
        }
        return batch;
    }

private:
    void repack() {
        std::vector<size_t> order(dialogues_.size());
        std::iota(order.begin(), order.end(), size_t{0});
        if (shuffle_) std::shuffle(order.begin(), order.end(), rng_);

        windows_.clear();
        size_t used = 0;
        for (size_t d : order) {
            const size_t length = std::min(dialogues_[d].second - dialogues_[d].first, T_ + 1);
            if (windows_.empty() || used + length > T_ + 1) {   // doesn't fit: start a new window
                windows_.emplace_back();
                used = 0;
            }
            windows_.back().push_back(d);
            used += length;
        }
        cursor_ = 0;
    }

    size_t batch_size_, T_;
    uint32_t pad_;
    bool shuffle_;
    std::mt19937_64 rng_;

    std::vector<uint32_t> tokens_;
    std::vector<uint8_t> mask_;
    std::vector<std::pair<size_t, size_t>> dialogues_;   // [begin, end) in tokens_
    std::vector<std::vector<size_t>> windows_;           // dialogue indices per window
    size_t cursor_ = 0;
    size_t epoch_ = 0;
};