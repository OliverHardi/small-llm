#pragma once

#include <vector>

#include "data/dataset.h"
#include "tensor.h"


struct Batch {
    std::vector<uint32_t> tokens;

    size_t batch_size;
    size_t context_length;

    size_t T() const { return tokens.size() / batch_size - 1; }

    std::vector<uint32_t> inputs() const {
        const size_t seq_len = T(), row_len = seq_len + 1;
        std::vector<uint32_t> out(batch_size * seq_len);
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = tokens[(i / seq_len) * row_len + (i % seq_len)];
        }
        return out;
    }

    Tensor targets() const {
        const size_t seq_len = T(), row_len = seq_len + 1;
        Tensor out({batch_size * seq_len});
        for (size_t i = 0; i < batch_size * seq_len; ++i) {
            out[i] = static_cast<float>(tokens[(i / seq_len) * row_len + (i % seq_len) + 1]);
        }
        return out;
    }
};

class DataLoader {
public:
    DataLoader(Dataset& dataset, size_t batch_size, size_t context_length);
    
    void reset();

    Batch nextBatch();

    uint64_t totalTokens() {
        return dataset_.totalTokens();
    }
    
    uint64_t position() const { return position_; }
    void setPosition(uint64_t p) { position_ = p; }

private:
    Dataset& dataset_;

    size_t batch_size_;
    size_t context_length_;

    uint64_t position_;

};