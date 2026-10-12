#include "data/dataloader.h"

#include <iostream>

DataLoader::DataLoader(
    Dataset& dataset,
    size_t batch_size,
    size_t context_length) :
        dataset_(dataset),
        batch_size_(batch_size),
        context_length_(context_length),
        position_(0) 
{
    // nothing here
}

void DataLoader::reset() {
    position_ = 0;
}

Batch DataLoader::nextBatch() {
    const uint64_t sequence_length = context_length_ + 1;

    const uint64_t total_needed = static_cast<uint64_t>(batch_size_) * sequence_length;

    if (position_ + total_needed > dataset_.totalTokens()) {
        position_ = 0;
    }

    std::vector<uint32_t> tokens = dataset_.readTokens(position_, total_needed);

    Batch batch;

    batch.batch_size = batch_size_;
    batch.context_length = context_length_;

    batch.tokens.resize( batch_size_ * (context_length_ + 1) );

    for (size_t b = 0; b < batch_size_; ++b) {

        const size_t token_start = b * sequence_length;
        const size_t batch_start = b * context_length_;

        for (size_t t = 0; t < context_length_ + 1; ++t) {
            batch.tokens[batch_start + t] = tokens[token_start + t];
        }
    }

    position_ += total_needed;

    return batch;
}