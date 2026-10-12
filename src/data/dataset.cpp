#include "data/dataset.h"

#include <iostream>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <algorithm>

Dataset::Dataset(std::vector<std::string> files) : files_(std::move(files)) {
    file_token_counts_.reserve(files_.size());

    for (const std::string& file : files_) {
        std::ifstream input(file, std::ios::binary | std::ios::ate);

        std::streamsize file_size = input.tellg();

        uint64_t token_count = static_cast<uint64_t>(file_size) / sizeof(uint32_t);

        file_token_counts_.push_back(token_count);
        total_tokens_ += token_count;
    }

}

std::vector<uint32_t> Dataset::readTokens(uint64_t start, uint64_t count) {
    if (count == 0) { return {}; }

    if (start + count > total_tokens_) {
        count = total_tokens_ - start;
    }

    std::vector<uint32_t> tokens;
    tokens.reserve(count);

    uint64_t remaining_start = start;
    uint64_t remaining_count = count;

    for (size_t i = 0; i < files_.size() && remaining_count > 0; ++i) {
        uint64_t file_tokens = file_token_counts_[i];

        // this file is before our start pos
        if (remaining_start >= file_tokens) {
            remaining_start -= file_tokens;
            continue;
        }

        uint64_t file_offset = remaining_start;

        uint64_t available = file_tokens - file_offset;
        uint64_t to_read = std::min(available, remaining_count);

        std::ifstream input(files_[i], std::ios::binary);

        input.seekg(
            static_cast<std::streamoff>(
                file_offset * sizeof(uint32_t)
            ),
            std::ios::beg
        );

        size_t old_size = tokens.size();

        tokens.resize(old_size + to_read);

        input.read(
            reinterpret_cast<char*>(tokens.data() + old_size),
            static_cast<std::streamsize>(
                to_read * sizeof(uint32_t)
            )
        );

        if (input.gcount() !=
            static_cast<std::streamsize>(
                to_read * sizeof(uint32_t)
            )) {

            throw std::runtime_error(
                "Failed to read expected number of tokens from: "
                + files_[i]
            );
        }

        remaining_count -= to_read;
        remaining_start = 0;

    }

    return tokens;
}

uint64_t Dataset::totalTokens() const {
    return total_tokens_;
}