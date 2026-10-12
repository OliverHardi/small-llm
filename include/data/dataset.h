#pragma once

#include <string>
#include <vector>

class Dataset{
public:
    Dataset(std::vector<std::string> files);

    std::vector<uint32_t> readTokens(uint64_t start, uint64_t count);

    uint64_t totalTokens() const;

private:

    std::vector<std::string> files_;
    std::vector<uint64_t> file_token_counts_;
    uint64_t total_tokens_ = 0;
};