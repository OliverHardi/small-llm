#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define NUM_MERGE 10

using Token = uint32_t;
using WordID = uint32_t;

struct Pair {
    Token first;
    Token second;
    
    bool operator==(const Pair& other) const {
        return first == other.first &&
               second == other.second;
    }
};

struct PairHash {
    size_t operator()(const Pair& p) const {
        return (static_cast<size_t>(p.first) << 32)
             ^ static_cast<size_t>(p.second);
    }
};

struct Candidate {
    Pair pair;
    uint64_t count;
    uint64_t version;

    bool operator<(const Candidate& other) const {
        return count < other.count;
    }
};

struct TokenVecHash {
    size_t operator()(const std::vector<Token>& v) const {
        size_t h = v.size();
        for (Token t : v) {
            h ^= std::hash<Token>{}(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

class Tokenizer {

public:
    Tokenizer() = default;

    void setMerges(int n) { num_merges_ = n; }

    void train(const std::string& text);

    std::vector<Token> encode(const std::string& text) const;
    std::string decode(const std::vector<Token>& tokens) const;

    void save(const std::string& path);
    void load(const std::string& path);

    const std::vector<std::pair<Pair, Token>>& merges() const { return merges_; }

    Token getEosToken() const { return 256 + num_merges_; } // 256 + 32506 = 32762 (5 blank left)
    
    size_t vocabSize() const { return getEosToken() + 1; }

private:

    void forEachMatch(const std::string& text,
                       const std::function<void(size_t, size_t)>& callback) const;

    void buildCorpusFromText(const std::string& text);

    int num_merges_;
    Token next_token_id_ = 256;

    // 0-255 -> base tokens
    // 256-32255 -> added tokens
    // 32256 -> EOS token
    std::vector<std::vector<Token>> corpus_;
    std::vector<uint64_t> word_freq_;

    std::unordered_map<Pair, uint64_t, PairHash> counts_;
    std::unordered_map<Pair, std::unordered_set<WordID>, PairHash> occurrences_;
    std::unordered_map<Pair, uint64_t, PairHash> versions_;
    std::priority_queue<Candidate> heap_;

    std::vector<std::pair<Pair, Token>> merges_;

    std::unordered_map<Token, std::vector<uint8_t>> token_bytes_;

    std::unordered_map<Pair, uint32_t, PairHash> merge_rank_;
    std::unordered_map<Pair, Token, PairHash> pair_to_token_;

    bool getBestPair(Candidate& result);

    std::vector<Token> mergeInSequence(const std::vector<Token>& sequence, Pair target, Token new_token) const;

    void adjustCount(Pair pair, int64_t delta, std::unordered_set<Pair, PairHash>& dirty);

    void removeWordFromIndex(
        const std::vector<Token>& sequence, WordID word_id, uint64_t freq,
        std::unordered_set<Pair, PairHash>& dirty);
    
    void addWordToIndex(
        const std::vector<Token>& sequence, WordID word_id, uint64_t freq,
        std::unordered_set<Pair, PairHash>& dirty);

    void buildInitialIndex();

    void initBaseTokenBytes();

    std::vector<Token> bpeMergeWord(std::vector<Token> seq) const;

};