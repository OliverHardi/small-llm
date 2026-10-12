#include "tokenizer.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept> 
#include <string_view>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

// pretok + dedup

void Tokenizer::forEachMatch(
    const std::string& text,
    const std::function<void(size_t, size_t)>& callback
) const {
    PCRE2_SPTR pattern = (PCRE2_SPTR)"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    int errorcode;
    PCRE2_SIZE erroroffset;
 
    pcre2_code* re = pcre2_compile(
        pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &errorcode,
        &erroroffset,
        NULL
    );
 
    if (!re) return;
 
    int jit_rc = pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
    bool use_jit = (jit_rc == 0);
 
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);
 
    PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(text.data());
    PCRE2_SIZE subject_length = text.size();
    PCRE2_SIZE start_offset = 0;
 
    while (start_offset < subject_length) {
        int rc = use_jit
            ? pcre2_jit_match(re, subject, subject_length, start_offset, 0, match_data, NULL)
            : pcre2_match(re, subject, subject_length, start_offset, 0, match_data, NULL);
 
        if (rc < 0) break;
 
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end = ovector[1];
 
        callback(match_start, match_end - match_start);
 
        start_offset = match_end;
    }
 
    pcre2_match_data_free(match_data);
    pcre2_code_free(re);
}

void Tokenizer::buildCorpusFromText(const std::string& text) {
 
    std::unordered_map<std::string_view, WordID> seen;
    seen.reserve(1 << 16);
 
    const char* raw = text.data();
 
    forEachMatch(text, [&](size_t start, size_t len) {
        std::string_view word(raw + start, len);
 
        auto it = seen.find(word);
 
        if (it == seen.end()) {
            WordID id = static_cast<WordID>(corpus_.size());
 
            std::vector<Token> tokens;
            tokens.reserve(word.size());
 
            for (unsigned char byte : word) {
                tokens.push_back(static_cast<Token>(byte));
            }
 
            corpus_.push_back(std::move(tokens));
            word_freq_.push_back(1);
 
            seen.emplace(word, id);
        } else {
            word_freq_[it->second]++;
        }
    });
}

void Tokenizer::initBaseTokenBytes() {
    for (int b = 0; b < 256; ++b) {
        token_bytes_[static_cast<Token>(b)] = { static_cast<uint8_t>(b) };
    }
}

// pair/index

bool Tokenizer::getBestPair(Candidate& result) {
    while (!heap_.empty()) {
        Candidate candidate = heap_.top();
        heap_.pop();
 
        auto count_it = counts_.find(candidate.pair);
        auto version_it = versions_.find(candidate.pair);
 
        if (count_it == counts_.end() || version_it == versions_.end()) {
            continue;
        }
 
        if (count_it->second != candidate.count ||
            version_it->second != candidate.version) {
            continue;
        }
 
        result = candidate;
        return true;
    }
 
    return false;
}

std::vector<Token> Tokenizer::mergeInSequence(
    const std::vector<Token>& sequence,
    Pair target,
    Token new_token
) const {
    std::vector<Token> result;
    result.reserve(sequence.size());
 
    size_t i = 0;
 
    while (i < sequence.size()) {
        if (i + 1 < sequence.size() &&
            sequence[i] == target.first &&
            sequence[i + 1] == target.second) {
 
            result.push_back(new_token);
            i += 2;
        } else {
            result.push_back(sequence[i]);
            ++i;
        }
    }
 
    return result;
}

void Tokenizer::adjustCount(
    Pair pair,
    int64_t delta,
    std::unordered_set<Pair, PairHash>& dirty
) {
    if (delta > 0) {
        counts_[pair] += static_cast<uint64_t>(delta);
    } else {
        uint64_t amount = static_cast<uint64_t>(-delta);
 
        auto it = counts_.find(pair);
 
        if (it == counts_.end() || it->second < amount) {
            throw std::runtime_error("Invalid pair count decrement");
        }
 
        it->second -= amount;
 
        if (it->second == 0) {
            counts_.erase(it);
        }
    }
 
    versions_[pair]++;
    dirty.insert(pair);
}

void Tokenizer::removeWordFromIndex(
    const std::vector<Token>& sequence,
    WordID word_id,
    uint64_t freq,
    std::unordered_set<Pair, PairHash>& dirty
) {
    std::unordered_set<Pair, PairHash> seen;
 
    for (size_t i = 0; i + 1 < sequence.size(); ++i) {
        Pair pair{sequence[i], sequence[i + 1]};
 
        adjustCount(pair, -static_cast<int64_t>(freq), dirty);
 
        seen.insert(pair);
    }
 
    for (const Pair& pair : seen) {
        auto occurrence_it = occurrences_.find(pair);
 
        if (occurrence_it != occurrences_.end()) {
            occurrence_it->second.erase(word_id);
 
            if (occurrence_it->second.empty()) {
                occurrences_.erase(occurrence_it);
            }
        }
    }
}


void Tokenizer::addWordToIndex(
    const std::vector<Token>& sequence,
    WordID word_id,
    uint64_t freq,
    std::unordered_set<Pair, PairHash>& dirty
) {
    std::unordered_set<Pair, PairHash> seen;
 
    for (size_t i = 0; i + 1 < sequence.size(); ++i) {
        Pair pair{sequence[i], sequence[i + 1]};
 
        adjustCount(pair, static_cast<int64_t>(freq), dirty);
 
        seen.insert(pair);
    }
 
    for (const Pair& pair : seen) {
        occurrences_[pair].insert(word_id);
    }
}

// decoding
 
std::string Tokenizer::decode(const std::vector<Token>& tokens) const {
    std::string result;
 
    for (Token t : tokens) {

        if (t == getEosToken()) {
            result += ("\n<EOS>\n");
            continue;
        }

        auto it = token_bytes_.find(t);
        if (it == token_bytes_.end()) {
            throw std::runtime_error(
                "Unknown token ID during decode: " + std::to_string(t) +
                ", expected EOS: " + std::to_string(getEosToken())
            );
        }

        const auto& bytes = token_bytes_.at(t); // throws if t was never learned
 
        result.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
 
    return result;
}


// encoding

std::vector<Token> Tokenizer::bpeMergeWord(std::vector<Token> seq) const {
    while (seq.size() > 1) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        Pair best_pair{0, 0};
 
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            Pair pair{seq[i], seq[i + 1]};
 
            auto it = merge_rank_.find(pair);
 
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = pair;
            }
        }
 
        if (best_rank == std::numeric_limits<uint32_t>::max()) {
            break;
        }
 
        Token new_token = pair_to_token_.at(best_pair);
        seq = mergeInSequence(seq, best_pair, new_token);
    }
 
    return seq;
}
 
std::vector<Token> Tokenizer::encode(const std::string& text) const {
    std::vector<Token> result;
 
    // cache
    std::unordered_map<std::string_view, std::vector<Token>> cache;
 
    const char* raw = text.data();
 
    forEachMatch(text, [&](size_t start, size_t len) {
        std::string_view word(raw + start, len);
 
        auto it = cache.find(word);
 
        if (it != cache.end()) {
            result.insert(result.end(), it->second.begin(), it->second.end());
            return;
        }
 
        std::vector<Token> seq;
        seq.reserve(len);
 
        for (unsigned char byte : word) {
            seq.push_back(static_cast<Token>(byte));
        }
 
        seq = bpeMergeWord(std::move(seq));
 
        result.insert(result.end(), seq.begin(), seq.end());
        cache.emplace(word, std::move(seq));
    });
 
    return result;
}

void Tokenizer::buildInitialIndex() {
    for (WordID word_id = 0; word_id < corpus_.size(); ++word_id) {
        const auto& sequence = corpus_[word_id];
        uint64_t freq = word_freq_[word_id];
 
        std::unordered_set<Pair, PairHash> seen_in_word;
 
        for (size_t i = 0; i + 1 < sequence.size(); ++i) {
            Pair pair{sequence[i], sequence[i + 1]};
 
            counts_[pair] += freq;
 
            if (seen_in_word.insert(pair).second) {
                occurrences_[pair].insert(word_id);
            }
        }
    }
 
    for (const auto& [pair, count] : counts_) {
        versions_[pair] = 0;
        heap_.push(Candidate{pair, count, versions_[pair]});
    }

}

// training


void Tokenizer::train(const std::string& text) {

    buildCorpusFromText(text);

    initBaseTokenBytes();

    buildInitialIndex();
 
    for (int merge_number = 0; merge_number < num_merges_; ++merge_number) {
 
        Candidate best;
 
        if (!getBestPair(best)) {
            std::cout << "No more pairs available.\n";
            break;
        }
 
        if (best.count < 2) {
            std::cout << "Stopping: no pair occurs at least twice.\n";
            break;
        }
 
        Token new_token = next_token_id_++;

        {                                            // (3) add this block
            std::vector<uint8_t> merged_bytes = token_bytes_.at(best.pair.first);
            const auto& second_bytes = token_bytes_.at(best.pair.second);
            merged_bytes.insert(merged_bytes.end(), second_bytes.begin(), second_bytes.end());
            token_bytes_[new_token] = std::move(merged_bytes);
        }
 
        auto occurrence_it = occurrences_.find(best.pair);
 
        if (occurrence_it == occurrences_.end()) {
            continue;
        }
 
        // copy
        std::vector<WordID> affected_words(
            occurrence_it->second.begin(),
            occurrence_it->second.end()
        );
 
        std::unordered_set<Pair, PairHash> dirty;
 
        for (WordID word_id : affected_words) {
            uint64_t freq = word_freq_[word_id];
 
            std::vector<Token> old_sequence = std::move(corpus_[word_id]);
 
            removeWordFromIndex(old_sequence, word_id, freq, dirty);
 
            std::vector<Token> new_sequence =
                mergeInSequence(old_sequence, best.pair, new_token);
 
            corpus_[word_id] = std::move(new_sequence);
 
            addWordToIndex(corpus_[word_id], word_id, freq, dirty);
        }

        merge_rank_[best.pair] = static_cast<uint32_t>(merges_.size());
        pair_to_token_[best.pair] = new_token;
        merges_.push_back({best.pair, new_token});
 
        for (const Pair& pair : dirty) {
            auto count_it = counts_.find(pair);
 
            if (count_it != counts_.end()) {
                heap_.push(Candidate{pair, count_it->second, versions_[pair]});
            }
        }
        if ((merge_number + 1) % 10 == 0 || merge_number + 1 == num_merges_) {
            float progress = 100.0f * (merge_number + 1) / num_merges_;
            std::cout << "\rTraining tokenizer: "<< std::format("{:.1f}", progress) << "%" << std::flush;
        }
    }
    std::cout << std::endl;
}

// saving/loading

void Tokenizer::save(const std::string& path) {
    std::ofstream out(path, std::ios::binary);

    // next token ID
    out.write(
        reinterpret_cast<const char*>(&next_token_id_),
        sizeof(next_token_id_)
    );

    // vocab size
    uint64_t vocab_size = token_bytes_.size();
    out.write(
        reinterpret_cast<const char*>(&vocab_size),
        sizeof(vocab_size)
    );

    // vocab list
    for (const auto& [token, bytes] : token_bytes_) {

        out.write(
            reinterpret_cast<const char*>(&token),
            sizeof(token)
        );

        uint64_t size = bytes.size();

        out.write(
            reinterpret_cast<const char*>(&size),
            sizeof(size)
        );

        out.write(
            reinterpret_cast<const char*>(bytes.data()),
            size
        );
    }

    // Number of merges
    int num_merges = merges_.size();

    out.write(
        reinterpret_cast<const char*>(&num_merges),
        sizeof(num_merges)
    );

    // Merge list
    for (const auto& [pair, token] : merges_) {

        out.write(
            reinterpret_cast<const char*>(&pair.first),
            sizeof(pair.first)
        );

        out.write(
            reinterpret_cast<const char*>(&pair.second),
            sizeof(pair.second)
        );

        out.write(
            reinterpret_cast<const char*>(&token),
            sizeof(token)
        );
    }

}

void Tokenizer::load(const std::string& path) {

    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "Failed to open tokenizer file"
        );
    }

    // Next token ID
    in.read(
        reinterpret_cast<char*>(&next_token_id_),
        sizeof(next_token_id_)
    );

    // Vocabulary
    uint64_t vocab_size;

    in.read(
        reinterpret_cast<char*>(&vocab_size),
        sizeof(vocab_size)
    );

    token_bytes_.clear();

    for (uint64_t i = 0; i < vocab_size; ++i) {

        Token token;

        in.read(
            reinterpret_cast<char*>(&token),
            sizeof(token)
        );

        uint64_t size;

        in.read(
            reinterpret_cast<char*>(&size),
            sizeof(size)
        );

        std::vector<uint8_t> bytes(size);

        in.read(
            reinterpret_cast<char*>(bytes.data()),
            size
        );

        token_bytes_[token] = std::move(bytes);
    }

    // Merges
    int num_merges;

    in.read(
        reinterpret_cast<char*>(&num_merges),
        sizeof(num_merges)
    );

    num_merges_ = num_merges;

    merges_.clear();
    merge_rank_.clear();
    pair_to_token_.clear();

    for (int i = 0; i < num_merges; ++i) {

        Pair pair;
        Token token;

        in.read(
            reinterpret_cast<char*>(&pair.first),
            sizeof(pair.first)
        );

        in.read(
            reinterpret_cast<char*>(&pair.second),
            sizeof(pair.second)
        );

        in.read(
            reinterpret_cast<char*>(&token),
            sizeof(token)
        );

        merges_.push_back({pair, token});

        merge_rank_[pair] =
            static_cast<uint32_t>(i);

        pair_to_token_[pair] = token;
    }
}