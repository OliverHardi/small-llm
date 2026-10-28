#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Blends several tokenized corpora into shards that each contain the target mix.
// Input files: uint32 tokens, every document ends with the EOS token (what tokenizeParquet writes).
// Output (in out_dir): mixed-00000.bin, mixed-00001.bin, ...  plus val_<corpus name>.bin per corpus.
// Documents are shuffled within each corpus and kept whole; small corpora repeat (reshuffled) to fill their share.

struct CorpusSpec {
    std::string name;                 // used in the val file name and the report
    std::vector<std::string> files;   // tokenized .bin files
    double weight;                    // relative share of the output TOKENS (e.g. 35, 15, 10, ...)
};

struct MixOptions {
    uint32_t eos_token = 0;
    uint64_t total_tokens = 0;                 // training tokens to write
    uint64_t shard_tokens = 128'000'000;       // a shard is closed once it reaches this size
    double val_fraction = 0.005;               // share of each corpus's documents held out for validation
    uint32_t seed = 1234;
    std::string out_dir;
};

void mixCorpora(const std::vector<CorpusSpec>& corpora, const MixOptions& options);

// All *.bin files in a directory, sorted by name.
std::vector<std::string> listBinFiles(const std::string& dir);