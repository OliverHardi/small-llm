#include "data/mix.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

struct Doc {
    uint32_t file;      // index into the corpus's file list
    uint64_t offset;    // in tokens, from the start of that file
    uint32_t length;    // in tokens, including the trailing EOS
};

struct Corpus {
    const CorpusSpec* spec = nullptr;
    std::vector<std::unique_ptr<std::ifstream>> inputs;
    std::vector<Doc> train;        // shuffled; reshuffled each time it is used up
    std::vector<Doc> val;
    uint64_t train_tokens = 0;
    size_t next = 0;
    uint64_t emitted = 0;
    uint64_t passes = 0;
    std::mt19937_64 rng;
};

// Scans a file once, recording where each document starts and how long it is.
void indexFile(const std::string& path, uint32_t file_id, uint32_t eos, std::vector<Doc>& docs) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("mix: cannot open " + path);

    constexpr size_t kChunk = 1u << 24;                 // 16M tokens = 64 MB per read
    std::vector<uint32_t> buf(kChunk);
    uint64_t pos = 0, doc_start = 0;

    while (in) {
        in.read(reinterpret_cast<char*>(buf.data()), kChunk * sizeof(uint32_t));
        const size_t n = static_cast<size_t>(in.gcount()) / sizeof(uint32_t);
        for (size_t i = 0; i < n; ++i) {
            if (buf[i] == eos) {
                docs.push_back({file_id, doc_start, static_cast<uint32_t>(pos + i + 1 - doc_start)});
                doc_start = pos + i + 1;
            }
        }
        pos += n;
    }
    if (doc_start < pos) {
        std::cerr << "mix: " << path << ": " << (pos - doc_start) << " trailing tokens without EOS ignored\n";
    }
}

void readDoc(Corpus& c, const Doc& d, std::vector<uint32_t>& out) {
    out.resize(d.length);
    std::ifstream& in = *c.inputs[d.file];
    in.clear();
    in.seekg(static_cast<std::streamoff>(d.offset * sizeof(uint32_t)), std::ios::beg);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(d.length * sizeof(uint32_t)));
    if (in.gcount() != static_cast<std::streamsize>(d.length * sizeof(uint32_t))) {
        throw std::runtime_error("mix: short read from " + c.spec->files[d.file]);
    }
}

struct ShardWriter {
    std::string dir;
    uint64_t shard_tokens;
    int index = 0;
    uint64_t in_shard = 0, total = 0;
    std::ofstream out;

    void write(const std::vector<uint32_t>& t) {
        if (!out.is_open() || in_shard >= shard_tokens) {
            char name[64];
            std::snprintf(name, sizeof(name), "mixed-%05d.bin", index++);
            out.close();
            out.clear();
            out.open((fs::path(dir) / name).string(), std::ios::binary);
            if (!out) throw std::runtime_error(std::string("mix: cannot create ") + name);
            in_shard = 0;
        }
        out.write(reinterpret_cast<const char*>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(uint32_t)));
        if (!out) throw std::runtime_error("mix: write failed (disk full?)");
        in_shard += t.size();
        total += t.size();
    }
};

} // namespace

std::vector<std::string> listBinFiles(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".bin") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

void mixCorpora(const std::vector<CorpusSpec>& specs, const MixOptions& opt) {
    if (specs.empty() || opt.total_tokens == 0) throw std::invalid_argument("mix: nothing to do");
    fs::create_directories(opt.out_dir);

    // 1. index every corpus, shuffle its documents, hold out the validation documents
    std::vector<Corpus> corpora(specs.size());
    for (size_t k = 0; k < specs.size(); ++k) {
        Corpus& c = corpora[k];
        c.spec = &specs[k];
        if (specs[k].weight <= 0.0) throw std::invalid_argument("mix: weights must be positive");
        c.rng.seed(opt.seed + 7919ull * (k + 1));

        std::vector<Doc> docs;
        for (size_t f = 0; f < specs[k].files.size(); ++f) {
            c.inputs.push_back(std::make_unique<std::ifstream>(specs[k].files[f], std::ios::binary));
            indexFile(specs[k].files[f], static_cast<uint32_t>(f), opt.eos_token, docs);
        }
        if (docs.size() < 2) throw std::runtime_error("mix: corpus '" + specs[k].name + "' has fewer than 2 documents");

        std::shuffle(docs.begin(), docs.end(), c.rng);
        const size_t val_count = std::max<size_t>(1, static_cast<size_t>(docs.size() * opt.val_fraction));
        c.val.assign(docs.begin(), docs.begin() + val_count);
        c.train.assign(docs.begin() + val_count, docs.end());
        for (const Doc& d : c.train) c.train_tokens += d.length;

        std::cout << "  " << specs[k].name << ": " << c.train.size() << " train docs ("
                  << c.train_tokens << " tokens), " << c.val.size() << " val docs\n";
    }

    std::vector<uint32_t> buf;

    // 2. validation files, one per corpus (so you can track validation loss per domain)
    for (Corpus& c : corpora) {
        std::ofstream out((fs::path(opt.out_dir) / ("val_" + c.spec->name + ".bin")).string(), std::ios::binary);
        for (const Doc& d : c.val) {
            readDoc(c, d, buf);
            out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(uint32_t)));
        }
    }

    // 3. training shards: always take the next document from the corpus furthest behind its share
    ShardWriter writer{opt.out_dir, opt.shard_tokens};
    while (writer.total < opt.total_tokens) {
        size_t pick = 0;
        double best = 1e300;
        for (size_t k = 0; k < corpora.size(); ++k) {
            const double progress = static_cast<double>(corpora[k].emitted) / corpora[k].spec->weight;
            if (progress < best) { best = progress; pick = k; }
        }
        Corpus& c = corpora[pick];
        if (c.next >= c.train.size()) {                 // used up: reshuffle and go around again
            std::shuffle(c.train.begin(), c.train.end(), c.rng);
            c.next = 0;
            ++c.passes;
        }
        const Doc& d = c.train[c.next++];
        readDoc(c, d, buf);
        writer.write(buf);
        c.emitted += d.length;

        if ((writer.total & ((1ull << 26) - 1)) < d.length) {          // progress roughly every 64M tokens
            std::cout << "\r  wrote " << writer.total / 1'000'000 << "M / " << opt.total_tokens / 1'000'000 << "M tokens" << std::flush;
        }
    }
    std::cout << "\n";

    // 4. report
    std::cout << "Mix written to " << opt.out_dir << " (" << writer.index << " shards, " << writer.total << " tokens)\n";
    for (const Corpus& c : corpora) {
        const double share = 100.0 * static_cast<double>(c.emitted) / static_cast<double>(writer.total);
        const double repeats = static_cast<double>(c.emitted) / static_cast<double>(c.train_tokens);
        std::printf("  %-12s share %5.1f%%   used %.2f x its data%s\n", c.spec->name.c_str(), share, repeats,
                    repeats > 4.0 ? "   <-- repeated more than 4 times, may overfit" : "");
    }
}