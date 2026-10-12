#include "checkpoint.h"
 
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
 
template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
 
template <typename T>
T read_pod(std::istream& in) {
    T value;
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}
 
} // namespace

void save_checkpoint(const std::string& path, Model& model, const AdamW& optimizer, const TrainState& state) {
    const std::string tmp_path = path + ".tmp";

    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out) throw std::runtime_error("save_checkpoint: cannot open " + tmp_path);

        write_pod(out, state.step);
        write_pod(out, state.data_position);

        // weights (name, shape, raw floats)
        const std::vector<Param> params = model.parameters();
        write_pod(out, static_cast<uint32_t>(params.size()));

        for (const Param& p : params) {
            write_pod(out, static_cast<uint32_t>(p.name.size()));
            out.write(p.name.data(), p.name.size());
 
            const std::vector<size_t>& shape = p.value->shape();
            write_pod(out, static_cast<uint32_t>(shape.size()));
            for (size_t d : shape) write_pod(out, static_cast<uint64_t>(d));
 
            out.write(reinterpret_cast<const char*>(p.value->data()), p.value->numel() * sizeof(float));
        }
        // save optimizer moment and step count
        optimizer.save(out);

        if (!out) throw std::runtime_error("save_checkpoint: write failed (disk full?)");
    }

    std::filesystem::rename(tmp_path, path);   // atomic replace
}
 
TrainState load_checkpoint(const std::string& path, Model& model, AdamW& optimizer) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("load_checkpoint: cannot open " + path);
 
    TrainState state;
    state.step = read_pod<uint64_t>(in);
    state.data_position = read_pod<uint64_t>(in);

    const std::vector<Param> params = model.parameters();
    if (read_pod<uint32_t>(in) != params.size()) {
        throw std::runtime_error("load_checkpoint: parameter count differs (different model config?)");
    }

    for (const Param& p : params) {
        std::string name(read_pod<uint32_t>(in), '\0');
        in.read(name.data(), name.size());
        if (name != p.name) {
            throw std::runtime_error("load_checkpoint: expected parameter '" + p.name + "', file has '" + name + "'");
        }

        const uint32_t ndim = read_pod<uint32_t>(in);
        std::vector<size_t> shape(ndim);
        for (uint32_t i = 0; i < ndim; ++i) shape[i] = read_pod<uint64_t>(in);
        if (shape != p.value->shape()) {
            throw std::runtime_error("load_checkpoint: shape mismatch for '" + p.name + "'");
        }

        in.read(reinterpret_cast<char*>(p.value->data()), p.value->numel() * sizeof(float));
    }

    optimizer.load(in);

    if (!in) throw std::runtime_error("load_checkpoint: file truncated or corrupt");
    return state;
}