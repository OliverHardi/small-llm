#include "ops/rope.h"

#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

namespace {

// rotate each consecutive pair (x[2i], x[2i+1]) by angle pos * theta^(-2i/head_dim).
Tensor apply_rope(const Tensor& x, float theta, float sign) {
    if (x.ndim() < 2) {
        throw std::invalid_argument("rope: expects at least 2 dimensions [..., T, head_dim]");
    }
 
    const size_t head_dim = x.dim(x.ndim() - 1);
    const size_t num_positions = x.dim(x.ndim() - 2);
 
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("rope: last dimension must be even");
    }
 
    const size_t pairs = head_dim / 2;
    Tensor result(x.shape());
 
    if (x.numel() == 0) {
        return result;
    }
 
    // precompute trig once per (position, pair)
    std::vector<float> cos_table(num_positions * pairs);
    std::vector<float> sin_table(num_positions * pairs);
 
    for (size_t pos = 0; pos < num_positions; ++pos) {
        for (size_t i = 0; i < pairs; ++i) {
            const float frequency =
                1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(pos) * frequency;
 
            cos_table[pos * pairs + i] = std::cos(angle);
            sin_table[pos * pairs + i] = sign * std::sin(angle);
        }
    }
 
    const size_t groups = x.numel() / (num_positions * head_dim);
 
    for (size_t group = 0; group < groups; ++group) {
        for (size_t pos = 0; pos < num_positions; ++pos) {
            const size_t base = (group * num_positions + pos) * head_dim;
 
            for (size_t i = 0; i < pairs; ++i) {
                const float c = cos_table[pos * pairs + i];
                const float s = sin_table[pos * pairs + i];
 
                const float x1 = x[base + 2 * i];
                const float x2 = x[base + 2 * i + 1];
 
                result[base + 2 * i]     = x1 * c - x2 * s;
                result[base + 2 * i + 1] = x1 * s + x2 * c;
            }
        }
    }
 
    return result;
}

} // namespace


Tensor ops::rope(const Tensor& x, float theta) {
    return apply_rope(x, theta, +1.0f);
}
 
Tensor ops::rope_backward(const Tensor& grad_output, float theta) {
    return apply_rope(grad_output, theta, -1.0f);
}