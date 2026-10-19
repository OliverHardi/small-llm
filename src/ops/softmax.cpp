#include "ops/softmax.h"

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

Tensor ops::softmax(const Tensor& x, int axis, bool causal) {
    if (axis != -1 && axis != static_cast<int>(x.ndim()) - 1) {
        throw std::invalid_argument("softmax: only the last axis is supported");
    }

    const size_t cols = x.dim(x.ndim() - 1);
    const size_t rows = x.numel() / cols;

    #ifdef USE_CUDA
        if (x.is_cuda()) {
            Tensor out(x.shape(), Device::CUDA);
            cuda::softmax(x.data(), out.data(), x.numel() / cols, cols, causal);
            return out;
        }
    #endif

    Tensor result(x.shape());   // zero-initialized, so masked entries stay 0

    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * cols;
        const size_t valid = causal ? (r % cols) + 1 : cols;   // row r is query position r % T

        float max_value = x[base];
        for (size_t j = 1; j < valid; ++j) max_value = std::max(max_value, x[base + j]);

        float sum = 0.0f;
        for (size_t j = 0; j < valid; ++j) {
            result[base + j] = std::exp(x[base + j] - max_value);
            sum += result[base + j];
        }

        for (size_t j = 0; j < valid; ++j) result[base + j] /= sum;
    }
    return result;
}

// Row-wise: dx = p * (g - sum(g * p)). Masked entries have p = 0, so their gradient is 0
Tensor ops::softmax_backward(const Tensor& probs, const Tensor& grad_output) {
    const size_t cols = probs.dim(probs.ndim() - 1);
    const size_t rows = probs.numel() / cols;

    #ifdef USE_CUDA
        if (probs.is_cuda()) {
            require_same_device(probs, grad_output, "softmax_backward");
            Tensor dx(probs.shape(), Device::CUDA);
            cuda::softmax_backward(probs.data(), grad_output.data(), dx.data(),
                                probs.numel() / cols, cols);
            return dx;
        }
    #endif

    Tensor dx(probs.shape());

    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * cols;

        float dot = 0.0f;
        for (size_t j = 0; j < cols; ++j) dot += grad_output[base + j] * probs[base + j];

        for (size_t j = 0; j < cols; ++j) {
            dx[base + j] = probs[base + j] * (grad_output[base + j] - dot);
        }
    }
    return dx;
}

// i of the largest value in each row of the last dimension.
std::vector<uint32_t> ops::argmax(const Tensor& x) {
    const size_t cols = x.dim(x.ndim() - 1);
    const size_t rows = x.numel() / cols;

    std::vector<uint32_t> best(rows, 0);

    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * cols;
        for (size_t j = 1; j < cols; ++j) {
            if (x[base + j] > x[base + best[r]]) best[r] = static_cast<uint32_t>(j);
        }
    }
    return best;
}