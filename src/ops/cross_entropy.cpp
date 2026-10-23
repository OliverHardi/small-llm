#include "ops/cross_entropy.h"

#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

// logits must be fp32
float ops::cross_entropy(const Tensor& logits, const Tensor& targets) {

    size_t vocab_size = logits.dim(logits.ndim() - 1);
    size_t num_predictions = logits.numel() / vocab_size;

    #ifdef USE_CUDA
        if (logits.is_cuda()) {
            require_same_device(logits, targets, "cross_entropy");
            const size_t V = logits.dim(logits.ndim() - 1);
            return cuda::cross_entropy(logits.data(), targets.data(), logits.numel() / V, V);
        }
    #endif

    double total_loss = 0.0f;

    for (size_t i = 0; i < num_predictions; ++i) {
        size_t base = i * vocab_size;

        float max_logit = logits[base];
        for (size_t j = 0; j < vocab_size; ++j) {
            max_logit = std::max(max_logit, logits[base + j]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            sum_exp += std::exp(logits[base + j] - max_logit);
        }

        float log_sum_exp = max_logit + std::log(sum_exp);

        size_t target = static_cast<size_t>(targets[i]);

        float loss = -logits[base + target] + log_sum_exp;

        total_loss += loss;
    }

    return static_cast<float>(total_loss / static_cast<double_t>(num_predictions));
}

Tensor ops::cross_entropy_backward(const Tensor& logits, const Tensor& targets, DType grad_dtype) {
    const size_t vocab_size = logits.dim(logits.ndim() - 1);
    const size_t num_predictions = logits.numel() / vocab_size;
    const float inv_n = 1.0f / static_cast<float>(num_predictions);

    #ifdef USE_CUDA
        if (logits.is_cuda()) {
            require_same_device(logits, targets, "cross_entropy_backward");
            const size_t V = logits.dim(logits.ndim() - 1);
            Tensor grad(logits.shape(), Device::CUDA, grad_dtype);
            if (grad.is_bf16()) {
                cuda::cross_entropy_backward(logits.data(), targets.data(), grad.bf16_data(),
                                             logits.numel() / V, V);
            } else {
                cuda::cross_entropy_backward(logits.data(), targets.data(), grad.data(),
                                             logits.numel() / V, V);
            }
            return grad;
        }
    #endif

    if (grad_dtype != DType::F32) throw std::invalid_argument("cross_entropy_backward: bf16 grad is CUDA-only");

    Tensor grad(logits.shape());

    for (size_t i = 0; i < num_predictions; ++i) {
        const size_t base = i * vocab_size;

        float max_logit = logits[base];
        for (size_t j = 1; j < vocab_size; ++j) {
            max_logit = std::max(max_logit, logits[base + j]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            grad[base + j] = std::exp(logits[base + j] - max_logit);
            sum_exp += grad[base + j];
        }

        for (size_t j = 0; j < vocab_size; ++j) {
            grad[base + j] = grad[base + j] / sum_exp * inv_n;
        }
        grad[base + static_cast<size_t>(targets[i])] -= inv_n;
    }

    return grad;
}
