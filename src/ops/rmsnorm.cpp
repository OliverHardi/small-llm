#include "ops/rmsnorm.h"

#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

Tensor ops::rmsnorm(const Tensor& x, const Tensor& weight, float eps, DType out_dtype) {

    #ifdef USE_CUDA
        if (x.is_cuda() || weight.is_cuda()) {
            require_same_device(x, weight, "rmsnorm");
            const size_t D = weight.numel();
            Tensor out(x.shape(), Device::CUDA, out_dtype);
            if (out_dtype == DType::BF16)
                cuda::rmsnorm(x.data(), weight.data(), out.bf16_data(), x.numel() / D, D, eps);
            else
                cuda::rmsnorm(x.data(), weight.data(), out.data(), x.numel() / D, D, eps);
            return out;
        }
    #endif
    if (out_dtype != DType::F32) throw std::invalid_argument("rmsnorm: bf16 output is CUDA-only");

    size_t hidden_size = x.dim(x.ndim() - 1);

    if (weight.ndim() != 1 || weight.dim(0) != hidden_size) {
        throw std::invalid_argument(
            "rmsnorm: weight must be 1D with size equal to the last dimension"
        );
    }

    Tensor result(x.shape());

    size_t groups = x.numel() / hidden_size;

    for (size_t group = 0; group < groups; ++group) {

        size_t base = group * hidden_size;

        // mean of squared values.
        float mean_square = 0.0f;

        for (size_t i = 0; i < hidden_size; ++i) {
            float value = x[base + i];
            mean_square += value * value;
        }

        mean_square /= static_cast<float>(hidden_size);

        // RMS normalization factor
        float inverse_rms = 1.0f / std::sqrt(mean_square + eps);

        // normalize and apply weight
        for (size_t i = 0; i < hidden_size; ++i) {
            result[base + i] =
                x[base + i] * inverse_rms * weight[i];
        }
    }

    return result;
}

// Given dL/dy, returns dL/dx and accumulates dL/dw into weight_grad.
// With r = 1/sqrt(mean(x^2) + eps) and gw_j = w_j * g_j:
//   dL/dw_i += g_i * x_i * r
//   dL/dx_i  = r * gw_i - (r^3 / D) * x_i * sum_j(gw_j * x_j)
Tensor ops::rmsnorm_backward(const Tensor& x, const Tensor& weight,
                             const Tensor& grad_output, Tensor& weight_grad, float eps) {

    #ifdef USE_CUDA
        if (x.is_cuda()) {
            const size_t D = weight.numel();
            Tensor dx(x.shape(), Device::CUDA);   // always fp32
            if (grad_output.is_bf16())
                cuda::rmsnorm_backward(x.data(), weight.data(), grad_output.bf16_data(), dx.data(),
                                       weight_grad.data(), x.numel() / D, D, eps);
            else
                cuda::rmsnorm_backward(x.data(), weight.data(), grad_output.data(), dx.data(),
                                       weight_grad.data(), x.numel() / D, D, eps);
            return dx;
        }
    #endif

    const size_t hidden_size = x.dim(x.ndim() - 1);

    if (weight.ndim() != 1 || weight.dim(0) != hidden_size ||
        grad_output.numel() != x.numel() || weight_grad.numel() != hidden_size) {
        throw std::invalid_argument("rmsnorm_backward: shape mismatch");
    }

    Tensor dx(x.shape());
    const size_t groups = x.numel() / hidden_size;

    for (size_t group = 0; group < groups; ++group) {
        const size_t base = group * hidden_size;

        float mean_square = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) mean_square += x[base + i] * x[base + i];
        mean_square /= static_cast<float>(hidden_size);
        const float inverse_rms = 1.0f / std::sqrt(mean_square + eps);

        float dot = 0.0f;  // sum_j gw_j * x_j
        for (size_t j = 0; j < hidden_size; ++j)
            dot += weight[j] * grad_output[base + j] * x[base + j];

        const float coeff = inverse_rms * inverse_rms * inverse_rms * dot
                            / static_cast<float>(hidden_size);

        for (size_t i = 0; i < hidden_size; ++i) {
            const float g = grad_output[base + i];
            weight_grad[i] += g * x[base + i] * inverse_rms;
            dx[base + i] = inverse_rms * weight[i] * g - coeff * x[base + i];
        }
    }
    return dx;
}