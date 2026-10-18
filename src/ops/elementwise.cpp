#include "ops/elementwise.h"

#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

Tensor ops::add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors not the same shape");
    }

    #ifdef USE_CUDA
    if (a.is_cuda() || b.is_cuda()) {
        require_same_device(a, b, "add");
        Tensor out(a.shape(), Device::CUDA);
        cuda::add(a.data(), b.data(), out.data(), a.numel());
        return out;
    }
    #endif

    Tensor result(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

Tensor ops::multiply(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors not the same shape");
    }

    #ifdef USE_CUDA
    if (a.is_cuda() || b.is_cuda()) {
        require_same_device(a, b, "multiply");
        Tensor out(a.shape(), Device::CUDA);
        cuda::multiply(a.data(), b.data(), out.data(), a.numel());
        return out;
    }
    #endif

    Tensor result(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i] * b[i];
    }
    
    return result;
}

Tensor ops::multiply(const Tensor& a, float scalar) {

    #ifdef USE_CUDA
    if (a.is_cuda()) {
        Tensor out(a.shape(), Device::CUDA);
        cuda::multiply_scalar(a.data(), scalar, out.data(), a.numel());
        return out;
    }
    #endif

    Tensor result(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i] * scalar;
    }
    
    return result;
}

// swish activation function
Tensor ops::silu(const Tensor& x) {

    #ifdef USE_CUDA
    if (x.is_cuda()) {
        Tensor out(x.shape(), Device::CUDA);
        cuda::silu(x.data(), out.data(), x.numel());
        return out;
    }
    #endif

    Tensor result(x.shape());

    for (size_t i = 0; i < x.numel(); ++i) {
        float value = x[i];

        float sigmoid = 1.0f / (1.0f + std::exp(-value));

        result[i] = value * sigmoid;
    }
    
    return result;
}

// d/dx of silu(x) = s * (1 + x * (1 - s)), s = sigmoid(x)
Tensor ops::silu_backward(const Tensor& input, const Tensor& grad_output) {
    
    #ifdef USE_CUDA
        if (input.is_cuda()) {
        require_same_device(input, grad_output, "silu_backward");
        Tensor out(input.shape(), Device::CUDA);
        cuda::silu_backward(input.data(), grad_output.data(), out.data(), input.numel());
        return out;
    }
    #endif

    Tensor result(input.shape());

    for (size_t i = 0; i < input.numel(); ++i) {
        const float x = input[i];
        const float s = 1.0f / (1.0f + std::exp(-x));
        result[i] = grad_output[i] * s * (1.0f + x * (1.0f - s));
    }
    return result;
}

Tensor ops::transpose(const Tensor& x) {

    

    auto shape = x.shape();

    if (shape.size() != 2) {
        throw std::invalid_argument("can't transpose higher dimensional tensors");
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    #ifdef USE_CUDA
    if (x.is_cuda()) {
        Tensor out({cols, rows}, Device::CUDA);
        cuda::transpose(x.data(), out.data(), rows, cols);
        return out;
    }
    #endif

    Tensor result({cols, rows});

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = x(i, j);
        }
    }

    return result;

}

void ops::scale_inplace(Tensor& t, float s) {
    #ifdef USE_CUDA
    if (t.is_cuda()) { cuda::scale_inplace(t.data(), s, t.numel()); return; }
    #endif
    for (size_t i = 0; i < t.numel(); ++i) t[i] *= s;
}

float ops::sum_squares(const Tensor& t) {
    #ifdef USE_CUDA
    if (t.is_cuda()) return cuda::sum_squares(t.data(), t.numel());
    #endif
    double total = 0.0;
    for (size_t i = 0; i < t.numel(); ++i) {
        total += static_cast<double>(t[i]) * t[i];
    }
    return static_cast<float>(total);
}

// bias1 = 1 - beta1^t, bias2 = 1 - beta2^t.
void ops::adamw_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, float bias1, float bias2) {

    #ifdef USE_CUDA
    if (param.is_cuda()) {
        cuda::adamw_step(param.data(), grad.data(), m.data(), v.data(), param.numel(),
                        lr, beta1, beta2, eps, weight_decay, bias1, bias2);
        return;
    }
    #endif

    for (size_t i = 0; i < param.numel(); ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

        const float m_hat = m[i] / bias1;
        const float v_hat = v[i] / bias2;

        param[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param[i]);
    }
}