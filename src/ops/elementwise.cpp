#include "ops/elementwise.h"

#include <cmath>
#include <stdexcept>

Tensor ops::add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors not the same shape");
    }

    Tensor result(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

Tensor ops::subtract(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors not the same shape");
    }

    Tensor result(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

Tensor ops::multiply(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors not the same shape");
    }

    Tensor result(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i] * b[i];
    }
    
    return result;
}

Tensor ops::multiply(const Tensor& a, float scalar) {
    Tensor result(a.shape());

    for (size_t i = 0; i < a.numel(); ++i) {
        result[i] = a[i] * scalar;
    }
    
    return result;
}

// swish activation function
Tensor ops::silu(const Tensor& x) {
    Tensor result(x.shape());

    for (size_t i = 0; i < x.numel(); ++i) {
        float value = x[i];

        float sigmoid = 1.0f / (1.0f + exp(-value));

        result[i] = value * sigmoid;
    }
    
    return result;
}

// d/dx of silu(x) = s * (1 + x * (1 - s)), s = sigmoid(x)
Tensor ops::silu_backward(const Tensor& input, const Tensor& grad_output) {
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

    Tensor result({cols, rows});

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = x(i, j);
        }
    }

    return result;

}

void ops::scale_inplace(Tensor& t, float s) {
    for (size_t i = 0; i < t.numel(); ++i) t[i] *= s;
}

float ops::sum_squares(const Tensor& t) {
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
    for (size_t i = 0; i < param.numel(); ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

        const float m_hat = m[i] / bias1;
        const float v_hat = v[i] / bias2;

        param[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param[i]);
    }
}