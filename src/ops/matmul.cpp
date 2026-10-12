#include "ops/matmul.h"

#include <cmath>
#include <stdexcept>
#include <vector>

Tensor ops::matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error(
            "matmul currently only supports 2D tensors"
        );
    }
    
    const size_t M = a.dim(0);
    const size_t K = a.dim(1);

    const size_t N = b.dim(1);
    
    Tensor result({M, N});

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Tensor ops::batched_matmul(const Tensor& a, const Tensor& b) {
    const size_t ndim = a.ndim();

    // Check batch dimensions.
    for (size_t i = 0; i < ndim - 2; ++i) {
        if (a.dim(i) != b.dim(i)) {
            throw std::invalid_argument(
                "batched_matmul: batch dimensions must match"
            );
        }
    }

    const size_t M = a.dim(ndim - 2);
    const size_t K = a.dim(ndim - 1);

    const size_t K_b = b.dim(ndim - 2);
    const size_t N = b.dim(ndim - 1);

    if (K != K_b) {
        throw std::invalid_argument(
            "batched_matmul: inner dimensions must match"
        );
    }

    // Result shape: [..., M, N]
    std::vector<size_t> result_shape;

    for (size_t i = 0; i < ndim - 2; ++i) {
        result_shape.push_back(a.dim(i));
    }

    result_shape.push_back(M);
    result_shape.push_back(N);

    Tensor result(result_shape);

    // Number of independent matrix multiplications.
    size_t num_batches = 1;

    for (size_t i = 0; i < ndim - 2; ++i) {
        num_batches *= a.dim(i);
    }


    for (size_t batch = 0; batch < num_batches; ++batch) {

        size_t a_base = batch * M * K;
        size_t b_base = batch * K * N;
        size_t c_base = batch * M * N;

        for (size_t i = 0; i < M; ++i) {

            for (size_t j = 0; j < N; ++j) {

                float sum = 0.0f;

                for (size_t k = 0; k < K; ++k) {

                    sum +=
                        a[a_base + i * K + k] *
                        b[b_base + k * N + j];
                }

                result[c_base + i * N + j] = sum;
            }
        }
    }

    return result;

}

// a[M,K] @ b[N,K]^T -> [M,N]. Both inner loops walk contiguous memory.
Tensor ops::matmul_bt(const Tensor& a, const Tensor& b) {
    const size_t M = a.dim(0), K = a.dim(1), N = b.dim(0);

    Tensor c({M, N});
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) sum += a[i * K + k] * b[j * K + k];
            c[i * N + j] = sum;
        }
    }
    return c;
}

// a[K,M]^T @ b[K,N] -> [M,N]. k outermost so b and c are read/written row-contiguously.
Tensor ops::matmul_at(const Tensor& a, const Tensor& b) {
    const size_t K = a.dim(0), M = a.dim(1), N = b.dim(1);

    Tensor c({M, N});   // zero-initialized
    for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < M; ++i) {
            const float a_ki = a[k * M + i];
            for (size_t j = 0; j < N; ++j) c[i * N + j] += a_ki * b[k * N + j];
        }
    }
    return c;
}