#include "ops/matmul.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

#ifdef USE_CUDA
namespace {
using GemmEx = void (*)(const uint16_t*, const uint16_t*, void*, bool, bool, size_t, size_t, size_t);

void check_bf16_pair(const Tensor& a, const Tensor& b, const char* op) {
    require_same_device(a, b, op);
    if (!a.is_cuda() || !a.is_bf16() || !b.is_bf16())
        throw std::invalid_argument(std::string(op) + ": bf16 path needs two bf16 CUDA tensors");
}

Tensor bf16_gemm(GemmEx fn, const char* op, const Tensor& a, const Tensor& b,
                 size_t M, size_t K, size_t N, DType out_dtype) {
    check_bf16_pair(a, b, op);
    Tensor output({M, N}, Device::CUDA, out_dtype);
    void* c = output.is_bf16() ? static_cast<void*>(output.bf16_data()) : static_cast<void*>(output.data());
    fn(a.bf16_data(), b.bf16_data(), c, output.is_bf16(), false, M, K, N);
    return output;
}
} // namespace
#endif

static void require_f32_out(DType out, const char* op) {
    if (out != DType::F32) throw std::invalid_argument(std::string(op) + ": fp32 inputs need out = F32");
}

Tensor ops::matmul(const Tensor& a, const Tensor& b, DType out) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error(
            "matmul currently only supports 2D tensors"
        );
    }
    
    const size_t M = a.dim(0);
    const size_t K = a.dim(1);

    const size_t N = b.dim(1);

    #ifdef USE_CUDA
    if (a.is_bf16() || b.is_bf16())
        return bf16_gemm(&cuda::matmul_ex, "matmul", a, b, M, K, N, out);
    #endif
    require_f32_out(out, "matmul");

    #ifdef USE_CUDA
    if (a.is_cuda()) {
        require_same_device(a, b, "matmul");
        Tensor output({M, N}, Device::CUDA);
        cuda::matmul(a.data(), b.data(), output.data(), M, K, N);
        return output;
    }
    #endif
    
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

Tensor ops::matmul(const Tensor& a, const Tensor& b) {
    return matmul(a, b, a.is_bf16() ? DType::BF16 : DType::F32);
}

Tensor ops::batched_matmul(const Tensor& a, const Tensor& b) {
    const size_t ndim = a.ndim();

    if (ndim < 2 || b.ndim() != ndim) {
        throw std::invalid_argument("batched_matmul: tensors must have the same rank >= 2");
    }

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

    // Number of independent matrix multiplications.
    size_t num_batches = 1;

    for (size_t i = 0; i < ndim - 2; ++i) {
        num_batches *= a.dim(i);
    }

    #ifdef USE_CUDA
    if (a.is_cuda() || b.is_cuda()) {
        require_same_device(a, b, "batched_matmul");
        Tensor output(result_shape, Device::CUDA);
        cuda::batched_matmul(a.data(), b.data(), output.data(), num_batches, M, K, N);
        return output;
    }
    #endif

    Tensor result(result_shape);

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

Tensor ops::matmul_bt(const Tensor& a, const Tensor& b) {
    return matmul_bt(a, b, a.is_bf16() ? DType::BF16 : DType::F32);
}

Tensor ops::matmul_bt(const Tensor& a, const Tensor& b, DType out) {
    const size_t M = a.dim(0), K = a.dim(1), N = b.dim(0);

    #ifdef USE_CUDA
    if (a.is_bf16() || b.is_bf16())
        return bf16_gemm(&cuda::matmul_bt_ex, "matmul_bt", a, b, M, K, N, out);
    #endif
    require_f32_out(out, "matmul_bt");

    #ifdef USE_CUDA
        if (a.is_cuda() || b.is_cuda()) {
            require_same_device(a, b, "matmul_bt");
            Tensor output({M, N}, Device::CUDA);
            cuda::matmul_bt(a.data(), b.data(), output.data(), M, K, N);
            return output;
        }
    #endif

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
    return matmul_at(a, b, a.is_bf16() ? DType::BF16 : DType::F32);
}

Tensor ops::matmul_at(const Tensor& a, const Tensor& b, DType out) {
    const size_t K = a.dim(0), M = a.dim(1), N = b.dim(1);

    #ifdef USE_CUDA
    if (a.is_bf16() || b.is_bf16())
        return bf16_gemm(&cuda::matmul_at_ex, "matmul_at", a, b, M, K, N, out);
    #endif
    require_f32_out(out, "matmul_at");

    #ifdef USE_CUDA
        if (a.is_cuda() || b.is_cuda()) {
            require_same_device(a, b, "matmul_at");
            Tensor output({M, N}, Device::CUDA);
            cuda::matmul_at(a.data(), b.data(), output.data(), M, K, N);
            return output;
        }
    #endif

    Tensor c({M, N});   // zero-initialized
    for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < M; ++i) {
            const float a_ki = a[k * M + i];
            for (size_t j = 0; j < N; ++j) c[i * N + j] += a_ki * b[k * N + j];
        }
    }
    return c;
}

// grad(fp32) += a^T @ b with a, b bf16: cuBLAS beta = 1, no temporary.
void ops::matmul_at_accumulate(const Tensor& a, const Tensor& b, Tensor& grad) {
#ifdef USE_CUDA
    const size_t K = a.dim(0), M = a.dim(1), N = b.dim(1);
    check_bf16_pair(a, b, "matmul_at_accumulate");
    if (grad.dtype() != DType::F32 || !grad.is_cuda() || grad.dim(0) != M || grad.dim(1) != N)
        throw std::invalid_argument("matmul_at_accumulate: grad must be fp32 CUDA [M,N]");
    cuda::matmul_at_ex(a.bf16_data(), b.bf16_data(), grad.data(), false, true, M, K, N);
#else
    (void)a; (void)b; (void)grad;
    throw std::runtime_error("matmul_at_accumulate: built without CUDA support");
#endif
}
