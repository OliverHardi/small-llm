#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

// cuBLAS is column-major; our tensors are row-major
// we ask cuBLAS for the column-major product C^T = B^T @ A^T:
// pass B first, then A, and swap the dimensions. No data is ever transposed.

// TODO: switch to cublasSgemmEx

namespace cuda {

namespace {
const float kOne = 1.0f;
const float kZero = 0.0f;
}

// c[M,N] = a[M,K] @ b[K,N]
void matmul(const float* a, const float* b, float* c, size_t M, size_t K, size_t N) {
    if (M == 0 || N == 0) return;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)N, (int)M, (int)K,
                             &kOne, b, (int)N, a, (int)K,
                             &kZero, c, (int)N));
}

// c[M,N] = a[M,K] @ b[N,K]^T
void matmul_bt(const float* a, const float* b, float* c, size_t M, size_t K, size_t N) {
    if (M == 0 || N == 0) return;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                             (int)N, (int)M, (int)K,
                             &kOne, b, (int)K, a, (int)K,
                             &kZero, c, (int)N));
}

// c[M,N] = a[K,M]^T @ b[K,N]
void matmul_at(const float* a, const float* b, float* c, size_t M, size_t K, size_t N) {
    if (M == 0 || N == 0) return;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                             (int)N, (int)M, (int)K,
                             &kOne, b, (int)N, a, (int)M,
                             &kZero, c, (int)N));
}

// fixed stride between matrices in the batch
void batched_matmul(const float* a, const float* b, float* c,
                    size_t batch, size_t M, size_t K, size_t N) {
    if (batch == 0 || M == 0 || N == 0) return;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &kOne, b, (int)N, (long long)(K * N),
               a, (int)K, (long long)(M * K),
        &kZero, c, (int)N, (long long)(M * N),
        (int)batch));
}

} // namespace cuda
