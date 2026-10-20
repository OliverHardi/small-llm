#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cstdlib>
// cuBLAS is column-major; our tensors are row-major
// we ask cuBLAS for the column-major product C^T = B^T @ A^T:
// pass B first, then A, and swap the dimensions. No data is ever transposed.

namespace cuda {

namespace {
const float kOne = 1.0f;
const float kZero = 0.0f;

bool g_bf16 = std::getenv("BF16_MATMUL") != nullptr;

void gemm_ex(cublasOperation_t op_b, cublasOperation_t op_a,
             const uint16_t* a, int lda, const uint16_t* b, int ldb,
             void* c, bool c_bf16, bool accumulate,
             size_t M, size_t K, size_t N) {
    if (M == 0 || N == 0) return;
    if (accumulate && c_bf16) throw std::invalid_argument("gemm_ex: accumulate needs an fp32 output");
    const float* beta = accumulate ? &kOne : &kZero;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(), op_b, op_a, (int)N, (int)M, (int)K,
                              &kOne, b, CUDA_R_16BF, ldb, a, CUDA_R_16BF, lda,
                              beta, c, c_bf16 ? CUDA_R_16BF : CUDA_R_32F, (int)N,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

struct Bf16Copy {
    uint16_t* p;
    Bf16Copy(const float* src, size_t n) : p(static_cast<uint16_t*>(device_alloc(n * sizeof(uint16_t)))) {
        cast_f32_to_bf16(src, p, n);
    }
    ~Bf16Copy() { device_free(p); }
    Bf16Copy(const Bf16Copy&) = delete;
    Bf16Copy& operator=(const Bf16Copy&) = delete;
};

void gemm_bf16(cublasOperation_t op_b, cublasOperation_t op_a,
               const float* a, int lda, const float* b, int ldb, float* c,
               size_t M, size_t K, size_t N) {
    Bf16Copy a16(a, M * K), b16(b, K * N);
    gemm_ex(op_b, op_a, a16.p, lda, b16.p, ldb, c, false, false, M, K, N);
}

} // namespace

void set_bf16_matmul(bool on) { g_bf16 = on; }


// c[M,N] = a[M,K] @ b[K,N]
void matmul_ex(const uint16_t* a, const uint16_t* b, void* c, bool c_bf16, bool accumulate,
               size_t M, size_t K, size_t N) {
    gemm_ex(CUBLAS_OP_N, CUBLAS_OP_N, a, (int)K, b, (int)N, c, c_bf16, accumulate, M, K, N);
}
// c[M,N] = a[M,K] @ b[N,K]^T
void matmul_bt_ex(const uint16_t* a, const uint16_t* b, void* c, bool c_bf16, bool accumulate,
                  size_t M, size_t K, size_t N) {
    gemm_ex(CUBLAS_OP_T, CUBLAS_OP_N, a, (int)K, b, (int)K, c, c_bf16, accumulate, M, K, N);
}
// c[M,N] = a[K,M]^T @ b[K,N]
void matmul_at_ex(const uint16_t* a, const uint16_t* b, void* c, bool c_bf16, bool accumulate,
                  size_t M, size_t K, size_t N) {
    gemm_ex(CUBLAS_OP_N, CUBLAS_OP_T, a, (int)M, b, (int)N, c, c_bf16, accumulate, M, K, N);
}

// c[M,N] = a[M,K] @ b[K,N]
void matmul(const float* a, const float* b, float* c, size_t M, size_t K, size_t N) {
    if (M == 0 || N == 0) return;
    if (g_bf16) { gemm_bf16(CUBLAS_OP_N, CUBLAS_OP_N, a, (int)K, b, (int)N, c, M, K, N); return; }
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)N, (int)M, (int)K,
                             &kOne, b, (int)N, a, (int)K,
                             &kZero, c, (int)N));
}

// c[M,N] = a[M,K] @ b[N,K]^T
void matmul_bt(const float* a, const float* b, float* c, size_t M, size_t K, size_t N) {
    if (M == 0 || N == 0) return;
    if (g_bf16) { gemm_bf16(CUBLAS_OP_T, CUBLAS_OP_N, a, (int)K, b, (int)K, c, M, K, N); return; }
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                             (int)N, (int)M, (int)K,
                             &kOne, b, (int)K, a, (int)K,
                             &kZero, c, (int)N));
}

// c[M,N] = a[K,M]^T @ b[K,N]
void matmul_at(const float* a, const float* b, float* c, size_t M, size_t K, size_t N) {
    if (M == 0 || N == 0) return;
    if (g_bf16) { gemm_bf16(CUBLAS_OP_N, CUBLAS_OP_T, a, (int)M, b, (int)N, c, M, K, N); return; }
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
