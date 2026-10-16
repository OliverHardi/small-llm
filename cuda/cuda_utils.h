#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
 
#include <cstddef>
#include <stdexcept>
#include <string>

inline void cuda_throw(const std::string& error, const char* file, int line) {
    throw std::runtime_error(error + " (" + file + ":" + std::to_string(line) + ")");
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess)                                                  \
            cuda_throw(std::string(#call) + ": " + cudaGetErrorString(err_),      \
                       __FILE__, __LINE__);                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st_ = (call);                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS)                                         \
            cuda_throw(std::string(#call) + ": cuBLAS status " +                  \
                       std::to_string(static_cast<int>(st_)), __FILE__, __LINE__);\
    } while (0)

#ifdef CUDA_DEBUG_SYNC
#define CUDA_LAUNCH_CHECK() \
    do { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); } while (0)
#else
#define CUDA_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())
#endif

constexpr int kBlockSize = 256;

inline unsigned blocks_for(size_t n, int block = kBlockSize) {
    return static_cast<unsigned>((n + block - 1) / block);
}

// Shared cuBLAS handle, created on first use.
inline cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}