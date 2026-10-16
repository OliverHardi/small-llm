#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

namespace cuda {

void* device_alloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

void device_zero(void* ptr, size_t bytes) {
    if (bytes) CUDA_CHECK(cudaMemset(ptr, 0, bytes));
}

void copy_h2d(void* dst_device, const void* src_host, size_t bytes) {
    if (bytes) CUDA_CHECK(cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice));
}

void copy_d2h(void* dst_host, const void* src_device, size_t bytes) {
    if (bytes) CUDA_CHECK(cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost));
}

void copy_d2d(void* dst_device, const void* src_device, size_t bytes) {
    if (bytes) CUDA_CHECK(cudaMemcpy(dst_device, src_device, bytes, cudaMemcpyDeviceToDevice));
}

void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda
