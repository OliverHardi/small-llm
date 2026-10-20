#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <unordered_map>
#include <vector>

namespace {
struct Pool {
    std::unordered_map<size_t, std::vector<void*>> free_blocks;  // rounded size -> cached blocks
    std::unordered_map<void*, size_t> sizes;                     // every block we own -> its size
};
// Intentionally leaked: Tensor destructors can run during static destruction at exit.
Pool& pool() { static Pool* p = new Pool; return *p; }
size_t round_up(size_t n) { return (n + 511) & ~size_t(511); }

void empty_cache() {
    Pool& p = pool();
    for (auto& [size, list] : p.free_blocks) {
        for (void* ptr : list) { cudaFree(ptr); p.sizes.erase(ptr); }
        list.clear();
    }
}
} // namespace

namespace cuda {

void* device_alloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    bytes = round_up(bytes);
    Pool& p = pool();
    auto& list = p.free_blocks[bytes];
    if (!list.empty()) { void* ptr = list.back(); list.pop_back(); return ptr; }

    void* ptr = nullptr;
    cudaError_t e = cudaMalloc(&ptr, bytes);
    if (e == cudaErrorMemoryAllocation) {   // out of memory: drop the cache and retry once
        cudaGetLastError();
        empty_cache();
        e = cudaMalloc(&ptr, bytes);
    }
    CUDA_CHECK(e);
    p.sizes[ptr] = bytes;
    return ptr;
}

void device_free(void* ptr) {
    if (!ptr) return;
    Pool& p = pool();
    p.free_blocks[p.sizes.at(ptr)].push_back(ptr);   // keep it for reuse; no cudaFree, no sync
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
