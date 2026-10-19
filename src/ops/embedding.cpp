#include "ops/embedding.h"

#include <cstdint>
#include <vector>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

#ifdef USE_CUDA
namespace {
Tensor ids_to_device(const std::vector<uint32_t>& ids) {
    Tensor f({ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) f[i] = static_cast<float>(ids[i]);
    return f.to(Device::CUDA);
}
}
#endif

Tensor ops::embedding(const std::vector<uint32_t>& token_ids, const Tensor& weights) {
    size_t embedding_dim = weights.dim(1);

    #ifdef USE_CUDA
    if (weights.is_cuda()) {
        const size_t N = token_ids.size();
        Tensor ids_dev = ids_to_device(token_ids);
        Tensor out({N, embedding_dim}, Device::CUDA);
        cuda::embedding(weights.data(), ids_dev.data(), out.data(), N, embedding_dim);
        return out;
    }
    #endif

    Tensor result({token_ids.size(), embedding_dim});

    for (size_t i = 0; i < token_ids.size(); ++i) {
        uint32_t token_id = token_ids[i];

        for (size_t j = 0; j < embedding_dim; ++j) {
            result(i,j) = weights(token_id, j);
        }
    }

    return result;
}

void ops::embedding_backward(const std::vector<uint32_t>& token_ids,
                             const Tensor& grad_output,   // [N, D]
                             Tensor& weight_grad) {       // [vocab, D]
    const size_t embedding_dim = weight_grad.dim(1);

    #ifdef USE_CUDA
    if (weight_grad.is_cuda()) {
        require_same_device(grad_output, weight_grad, "embedding_backward");
        Tensor ids_dev = ids_to_device(token_ids);
        cuda::embedding_backward(ids_dev.data(), grad_output.data(), weight_grad.data(),
                                 token_ids.size(), embedding_dim);
        return;
    }
    #endif

    for (size_t i = 0; i < token_ids.size(); ++i) {
        const uint32_t token_id = token_ids[i];
        for (size_t j = 0; j < embedding_dim; ++j) {
            weight_grad(token_id, j) += grad_output(i, j);
        }
    }
}