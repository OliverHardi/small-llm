#include "ops/embedding.h"

#include <cstdint>
#include <vector>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

Tensor ops::embedding(const std::vector<uint32_t>& token_ids, const Tensor& weights) {
    size_t embedding_dim = weights.dim(1);

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

    for (size_t i = 0; i < token_ids.size(); ++i) {
        const uint32_t token_id = token_ids[i];
        for (size_t j = 0; j < embedding_dim; ++j) {
            weight_grad(token_id, j) += grad_output(i, j);
        }
    }
}