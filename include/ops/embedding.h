#pragma once

#include "tensor.h"

namespace ops {

Tensor embedding(const std::vector<uint32_t>& token_ids, const Tensor& weights);

void embedding_backward(const std::vector<uint32_t>& token_ids,
                        const Tensor& grad_output,   // [N, D]
                        Tensor& weight_grad);

}