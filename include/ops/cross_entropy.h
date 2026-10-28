#pragma once

#include "tensor.h"

namespace ops {

// does softmax then cross entropy using numerically stable log -> sum -> exp method
float cross_entropy(const Tensor& logits, const Tensor& targets, size_t num_valid = 0);

Tensor cross_entropy_backward(const Tensor& logits, const Tensor& targets,
                              DType grad_dtype = DType::F32, size_t num_valid = 0);

}