#pragma once

#include "tensor.h"

namespace ops {

// does softmax then cross entropy using numerically stable log -> sum -> exp method
float cross_entropy(const Tensor& logits, const Tensor& targets);

Tensor cross_entropy_backward(const Tensor& logits, const Tensor& targets);

}