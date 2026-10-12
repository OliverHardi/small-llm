#pragma once

#include "tensor.h"

namespace ops {

Tensor softmax(const Tensor& x, int axis, bool causal = false);

Tensor softmax_backward(const Tensor& probs, const Tensor& grad_output);    // row-wise, over the last dim

std::vector<uint32_t> argmax(const Tensor& x);  // one index per row of the last dim

}