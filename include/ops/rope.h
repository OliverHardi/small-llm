#pragma once

#include "tensor.h"

namespace ops {

Tensor rope(const Tensor& x, float theta = 10000.0f);

Tensor rope_backward(const Tensor& grad_output, float theta = 10000.0f);

}