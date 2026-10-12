#pragma once

#include "tensor.h"

namespace ops {

Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-6f);

Tensor rmsnorm_backward(const Tensor& x, const Tensor& weight,
                        const Tensor& grad_output, Tensor& weight_grad, float eps);

}