#pragma once

#include "tensor.h"

namespace ops {

// x is fp32
// out_dtype BF16 makes the output bf16
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-6f, DType out_dtype = DType::F32);

// x and weight fp32. grad_output fp32 or bf16. Returns dx in fp32
Tensor rmsnorm_backward(const Tensor& x, const Tensor& weight,
                        const Tensor& grad_output, Tensor& weight_grad, float eps);

}