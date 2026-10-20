#pragma once

#include "tensor.h"

namespace ops {

Tensor add(const Tensor& a, const Tensor& b);

Tensor multiply(const Tensor& a, const Tensor& b);

Tensor multiply(const Tensor& a, float scalar);

Tensor silu(const Tensor& x);

Tensor silu_backward(const Tensor& input, const Tensor& grad_output);   // d/dx of silu(x) * grad

// i know its not an element wise operation
Tensor transpose(const Tensor& x);

void scale_inplace(Tensor& t, float s);

float sum_squares(const Tensor& t);                                    // for the gradient norm

void adamw_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
                float lr, float beta1, float beta2, float eps,
                float weight_decay, float bias1, float bias2);        // bias1/2 = 1 - beta^step

}