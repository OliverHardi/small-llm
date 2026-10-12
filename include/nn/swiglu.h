#pragma once

#include "config.h"
#include "tensor.h"

class SwiGLU {
public:

    explicit SwiGLU(const ModelConfig& config); // takes config as input

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    // params
    Tensor& gate_weights();
    Tensor& up_weights();
    Tensor& down_weights();

    // grads
    Tensor& gate_grad();
    Tensor& up_grad();
    Tensor& down_grad();

private:

    // weights
    Tensor gate_weights_;
    Tensor up_weights_;
    Tensor down_weights_;

    // grads
    Tensor gate_grad_;
    Tensor up_grad_;
    Tensor down_grad_;

    // forward cache for backward pass
    Tensor input_;
    Tensor gate_;
    Tensor up_;
    Tensor activated_;
    Tensor hidden_;
};