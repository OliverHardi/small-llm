#pragma once

#include "config.h"
#include "tensor.h"

class SwiGLU {
public:

    explicit SwiGLU(const ModelConfig& config); // takes config as input

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    void enable_bf16(bool on);   // call after the weights are on CUDA
    bool bf16() const { return bf16_; }

    // params
    Tensor& gate_weights();
    Tensor& up_weights();
    Tensor& down_weights();

    // grads
    Tensor& gate_grad();
    Tensor& up_grad();
    Tensor& down_grad();

    // bf16 shadows (empty unless bf16 mode)
    Tensor& gate_weights16() { return gate_w16_; }
    Tensor& up_weights16()   { return up_w16_; }
    Tensor& down_weights16() { return down_w16_; }

private:
    Tensor forward_bf16(const Tensor& x);
    Tensor backward_bf16(const Tensor& grad_output);

    // weights
    Tensor gate_weights_;
    Tensor up_weights_;
    Tensor down_weights_;

    // grads
    Tensor gate_grad_;
    Tensor up_grad_;
    Tensor down_grad_;

    // bf16 shadows
    Tensor gate_w16_;
    Tensor up_w16_;
    Tensor down_w16_;
    bool bf16_ = false;

    // forward cache for backward pass
    Tensor input_;
    Tensor gate_;
    Tensor up_;
    Tensor activated_;
    Tensor hidden_;
};