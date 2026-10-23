#include "nn/swiglu.h"

#include <cmath>
#include <stdexcept>

#include "ops/elementwise.h"
#include "ops/matmul.h"

SwiGLU::SwiGLU(const ModelConfig& config)
    : gate_weights_({config.d_model, config.ffn_hidden}),
      up_weights_({config.d_model, config.ffn_hidden}),
      down_weights_({config.ffn_hidden, config.d_model}),
      gate_grad_({config.d_model, config.ffn_hidden}, 0.0f),
      up_grad_({config.d_model, config.ffn_hidden}, 0.0f),
      down_grad_({config.ffn_hidden, config.d_model}, 0.0f)
{
    const float std_in  = 0.02f;
    const float std_out = 0.02f / std::sqrt(2.0f * static_cast<float>(config.num_layers));

    gate_weights_.randomNormal(0.0f, std_in);
    up_weights_.randomNormal(0.0f, std_in);
    down_weights_.randomNormal(0.0f, std_out);
}

void SwiGLU::enable_bf16(bool on) {
    bf16_ = on;
    if (on) {
        if (!gate_weights_.is_cuda()) throw std::logic_error("SwiGLU::enable_bf16: move the model to CUDA first");
        gate_w16_ = Tensor(gate_weights_.shape(), Device::CUDA, DType::BF16);
        up_w16_   = Tensor(up_weights_.shape(),   Device::CUDA, DType::BF16);
        down_w16_ = Tensor(down_weights_.shape(), Device::CUDA, DType::BF16);
    } else {
        gate_w16_ = up_w16_ = down_w16_ = Tensor();
    }
}

Tensor SwiGLU::forward(const Tensor& x) {
    if (bf16_) return forward_bf16(x);

    const size_t d_model = gate_weights_.dim(0);
 
    if (x.ndim() < 2 || x.dim(x.ndim() - 1) != d_model) {
        throw std::invalid_argument("SwiGLU::forward expects [..., d_model]");
    }

    const size_t rows = x.numel() / d_model;
    input_ = x.reshaped({rows, d_model});          // [N, D]
 
    gate_ = ops::matmul(input_, gate_weights_);    // [N, F]
    up_   = ops::matmul(input_, up_weights_);      // [N, F]

    activated_ = ops::silu(gate_);
    hidden_ = ops::multiply(activated_, up_);
 
    // restore leading dimensions
    return ops::matmul(hidden_, down_weights_).reshaped(x.shape());
}

Tensor SwiGLU::backward(const Tensor& grad_output) {
    if (bf16_) return backward_bf16(grad_output);

    const size_t d_model = gate_weights_.dim(0);

    if (grad_output.numel() != input_.numel() ||
        grad_output.dim(grad_output.ndim() - 1) != d_model) {
        throw std::invalid_argument("SwiGLU::backward: incorrect gradient shape");
    }

    // input_ is stored flattened as [N, D]; flatten the gradient to match
    const size_t rows = input_.dim(0);
    Tensor grad_flat = grad_output.reshaped({rows, d_model});

    // hidden = activated * up  (recomputed, so no extra cache member is needed)
    Tensor hidden = ops::multiply(activated_, up_);

    // output = hidden @ W_down
    down_grad_ = ops::add(down_grad_, ops::matmul_at(hidden, grad_flat));
    Tensor grad_hidden = ops::matmul_bt(grad_flat, down_weights_);

    // hidden = activated * up
    Tensor grad_activated = ops::multiply(grad_hidden, up_);
    Tensor grad_up        = ops::multiply(grad_hidden, activated_);

    // activated = SiLU(gate);  SiLU'(x) = s * (1 + x * (1 - s)),  s = sigmoid(x)
    Tensor grad_gate = ops::silu_backward(gate_, grad_activated);

    // gate = x @ W_gate,  up = x @ W_up
    Tensor input_T = ops::transpose(input_);
    gate_grad_ = ops::add(gate_grad_, ops::matmul(input_T, grad_gate));
    up_grad_   = ops::add(up_grad_,   ops::matmul(input_T, grad_up));

    // sum gradients of each path
    Tensor dx = ops::add(
        ops::matmul_bt(grad_gate, gate_weights_),
        ops::matmul_bt(grad_up, up_weights_)
    );

    return dx.reshaped(grad_output.shape());
}

// ---- bf16 path ---------------------------------------------------------------------
Tensor SwiGLU::forward_bf16(const Tensor& x) {
    const size_t d_model = gate_weights_.dim(0);

    if (x.ndim() < 2 || x.dim(x.ndim() - 1) != d_model || !x.is_bf16()) {
        throw std::invalid_argument("SwiGLU::forward (bf16) expects bf16 [..., d_model]");
    }

    const size_t rows = x.numel() / d_model;
    input_ = x.reshaped({rows, d_model});

    gate_ = ops::matmul(input_, gate_w16_);        // bf16 [N, F]
    up_   = ops::matmul(input_, up_w16_);
    activated_ = ops::silu(gate_);
    hidden_ = ops::multiply(activated_, up_);

    return ops::matmul(hidden_, down_w16_).reshaped(x.shape());
}

Tensor SwiGLU::backward_bf16(const Tensor& grad_output) {
    const size_t d_model = gate_weights_.dim(0);

    if (grad_output.numel() != input_.numel() ||
        grad_output.dim(grad_output.ndim() - 1) != d_model) {
        throw std::invalid_argument("SwiGLU::backward: incorrect gradient shape");
    }

    const size_t rows = input_.dim(0);
    Tensor grad_flat = grad_output.to_dtype(DType::BF16).reshaped({rows, d_model});   // no-op if already bf16

    // output = hidden @ W_down;  weight grads accumulate straight into the fp32 grads
    ops::matmul_at_accumulate(hidden_, grad_flat, down_grad_);
    Tensor grad_hidden = ops::matmul_bt(grad_flat, down_w16_);                       // bf16 [N, F]

    Tensor grad_activated = ops::multiply(grad_hidden, up_);
    Tensor grad_up        = ops::multiply(grad_hidden, activated_);
    Tensor grad_gate      = ops::silu_backward(gate_, grad_activated);

    ops::matmul_at_accumulate(input_, grad_gate, gate_grad_);
    ops::matmul_at_accumulate(input_, grad_up,   up_grad_);

    // dx = grad_gate @ Wg^T + grad_up @ Wu^T, summed in one fp32 buffer
    Tensor dx = ops::matmul_bt(grad_gate, gate_w16_, DType::F32);
    ops::matmul_bt_accumulate(grad_up, up_w16_, dx);

    return dx.reshaped(grad_output.shape());
}

Tensor& SwiGLU::gate_weights() {
    return gate_weights_;
}

Tensor& SwiGLU::up_weights() {
    return up_weights_;
}

Tensor& SwiGLU::down_weights() {
    return down_weights_;
}

Tensor& SwiGLU::gate_grad() {
    return gate_grad_;
}

Tensor& SwiGLU::up_grad() {
    return up_grad_;
}

Tensor& SwiGLU::down_grad() {
    return down_grad_;
}