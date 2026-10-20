#include "optim/adamw.h"

#include "ops/elementwise.h"

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

#include <cmath>
#include <istream>
#include <ostream>
#include <utility>

float clip_grad_norm(const std::vector<Param>& params, float max_norm, float grad_scale) {
    double sum_sq = 0.0;
    #ifdef USE_CUDA
        if (!params.empty() && params.front().grad->is_cuda()) {
            cuda::sum_squares_reset();
            for (const Param& p : params) cuda::sum_squares_add(p.grad->data(), p.grad->numel());
            sum_sq = cuda::sum_squares_read();
        } else
    #endif
    {
        for (const Param& p : params) sum_sq += ops::sum_squares(*p.grad);
    }

    const float norm = static_cast<float>(std::sqrt(sum_sq)) * grad_scale;
    float scale = grad_scale;
    if (norm > max_norm) scale *= max_norm / (norm + 1e-6f);
    if (scale != 1.0f) {
        for (const Param& p : params) ops::scale_inplace(*p.grad, scale);
    }
    return norm;
}

AdamW::AdamW(std::vector<Param> params, float beta1, float beta2, float eps, float weight_decay)
    : params_(std::move(params)),
      beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
{
    for (const Param& p : params_) {
        const Device d = p.value->device();
        m_.emplace_back(p.value->shape(), d);
        v_.emplace_back(p.value->shape(), d);
        m_.back().zero();
        v_.back().zero();
    }
}

void AdamW::step(float lr) {
    ++step_count_;

    const float bias1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    const float bias2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

    for (size_t k = 0; k < params_.size(); ++k) {
        ops::adamw_step(*params_[k].value, *params_[k].grad, m_[k], v_[k],
                        lr, beta1_, beta2_, eps_,
                        params_[k].decay ? weight_decay_ : 0.0f, bias1, bias2);
    }
}

void AdamW::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&step_count_), sizeof(step_count_));
    for (size_t k = 0; k < params_.size(); ++k) {
        const Tensor m_host = m_[k].to(Device::CPU);
        const Tensor v_host = v_[k].to(Device::CPU);
        out.write(reinterpret_cast<const char*>(m_host.data()), m_host.numel() * sizeof(float));
        out.write(reinterpret_cast<const char*>(v_host.data()), v_host.numel() * sizeof(float));
    }
}

void AdamW::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&step_count_), sizeof(step_count_));
    for (size_t k = 0; k < params_.size(); ++k) {
        Tensor m_host(m_[k].shape()), v_host(v_[k].shape());
        in.read(reinterpret_cast<char*>(m_host.data()), m_host.numel() * sizeof(float));
        in.read(reinterpret_cast<char*>(v_host.data()), v_host.numel() * sizeof(float));
        m_[k] = m_host.to(params_[k].value->device());
        v_[k] = v_host.to(params_[k].value->device());
    }
}