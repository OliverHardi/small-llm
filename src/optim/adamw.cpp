#include "optim/adamw.h"

#include "ops/elementwise.h"

#include <cmath>
#include <istream>
#include <ostream>

float clip_grad_norm(const std::vector<Param>& params, float max_norm) {
    double sum_sq = 0.0;
    for (const Param& p : params) sum_sq += ops::sum_squares(*p.grad);

    const float norm = static_cast<float>(std::sqrt(sum_sq));
    if (norm > max_norm) {
        const float scale = max_norm / (norm + 1e-6f);
        for (const Param& p : params) ops::scale_inplace(*p.grad, scale);
    }
    return norm;
}

AdamW::AdamW(std::vector<Param> params, float beta1, float beta2, float eps, float weight_decay)
    : params_(std::move(params)),
      beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
{
    for (const Param& p : params_) {
        m_.emplace_back(p.value->shape(), 0.0f);
        v_.emplace_back(p.value->shape(), 0.0f);
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
        out.write(reinterpret_cast<const char*>(m_[k].data()), m_[k].numel() * sizeof(float));
        out.write(reinterpret_cast<const char*>(v_[k].data()), v_[k].numel() * sizeof(float));
    }
}

void AdamW::load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&step_count_), sizeof(step_count_));
    for (size_t k = 0; k < params_.size(); ++k) {
        in.read(reinterpret_cast<char*>(m_[k].data()), m_[k].numel() * sizeof(float));
        in.read(reinterpret_cast<char*>(v_[k].data()), v_[k].numel() * sizeof(float));
    }
}