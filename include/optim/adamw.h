#pragma once

#include <vector>
#include <iosfwd>

#include "model.h"
#include "tensor.h"

float clip_grad_norm(const std::vector<Param>& params, float max_norm, float grad_scale);

class AdamW {
public:
    AdamW(std::vector<Param> params,
        float beta1 = 0.9f,
        float beta2 = 0.95f,
        float eps = 1e-8f,
        float weight_decay = 0.1f
    );

    void step(float lr);

    void save(std::ostream& out) const;
    void load(std::istream& in);

private:
    std::vector<Param> params_;
    std::vector<Tensor> m_;   // first moment, one per param
    std::vector<Tensor> v_;   // second moment, one per param

    float beta1_, beta2_, eps_, weight_decay_;
    size_t step_count_ = 0;
};