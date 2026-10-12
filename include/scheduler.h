#pragma once

#include <cstddef>

class Scheduler {
public:

    Scheduler(float peak, float warmup, size_t num_steps)
        : peak_(peak),
          warmup_(warmup),
          num_steps_(num_steps)
    {
        min_ = 0.1f * peak_;
    }

    float lr(size_t step) {
        size_t warmupSteps = num_steps_ * warmup_;

        if (step < warmupSteps) {
            return peak_ * static_cast<float>(step + 1) / static_cast<float>(warmupSteps);
        }
        float inside = static_cast<float>(step - warmupSteps) / static_cast<float>(num_steps_ - warmupSteps);
        constexpr float pi = 3.14159265358979f;
        return min_ + (peak_ - min_) * 0.5f * (1.0f + std::cos(pi * inside));
    }

private:
    float peak_;
    float min_;
    float warmup_;
    size_t num_steps_;
};