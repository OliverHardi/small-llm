#pragma once

#include <cstdint>
#include <string>
 
#include "model.h"
#include "optim/adamw.h"

struct TrainState {
    uint64_t step = 0;
    uint64_t data_position = 0; 
};

void save_checkpoint(const std::string& path, Model& model, const AdamW& optimizer, const TrainState& state);

TrainState load_checkpoint(const std::string& path, Model& model, AdamW& optimizer);