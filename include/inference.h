#pragma once

#include "model.h"
#include "tokenizer.h"

#include <string>

class Inference {
public:
    Inference(Model& model, Tokenizer& tokenizer);

    std::string generate(
        const std::string& prompt,
        size_t max_new_tokens
    );

private:
    Model& model_;
    Tokenizer& tokenizer_;
};