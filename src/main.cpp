#include <iostream>

#include "config.h"

#include "data/dataloader.h"
#include "data/dataset.h"
#include "data/parquet_reader.h"

#include "model.h"
#include "optim/adamw.h"
#include "ops/cross_entropy.h"
#include "scheduler.h"
#include "inference.h"
#include "checkpoint.h"

#include "ops/elementwise.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifdef USE_CUDA
int run_cuda_parity_tests();
int run_model_parity_test();
#endif

int main() {

    #ifdef USE_CUDA
    try {
        return run_cuda_parity_tests() + run_model_parity_test();
    } catch (const std::exception& e) {
        std::printf("CUDA test failed: %s\n", e.what());
        return 1;
    }
    #endif

    return 0;

    ModelConfig modelConfig;

    // tiny model for cpu testing
    modelConfig.context_length = 32;
    modelConfig.num_layers = 2;
    modelConfig.d_model = 64;
    modelConfig.num_heads = 4;
    modelConfig.ffn_hidden = 172;

    TrainingConfig trainingConfig;

    trainingConfig.batch_size = 2;

    Tensor::seed(modelConfig.seed);

    Tokenizer tokenizer;
    tokenizer.load("tokenizer/tokenizer.bin");

    Dataset dataset({
        "tokenized/train-00000-of-00012-6fbcb5acda05b3c0.bin",
        "tokenized/train-00001-of-00012-2bb9d088068a84c9.bin",
        "tokenized/train-00002-of-00012-efb6c8de04272068.bin",
        "tokenized/train-00003-of-00012-47006e5a888a9324.bin",
        "tokenized/train-00004-of-00012-a6a94a0207e8e96c.bin",
        "tokenized/train-00005-of-00012-d255c96cd87a0aa7.bin",
        "tokenized/train-00006-of-00012-89040916c30140e6.bin",
        "tokenized/train-00007-of-00012-239b43e016d4ac92.bin",
        "tokenized/train-00008-of-00012-3273ba93936ad8ef.bin",
        "tokenized/train-00009-of-00012-0b640f47936d940a.bin",
        "tokenized/train-00010-of-00012-d266658ccbfa0537.bin",
        "tokenized/train-00011-of-00012-aec474909333c631.bin"
    });

    DataLoader dataloader(dataset, trainingConfig.batch_size, modelConfig.context_length);

    Model model(modelConfig);

    AdamW optimizer(model.parameters());

    size_t start_step = 0;
    size_t total_steps = static_cast<size_t>(
        dataloader.totalTokens() / (modelConfig.context_length * trainingConfig.batch_size)
    );

    Scheduler scheduler(
        trainingConfig.learning_rate,   // max lr
        0.05f,                           // 5% linear warmup
        total_steps                     // total steps (training tokens / (context length * batch size))
    );

    // save_checkpoint("checkpoints/checkpoint_test.bin", model, optimizer, TrainState{0, dataloader.position()});

    // TrainState state = load_checkpoint("checkpoints/checkpoint_test.bin", model, optimizer);
    // dataloader.setPosition(state.data_position);
    // start_step = state.step;
    
    for (size_t step = start_step; step < total_steps; ++step) {
        const float lr = scheduler.lr(step);
        
        model.zero_grad();

        const Batch batch = dataloader.nextBatch();
        Tensor targets = batch.targets();

        Tensor logits = model.forward(batch.inputs(), batch.batch_size, batch.T());
        const float loss = ops::cross_entropy(logits, targets);

        model.backward(ops::cross_entropy_backward(logits, targets));

        const float grad_norm = clip_grad_norm(model.parameters(), 1.0f);
        
        optimizer.step(lr);

        if (step % 25 == 0) {
            std::printf("  step %4zu  loss %.4f  grad_norm %.4f  lr %.12f\n", step, loss, grad_norm, lr);
        }
    }

    return 0;
}
