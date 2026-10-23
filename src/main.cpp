#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "config.h"
#include <chrono>

#include "data/dataloader.h"
#include "data/dataset.h"
#include "data/parquet_reader.h"

#include "model.h"
#include "optim/adamw.h"
#include "ops/cross_entropy.h"
#include "scheduler.h"
#include "inference.h"
#include "checkpoint.h"

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

int main() {

    #ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED); // keep the machine awake while this process runs
    SetConsoleOutputCP(CP_UTF8); // set console output to UTF-8
    #endif

    ModelConfig modelConfig;
    TrainingConfig trainingConfig;

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

    #ifdef USE_CUDA
    const Device device = Device::CUDA;
    #else
    const Device device = Device::CPU;
    #endif

    model.to(device);

    #ifdef USE_CUDA
    model.enable_bf16(true);
    #endif

    AdamW optimizer(model.parameters());

    size_t start_step = 0;
    size_t total_steps = static_cast<size_t>(
        dataloader.totalTokens() / (modelConfig.context_length * trainingConfig.batch_size * trainingConfig.grad_accum_steps)
    );

    Scheduler scheduler(
        trainingConfig.learning_rate,   // max lr
        0.05f,                           // 5% linear warmup
        total_steps                     // total steps (training tokens / (context length * batch size))
    );

    // checkpoint training
    // TrainState state = load_checkpoint("checkpoints/checkpoint_step_20000.bin", model, optimizer);
    // model.refresh_bf16_weights();
    // start_step = state.step;
    // dataloader.setPosition(state.data_position);

    Inference inference(model, tokenizer);

    // inference.generate("Paris is the capital of", 20);
    // inference.generate("The capital of Japan is", 20);
    // inference.generate("The largest planet in our solar system is", 20);
    // inference.generate("Water freezes at", 20);
    // inference.generate("The sun is a", 20);
    // inference.generate("The Earth revolves around", 20);
    // inference.generate("William Shakespeare was", 20);
    // inference.generate("The Great Wall of China", 20);

    return 0;

    const size_t accum = trainingConfig.grad_accum_steps;

    for (size_t step = start_step; step < total_steps; ++step) {
        const float lr = scheduler.lr(step);
        
        model.zero_grad();

        float loss_sum = 0.0f;
        for (size_t micro = 0; micro < accum; ++micro) {
            const Batch batch = dataloader.nextBatch();
            Tensor targets = batch.targets().to(device);

            Tensor logits = model.forward(batch.inputs(), batch.batch_size, batch.T());
            loss_sum += ops::cross_entropy(logits, targets);

            model.backward(ops::cross_entropy_backward(logits, targets, model.logits_grad_dtype()));
        }

        const float loss = loss_sum / static_cast<float>(accum);

        const float grad_norm = clip_grad_norm(model.parameters(), 1.0f, 1.0f / static_cast<float>(accum));
        
        optimizer.step(lr);
        model.refresh_bf16_weights();
        if (step % 25 == 0) {
            std::printf("  step %4zu/%zu  loss %.4f  grad_norm %.4f  lr %.6f\n", step, total_steps, loss, grad_norm, lr);
        }
        if (step % 2000 == 0 || step == total_steps - 1) {
            std::string checkpoint_path = "checkpoints/checkpoint_step_" + std::to_string(step) + ".bin";
            save_checkpoint(checkpoint_path, model, optimizer, TrainState{step, dataloader.position()});
        }
    }

    return 0;
}
