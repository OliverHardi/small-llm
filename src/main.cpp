#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <filesystem>

#include "config.h"

#include "data/dataloader.h"
#include "data/dataset.h"
#include "data/parquet_reader.h"
#include "tokenizer.h"
#include "data/mix.h"

#include "model.h"
#include "optim/adamw.h"
#include "ops/cross_entropy.h"
#include "scheduler.h"
#include "inference.h"
#include "checkpoint.h"

#include "data/sft.h"

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define TRAIN

static void runChat(Tokenizer& tokenizer, const ModelConfig& modelConfig) {
    const std::string checkpoint = "checkpoints/sft_model.bin";   // the SFT checkpoint to use

    #ifdef USE_CUDA
    const Device device = Device::CUDA;
    #else
    const Device device = Device::CPU;
    #endif

    Model model(modelConfig);
    model.to(device);
    #ifdef USE_CUDA
    model.enable_bf16(true);
    #endif
    load_checkpoint_weights(checkpoint, model);
    model.refresh_bf16_weights();

    Inference inference(model, tokenizer);
    SamplingOptions options;
    options.temperature = 0.7f;
    std::string line;
    std::cout << "> " << std::flush;
    while (std::getline(std::cin, line)) {
        if (!line.empty()) inference.generateChat(line, 150, options);
        std::cout << "> " << std::flush;
    }
}

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

    runChat(tokenizer, modelConfig); return 0;

    Dataset dataset({
        "tokenized/mix/mixed-00000.bin",
        "tokenized/mix/mixed-00001.bin",
        "tokenized/mix/mixed-00002.bin",
        "tokenized/mix/mixed-00003.bin",
        "tokenized/mix/mixed-00004.bin",
        "tokenized/mix/mixed-00005.bin",
        "tokenized/mix/mixed-00006.bin",
        "tokenized/mix/mixed-00007.bin",
        "tokenized/mix/mixed-00008.bin",
        "tokenized/mix/mixed-00009.bin",
        "tokenized/mix/mixed-00010.bin",
        "tokenized/mix/mixed-00011.bin",
        "tokenized/mix/mixed-00012.bin",
        "tokenized/mix/mixed-00013.bin",
        "tokenized/mix/mixed-00014.bin",
        "tokenized/mix/mixed-00015.bin",
        "tokenized/mix/mixed-00016.bin",
        "tokenized/mix/mixed-00017.bin",
        "tokenized/mix/mixed-00018.bin",
        "tokenized/mix/mixed-00019.bin",
        "tokenized/mix/mixed-00020.bin",
        "tokenized/mix/mixed-00021.bin",
        "tokenized/mix/mixed-00022.bin",
        "tokenized/mix/mixed-00023.bin",
        "tokenized/mix/mixed-00024.bin",
        "tokenized/mix/mixed-00025.bin",
        "tokenized/mix/mixed-00026.bin",
        "tokenized/mix/mixed-00027.bin",
        "tokenized/mix/mixed-00028.bin",
        "tokenized/mix/mixed-00029.bin",
        "tokenized/mix/mixed-00030.bin"
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
        0.03f,                          // 3% linear warmup
        total_steps                     // total steps (training tokens / (context length * batch size))
    );

    // checkpoint training
    TrainState state = load_checkpoint("checkpoints_2/checkpoint_step_59508.bin", model, optimizer);
    model.refresh_bf16_weights();
    start_step = state.step;
    dataloader.setPosition(state.data_position);

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
            std::string checkpoint_path = "checkpoints_2/checkpoint_step_" + std::to_string(step) + ".bin";
            save_checkpoint(checkpoint_path, model, optimizer, TrainState{step + 1, dataloader.position()});
        }
    }

    return 0;
}
