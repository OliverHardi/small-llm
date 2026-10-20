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

#ifdef USE_CUDA
int run_cuda_parity_tests();
int run_model_parity_test();
int run_cudnn_attention_test();
int run_bf16_stage12_test();
#endif

int main() {

    // #ifdef USE_CUDA
    // try {
    //     return run_cuda_parity_tests() + run_model_parity_test() + run_cudnn_attention_test() + run_bf16_stage12_test();
    // } catch (const std::exception& e) {
    //     std::printf("CUDA test failed: %s\n", e.what());
    //     return 1;
    // }
    // return 0;
    // #endif


    ModelConfig modelConfig;
    TrainingConfig trainingConfig;

    // tiny model for cpu testing
    // modelConfig.context_length = 32;
    // modelConfig.num_layers = 2;
    // modelConfig.d_model = 64;
    // modelConfig.num_heads = 4;
    // modelConfig.ffn_hidden = 172;


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

    const size_t accum = trainingConfig.grad_accum_steps;

    // temp timing
    auto sync = [] {
    #ifdef USE_CUDA
        cuda::synchronize();
    #endif
    };
    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };

    const size_t warmup = 3, timed_steps = 20;
    double t_data = 0, t_fwd = 0, t_bwd = 0, t_clip = 0, t_opt = 0, t_total = 0;

    for (size_t step = start_step; step < start_step + warmup + timed_steps; ++step) {
        const float lr = 1e-3f;
        const auto t_step = Clock::now();
        double d_data = 0, d_fwd = 0, d_bwd = 0;     // summed over the micro-batches

        model.zero_grad();
        float loss_sum = 0.0f;
        for (size_t micro = 0; micro < accum; ++micro) {
            auto t = Clock::now();
            const Batch batch = dataloader.nextBatch();
            Tensor targets = batch.targets().to(device);
            sync();                       d_data += ms_since(t);  t = Clock::now();

            Tensor logits = model.forward(batch.inputs(), batch.batch_size, batch.T());
            loss_sum += ops::cross_entropy(logits, targets);
            sync();                       d_fwd += ms_since(t);   t = Clock::now();

            model.backward(ops::cross_entropy_backward(logits, targets));
            sync();                       d_bwd += ms_since(t);
        }
        const float loss = loss_sum / static_cast<float>(accum);

        auto t = Clock::now();
        const float grad_norm = clip_grad_norm(model.parameters(), 1.0f, 1.0f / static_cast<float>(accum));
        sync();                           const double d_clip = ms_since(t);  t = Clock::now();

        optimizer.step(lr);
        sync();                           const double d_opt = ms_since(t);

        if (step >= start_step + warmup) {
            t_data += d_data; t_fwd += d_fwd; t_bwd += d_bwd; t_clip += d_clip; t_opt += d_opt;
            t_total += ms_since(t_step);
        }
        std::printf("  step %4zu  loss %.4f  grad_norm %.4f\n", step, loss, grad_norm);
    }

    const double n = static_cast<double>(timed_steps);
    const double tokens = static_cast<double>(accum) * trainingConfig.batch_size * modelConfig.context_length;
    std::printf("\nper optimizer step (avg of %zu, %zu micro-batches, %.0f tokens): total %.0f ms  (%.0f tokens/s)\n",
                timed_steps, accum, tokens, t_total / n, tokens / (t_total / n / 1000.0));
    std::printf("  data %.1f | forward+loss %.1f | backward %.1f | clip %.1f | adamw %.1f  (ms)\n",
                t_data / n, t_fwd / n, t_bwd / n, t_clip / n, t_opt / n);
    std::printf("  per batch: total %.1f ms\n", t_total / n / static_cast<double>(accum) / static_cast<double>(trainingConfig.batch_size));

    return 0;

    for (size_t step = start_step; step < total_steps; ++step) {
        const float lr = scheduler.lr(step);
        
        model.zero_grad();

        float loss_sum = 0.0f;
        for (size_t micro = 0; micro < accum; ++micro) {
            const Batch batch = dataloader.nextBatch();
            Tensor targets = batch.targets().to(device);

            Tensor logits = model.forward(batch.inputs(), batch.batch_size, batch.T());
            loss_sum += ops::cross_entropy(logits, targets);

            model.backward(ops::cross_entropy_backward(logits, targets));
        }

        const float loss = loss_sum / static_cast<float>(accum);

        const float grad_norm = clip_grad_norm(model.parameters(), 1.0f, 1.0f / static_cast<float>(accum));
        
        optimizer.step(lr);

        if (step % 25 == 0) {
            std::printf("  step %4zu  loss %.4f  grad_norm %.4f  lr %.12f\n", step, loss, grad_norm, lr);
        }
    }

    return 0;
}
