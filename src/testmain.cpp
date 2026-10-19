// CPU-vs-GPU parity tests
#ifdef USE_CUDA

#include "tensor.h"
#include "ops/attention.h"
#include "ops/cross_entropy.h"
#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/matmul.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/softmax.h"

#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

// max |gpu - cpu| / (max |cpu| + tiny): scale-independent, so one tolerance fits everything.
float rel_error(const Tensor& cpu, const Tensor& gpu_on_gpu) {
    Tensor gpu = gpu_on_gpu.to(Device::CPU);
    if (cpu.shape() != gpu.shape()) return 1e9f;
    float max_diff = 0.0f, max_ref = 0.0f;
    for (size_t i = 0; i < cpu.numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(cpu[i] - gpu[i]));
        max_ref  = std::max(max_ref, std::fabs(cpu[i]));
    }
    return max_diff / (max_ref + 1e-8f);
}

void report(const char* name, float err, float tol = 1e-5f) {
    const bool ok = err <= tol;
    if (!ok) ++failures;
    std::printf("  %-28s rel err %.3g  %s\n", name, err, ok ? "ok" : "FAIL");
}

Tensor randn(const std::vector<size_t>& shape, uint32_t seed, float stddev = 1.0f) {
    Tensor t(shape);
    t.randomNormal(0.0f, stddev, seed);
    return t;
}

// ids in [0, vocab), stored as floats like the real batches
Tensor random_ids(size_t n, size_t vocab, uint32_t seed) {
    Tensor t({n});
    std::mt19937 gen(seed);
    for (size_t i = 0; i < n; ++i) t[i] = static_cast<float>(gen() % vocab);
    return t;
}

Tensor G(const Tensor& t) { return t.to(Device::CUDA); }

} // namespace

int run_cuda_parity_tests() {

    std::printf("Running CPU-vs-GPU parity tests for all ops with CUDA branches...\n\n");

    failures = 0;
    constexpr float kEps = 1e-5f;

    std::printf("matmul (non-square on purpose)\n");
    {
        // a[7,13] @ b[13,5]
        Tensor a = randn({7, 13}, 1), b = randn({13, 5}, 2);
        report("matmul", rel_error(ops::matmul(a, b), ops::matmul(G(a), G(b))));

        // a[7,13] @ b[5,13]^T
        Tensor bt = randn({5, 13}, 3);
        report("matmul_bt", rel_error(ops::matmul_bt(a, bt), ops::matmul_bt(G(a), G(bt))));

        // a[13,7]^T @ b[13,5]
        Tensor at = randn({13, 7}, 4), b2 = randn({13, 5}, 5);
        report("matmul_at", rel_error(ops::matmul_at(at, b2), ops::matmul_at(G(at), G(b2))));

        // batched: [2,3,7,13] @ [2,3,13,5]
        Tensor ba = randn({2, 3, 7, 13}, 6), bb = randn({2, 3, 13, 5}, 7);
        report("batched_matmul", rel_error(ops::batched_matmul(ba, bb), ops::batched_matmul(G(ba), G(bb))));
    }

    std::printf("rmsnorm\n");
    {
        // D = 70 is deliberately not a multiple of 32 or 256
        Tensor x = randn({5, 70}, 10), w = randn({70}, 11), g = randn({5, 70}, 12);
        report("rmsnorm forward", rel_error(ops::rmsnorm(x, w, kEps), ops::rmsnorm(G(x), G(w), kEps)));

        Tensor wg_cpu({70}), wg_gpu = G(Tensor({70}));
        Tensor dx_cpu = ops::rmsnorm_backward(x, w, g, wg_cpu, kEps);
        Tensor dx_gpu = ops::rmsnorm_backward(G(x), G(w), G(g), wg_gpu, kEps);
        report("rmsnorm_backward dx", rel_error(dx_cpu, dx_gpu));
        report("rmsnorm_backward dw", rel_error(wg_cpu, wg_gpu), 1e-4f);   // atomics: order differs
    }

    std::printf("softmax\n");
    {
        Tensor x = randn({2, 3, 9, 9}, 20, 2.0f), g = randn({2, 3, 9, 9}, 21);
        report("softmax (full)",   rel_error(ops::softmax(x, -1, false), ops::softmax(G(x), -1, false)));
        Tensor p_cpu = ops::softmax(x, -1, true);
        Tensor p_gpu = ops::softmax(G(x), -1, true);
        report("softmax (causal)", rel_error(p_cpu, p_gpu));
        report("softmax_backward", rel_error(ops::softmax_backward(p_cpu, g), ops::softmax_backward(p_gpu, G(g))));

        // long rows: more columns than threads in a block
        Tensor big = randn({4, 1000}, 22);
        report("softmax (1000 cols)", rel_error(ops::softmax(big, -1, false), ops::softmax(G(big), -1, false)));
    }

    std::printf("cross entropy\n");
    {
        Tensor logits = randn({6, 50}, 30, 3.0f);
        Tensor targets = random_ids(6, 50, 31);
        const float loss_cpu = ops::cross_entropy(logits, targets);
        const float loss_gpu = ops::cross_entropy(G(logits), G(targets));
        report("cross_entropy loss", std::fabs(loss_cpu - loss_gpu) / std::fabs(loss_cpu));
        report("cross_entropy_backward",
               rel_error(ops::cross_entropy_backward(logits, targets),
                         ops::cross_entropy_backward(G(logits), G(targets))));
    }

    std::printf("embedding\n");
    {
        const size_t vocab = 40, d = 12, n = 25;
        Tensor table = randn({vocab, d}, 40);

        std::vector<uint32_t> ids(n);
        std::mt19937 gen(41);
        for (auto& id : ids) id = gen() % vocab;     // repeated ids on purpose

        report("embedding", rel_error(ops::embedding(ids, table), ops::embedding(ids, G(table))));

        Tensor grad = randn({n, d}, 42);
        Tensor tg_cpu({vocab, d}), tg_gpu = G(Tensor({vocab, d}));
        ops::embedding_backward(ids, grad, tg_cpu);
        ops::embedding_backward(ids, G(grad), tg_gpu);
        report("embedding_backward", rel_error(tg_cpu, tg_gpu), 1e-4f);   // atomics
    }

    std::printf("rope\n");
    {
        Tensor x = randn({2, 3, 17, 8}, 50);
        report("rope", rel_error(ops::rope(x, 10000.0f), ops::rope(G(x), 10000.0f)), 1e-4f);
        report("rope_backward", rel_error(ops::rope_backward(x, 10000.0f), ops::rope_backward(G(x), 10000.0f)), 1e-4f);
    }

    std::printf("heads\n");
    {
        Tensor x = randn({2, 5, 12}, 60);
        report("split_heads", rel_error(ops::split_heads(x, 3), ops::split_heads(G(x), 3)));
        Tensor h = randn({2, 3, 5, 4}, 61);
        report("merge_heads", rel_error(ops::merge_heads(h), ops::merge_heads(G(h))));
    }

        std::printf("attention (unfused)\n");
    {
        Tensor q = randn({2, 3, 9, 8}, 70, 0.3f), k = randn({2, 3, 9, 8}, 71, 0.3f),
               v = randn({2, 3, 9, 8}, 72), dout = randn({2, 3, 9, 8}, 73);

        auto cpu = ops::attention(q, k, v, true);
        auto gpu = ops::attention(G(q), G(k), G(v), true);
        report("attention forward", rel_error(cpu.out, gpu.out));

        auto gc = ops::attention_backward(q, k, v, cpu.out, cpu.stats, dout, true);
        auto gg = ops::attention_backward(G(q), G(k), G(v), gpu.out, gpu.stats, G(dout), true);
        report("attention dq", rel_error(gc.dq, gg.dq), 1e-4f);
        report("attention dk", rel_error(gc.dk, gg.dk), 1e-4f);
        report("attention dv", rel_error(gc.dv, gg.dv), 1e-4f);
    }

    std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures;
}
#define STEP(msg) do { std::printf("  [%s]\n", msg); std::fflush(stdout); } while (0)

int run_model_parity_test() {
    failures = 0;

    ModelConfig cfg;
    cfg.vocab_size = 50;  cfg.d_model = 32;  cfg.num_heads = 4;  cfg.num_layers = 2;
    cfg.context_length = 16;  cfg.ffn_hidden = 64;     // + whatever else your config needs

    const size_t B = 2, T = 8;
    std::vector<uint32_t> ids(B * T);
    Tensor targets({B * T});
    std::mt19937 gen(7);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = gen() % cfg.vocab_size;
        targets[i] = static_cast<float>(gen() % cfg.vocab_size);
    }

    Model cpu(cfg), gpu(cfg);
    {   // give the GPU model the CPU model's exact weights
        auto pc = cpu.parameters();
        auto pg = gpu.parameters();
        for (size_t k = 0; k < pc.size(); ++k) *pg[k].value = pc[k].value->clone();
    }
    gpu.to(Device::CUDA);

    std::printf("whole model\n");

    STEP("cpu forward");
    Tensor logits_c = cpu.forward(ids, B, T);

    STEP("gpu forward");
    Tensor logits_g = gpu.forward(ids, B, T);

    STEP("compare logits");
    report("logits", rel_error(logits_c, logits_g), 1e-4f);

    STEP("loss");
    const float loss_c = ops::cross_entropy(logits_c, targets);
    const float loss_g = ops::cross_entropy(logits_g, G(targets));
    report("loss", std::fabs(loss_c - loss_g) / std::fabs(loss_c), 1e-5f);

    STEP("zero_grad");
    cpu.zero_grad();
    gpu.zero_grad();

    STEP("cpu backward");
    cpu.backward(ops::cross_entropy_backward(logits_c, targets));

    STEP("gpu backward");
    gpu.backward(ops::cross_entropy_backward(logits_g, G(targets)));

    STEP("compare grads");
    auto pc = cpu.parameters();
    auto pg = gpu.parameters();
    for (size_t k = 0; k < pc.size(); ++k) {
        report(pc[k].name.c_str(), rel_error(*pc[k].grad, *pg[k].grad), 1e-3f);
    }

    std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures;
}

#endif // USE_CUDA