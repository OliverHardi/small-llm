// // Whole-model gradient check: compares Model::backward against central finite differences
// // for every parameter returned by model.parameters(), including the tied embedding.
// //
// // Exit code is 0 if all parameters pass, 1 otherwise, so it can be run from CTest.

// #include <algorithm>
// #include <cmath>
// #include <cstdint>
// #include <cstdio>
// #include <vector>

// #include "config.h"
// #include "model.h"
// #include "ops/cross_entropy.h"

// int main() {

//     // Tiny model: cheap enough to run many forward passes per parameter.
//     ModelConfig cfg;
//     cfg.vocab_size = 32;
//     cfg.context_length = 8;
//     cfg.num_layers = 2;
//     cfg.d_model = 16;
//     cfg.num_heads = 2;      // head_dim = 8
//     cfg.ffn_hidden = 24;

//     Tensor::seed(1);
//     Model model(cfg);
//     std::vector<Param> params = model.parameters();

//     // Training init (std 0.02) makes gradients tiny and the check weak.
//     // Use larger weights; norm weights stay near 1 but are no longer exactly 1.
//     uint32_t seed = 100;
//     for (Param& p : params) {
//         if (p.decay) p.value->randomNormal(0.0f, 0.3f, seed++);
//         else         p.value->randomNormal(1.0f, 0.2f, seed++);
//     }

//     // Fixed random batch
//     const size_t B = 2, T = 8;
//     std::vector<uint32_t> ids(B * T);
//     Tensor targets({B * T});

//     uint32_t state = 7;
//     auto next_id = [&]() {
//         state = state * 1664525u + 1013904223u;
//         return (state >> 8) % static_cast<uint32_t>(cfg.vocab_size);
//     };
//     for (size_t i = 0; i < B * T; ++i) {
//         ids[i] = next_id();
//         targets[i] = static_cast<float>(next_id());
//     }

//     // Loss as a function of the current weights (forward only).
//     auto loss_fn = [&]() {
//         return static_cast<double>(ops::cross_entropy(model.forward(ids, B, T), targets));
//     };

//     // Analytic gradients from one forward + backward.
//     model.zero_grad();
//     Tensor logits = model.forward(ids, B, T);
//     model.backward(ops::cross_entropy_backward(logits, targets));

//     std::vector<Tensor> analytic;
//     for (const Param& p : params) analytic.push_back(p.grad->clone());

//     // Finite differences, ~20 sampled entries per parameter tensor.
//     const float eps = 1e-2f;   // float32: much smaller and rounding error dominates
//     const double tolerance = 2e-2;
//     int failures = 0;

//     std::printf("initial loss %.4f (ln(vocab) = %.4f)\n", loss_fn(), std::log(static_cast<double>(cfg.vocab_size)));

//     for (size_t k = 0; k < params.size(); ++k) {
//         Tensor& w = *params[k].value;
//         const size_t step = std::max<size_t>(1, w.numel() / 20);

//         double diff_sq = 0.0, analytic_sq = 0.0, numeric_sq = 0.0;

//         for (size_t i = 0; i < w.numel(); i += step) {
//             const float original = w[i];

//             w[i] = original + eps;  const double loss_plus = loss_fn();
//             w[i] = original - eps;  const double loss_minus = loss_fn();
//             w[i] = original;

//             const double numeric = (loss_plus - loss_minus) / (2.0 * eps);
//             const double a = analytic[k][i];

//             diff_sq += (a - numeric) * (a - numeric);
//             analytic_sq += a * a;
//             numeric_sq += numeric * numeric;
//         }

//         const double rel = std::sqrt(diff_sq) / (std::sqrt(analytic_sq) + std::sqrt(numeric_sq) + 1e-12);
//         const bool ok = rel < tolerance;
//         if (!ok) ++failures;

//         std::printf("  %-5s %-22s rel err %.5f\n", ok ? "ok" : "FAIL", params[k].name.c_str(), rel);
//     }

//     std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
//     return failures ? 1 : 0;
// }