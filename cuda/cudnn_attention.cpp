// The only file in the project that includes cudnn-frontend. Everything else sees
// cudnn_attention.h, so the frontend's template compile time is paid here and nowhere else.
//
// Build as C++17 or newer. Include paths: the cudnn-frontend headers and the cuDNN/CUDA headers.
// Link: cudnn (cudnn.lib on Windows) and cudart.

#include "cudnn_attention.h"

#include <cudnn.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace {

using Graph = fe::graph::Graph;
using TensorAttrs = fe::graph::Tensor_attributes;

// Tensor ids used to bind device pointers at execute time.
enum : int64_t { UID_Q = 1, UID_K, UID_V, UID_O, UID_STATS, UID_DO, UID_DQ, UID_DK, UID_DV };

void check_cudnn(cudnnStatus_t status, const char* what) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + cudnnGetErrorString(status));
    }
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_fe(cudnn_frontend::error_t status, const char* what) {
    if (status.is_bad()) {
        throw std::runtime_error(std::string(what) + ": " + status.get_message());
    }
}

// A contiguous bf16 [B, H, T, D] tensor.
std::shared_ptr<TensorAttrs> bhtd(Graph& g, const char* name, int64_t uid,
                                  int64_t B, int64_t H, int64_t T, int64_t D) {
    return g.tensor(TensorAttrs()
                        .set_name(name)
                        .set_uid(uid)
                        .set_dim({B, H, T, D})
                        .set_stride({H * T * D, T * D, D, 1}));
}

// The fp32 softmax statistics, [B, H, T, 1].
std::shared_ptr<TensorAttrs> stats_input(Graph& g, int64_t B, int64_t H, int64_t T) {
    return g.tensor(TensorAttrs()
                        .set_name("stats")
                        .set_uid(UID_STATS)
                        .set_dim({B, H, T, 1})
                        .set_stride({H * T, T, 1, 1})
                        .set_data_type(fe::DataType_t::FLOAT));
}

void set_precisions(Graph& g) {
    g.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
}

std::shared_ptr<Graph> build_forward(cudnnHandle_t handle, int64_t B, int64_t H, int64_t T, int64_t D) {
    auto g = std::make_shared<Graph>();
    set_precisions(*g);

    auto q = bhtd(*g, "Q", UID_Q, B, H, T, D);
    auto k = bhtd(*g, "K", UID_K, B, H, T, D);
    auto v = bhtd(*g, "V", UID_V, B, H, T, D);

    auto options = fe::graph::SDPA_attributes()
                       .set_name("sdpa_forward")
                       .set_generate_stats(true)   // the backward pass needs these
                       .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                       .set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                       .set_diagonal_band_right_bound(0);   // right bound 0 = causal

    auto [o, stats] = g->sdpa(q, k, v, options);

    o->set_output(true).set_dim({B, H, T, D}).set_stride({H * T * D, T * D, D, 1}).set_uid(UID_O);
    stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(UID_STATS);

    // Validates, builds the operation graph, picks an engine and builds the plan.
    check_fe(g->build(handle, {fe::HeurMode_t::A}), "cuDNN SDPA forward graph build");
    return g;
}

std::shared_ptr<Graph> build_backward(cudnnHandle_t handle, int64_t B, int64_t H, int64_t T, int64_t D) {
    auto g = std::make_shared<Graph>();
    set_precisions(*g);

    auto q  = bhtd(*g, "Q",  UID_Q,  B, H, T, D);
    auto k  = bhtd(*g, "K",  UID_K,  B, H, T, D);
    auto v  = bhtd(*g, "V",  UID_V,  B, H, T, D);
    auto o  = bhtd(*g, "O",  UID_O,  B, H, T, D);
    auto d_o = bhtd(*g, "dO", UID_DO, B, H, T, D);
    auto stats = stats_input(*g, B, H, T);

    // Masking and scale must match the forward graph exactly.
    auto options = fe::graph::SDPA_backward_attributes()
                       .set_name("sdpa_backward")
                       .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                       .set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                       .set_diagonal_band_right_bound(0);
    // .set_deterministic_algorithm(true)   // reproducible gradients, possibly slower

    auto [dq, dk, dv] = g->sdpa_backward(q, k, v, o, d_o, stats, options);

    dq->set_output(true).set_dim({B, H, T, D}).set_stride({H * T * D, T * D, D, 1}).set_uid(UID_DQ);
    dk->set_output(true).set_dim({B, H, T, D}).set_stride({H * T * D, T * D, D, 1}).set_uid(UID_DK);
    dv->set_output(true).set_dim({B, H, T, D}).set_stride({H * T * D, T * D, D, 1}).set_uid(UID_DV);

    check_fe(g->build(handle, {fe::HeurMode_t::A}), "cuDNN SDPA backward graph build");
    return g;
}

} // namespace

struct CudnnAttention::Impl {
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<Graph> forward_graph;
    std::shared_ptr<Graph> backward_graph;
    void* workspace = nullptr;   // shared by both graphs; calls are serialized on one stream

    Impl(int64_t B, int64_t H, int64_t T, int64_t D) {
        // Limits from the cuDNN SDPA support matrix for Ada, bf16.
        if (D % 8 != 0 || D > 128) {
            throw std::invalid_argument("CudnnAttention: head dim must be a multiple of 8 and at most 128 (backward limit on Ada)");
        }

        check_cudnn(cudnnCreate(&handle), "cudnnCreate");

        forward_graph = build_forward(handle, B, H, T, D);
        backward_graph = build_backward(handle, B, H, T, D);

        const int64_t workspace_size =
            std::max(forward_graph->get_workspace_size(), backward_graph->get_workspace_size());
        if (workspace_size > 0) {
            check_cuda(cudaMalloc(&workspace, static_cast<size_t>(workspace_size)), "cudaMalloc workspace");
        }
    }

    ~Impl() {
        if (workspace) cudaFree(workspace);
        if (handle) cudnnDestroy(handle);
    }
};

CudnnAttention::CudnnAttention(int64_t B, int64_t H, int64_t T, int64_t D)
    : impl_(std::make_unique<Impl>(B, H, T, D)) {}

CudnnAttention::~CudnnAttention() = default;

void CudnnAttention::forward(const void* q, const void* k, const void* v,
                             void* o, void* stats, cudaStream_t stream) {
    check_cudnn(cudnnSetStream(impl_->handle, stream), "cudnnSetStream");

    std::unordered_map<int64_t, void*> pack = {
        {UID_Q, const_cast<void*>(q)},
        {UID_K, const_cast<void*>(k)},
        {UID_V, const_cast<void*>(v)},
        {UID_O, o},
        {UID_STATS, stats},
    };
    check_fe(impl_->forward_graph->execute(impl_->handle, pack, impl_->workspace), "cuDNN SDPA forward");
}

void CudnnAttention::backward(const void* q, const void* k, const void* v, const void* o,
                              const void* d_o, const void* stats,
                              void* dq, void* dk, void* dv, cudaStream_t stream) {
    check_cudnn(cudnnSetStream(impl_->handle, stream), "cudnnSetStream");

    std::unordered_map<int64_t, void*> pack = {
        {UID_Q, const_cast<void*>(q)},
        {UID_K, const_cast<void*>(k)},
        {UID_V, const_cast<void*>(v)},
        {UID_O, const_cast<void*>(o)},
        {UID_DO, const_cast<void*>(d_o)},
        {UID_STATS, const_cast<void*>(stats)},
        {UID_DQ, dq},
        {UID_DK, dk},
        {UID_DV, dv},
    };
    check_fe(impl_->backward_graph->execute(impl_->handle, pack, impl_->workspace), "cuDNN SDPA backward");
}