#pragma once

#include "tensor.h"

namespace ops {

Tensor matmul   (const Tensor& a, const Tensor& b, DType out);
Tensor matmul(const Tensor& a, const Tensor& b);

Tensor batched_matmul(const Tensor& a, const Tensor& b);

Tensor matmul_bt(const Tensor& a, const Tensor& b, DType out);
Tensor matmul_bt(const Tensor& a, const Tensor& b);   // [M,K] @ [N,K]^T

Tensor matmul_at(const Tensor& a, const Tensor& b, DType out);
Tensor matmul_at(const Tensor& a, const Tensor& b);   // [K,M]^T @ [K,N]

void matmul_at_accumulate(const Tensor& a, const Tensor& b, Tensor& grad);
void matmul_bt_accumulate(const Tensor& a, const Tensor& b, Tensor& out);
}