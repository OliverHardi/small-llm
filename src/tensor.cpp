#include "tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.h"
#endif

// helper
static size_t product(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

// storage

Tensor::Storage::~Storage() {
#ifdef USE_CUDA
    // Never throw from a destructor.
    try { cuda::device_free(device); } catch (...) {}
#endif
}

void Tensor::allocate() {
    if (device_ == Device::CPU) {
        storage_->host.assign(numel_, 0.0f);
        return;
    }
#ifdef USE_CUDA
    storage_->device = static_cast<float*>(cuda::device_alloc(numel_ * sizeof(float)));
#else
    throw std::runtime_error("Tensor: built without CUDA support (define USE_CUDA)");
#endif
}

// constructors

Tensor::Tensor(const std::vector<size_t>& shape)
    : storage_(std::make_shared<Storage>()), shape_(shape), numel_(product(shape)) {
    calculateStrides();
    storage_->host.assign(numel_, 0.0f);
}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : storage_(std::make_shared<Storage>()), shape_(shape), numel_(product(shape_)) {
    calculateStrides();
    storage_->host.assign(numel_, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape, float value)
    : storage_(std::make_shared<Storage>()), shape_(shape), numel_(product(shape)) {
    calculateStrides();
    storage_->host.assign(numel_, value);
}

Tensor::Tensor(const std::vector<size_t>& shape, Device device)
    : storage_(std::make_shared<Storage>()), shape_(shape), numel_(product(shape)), device_(device) {
    calculateStrides();
    allocate();
}

// shape

size_t Tensor::dim(size_t axis) const {
    if (axis >= ndim()) {
        throw std::out_of_range("Tensor dimension out of range");
    }

    return shape_[axis];
}

// device

Tensor Tensor::to(Device target) const {
    if (target == device_) {
        return *this;
    }

#ifdef USE_CUDA
    Tensor result(shape_, target);
    const size_t bytes = numel_ * sizeof(float);
 
    if (target == Device::CUDA) {
        cuda::copy_h2d(result.data(), data(), bytes);
    } else {
        cuda::copy_d2h(result.data(), data(), bytes);   // synchronizes
    }
    return result;
#else
    throw std::runtime_error("Tensor::to: built without CUDA support (define USE_CUDA)");
#endif
}

// fill

void Tensor::fill(float value) {
    std::fill(data(), data() + numel_, value);
}

void Tensor::zero() {
    if (device_ == Device::CUDA) {
        #ifdef USE_CUDA
        cuda::device_zero(data(), numel());
        #endif
    } else {
        std::fill(data(), data() + numel(), 0.0f);
    }
}

// random init

void Tensor::seed(uint32_t s) { rng().seed(s); }
std::mt19937& Tensor::rng() { static std::mt19937 gen(0); return gen; }

void Tensor::randomNormal(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    float* p = data();
    for (size_t i = 0; i < numel_; ++i) p[i] = distribution(rng());
}

void Tensor::randomNormal(float mean, float stddev, uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(mean, stddev);

    float* p = data();
    for (size_t i = 0; i < numel_; ++i) p[i] = distribution(generator);
}

// stats

float Tensor::sum() const {
    float result = 0.0f;
    const float* p = data();
    for (size_t i = 0; i < numel_; ++i) result += p[i];
    return result;
}

float Tensor::max() const {
    if (numel_ == 0) {
        throw std::runtime_error("Cannot find max of empty tensor");
    }
    return *std::max_element(data(), data() + numel_);
}

float Tensor::min() const {
    if (numel_ == 0) {
        throw std::runtime_error("Cannot find min of empty tensor");
    }
    return *std::min_element(data(), data() + numel_);
}

// reshaping
void Tensor::reshape(const std::vector<size_t>& newShape) {
    if (product(newShape) != numel_) {
        throw std::runtime_error("Cannot reshape tensor: element count differs");
    }

    shape_ = newShape;
    calculateStrides();
}

void Tensor::reshape(std::initializer_list<size_t> newShape) {
    reshape(std::vector<size_t>(newShape));
}

// A view: same data, new shape. No copy.
Tensor Tensor::reshaped(const std::vector<size_t>& new_shape) const {
    Tensor view = *this;      // shallow copy: shares storage_
    view.reshape(new_shape);
    return view;
}

Tensor Tensor::reshaped(std::initializer_list<size_t> new_shape) const {
    return reshaped(std::vector<size_t>(new_shape));
}

// deep copy

Tensor Tensor::clone() const {
    Tensor out(shape_, device_);
    if (device_ == Device::CUDA) {
        #ifdef USE_CUDA
        cuda::copy_d2d(out.data(), data(), numel() * sizeof(float));
        #endif
    } else {
        std::copy(data(), data() + numel(), out.data());
    }
    return out;
}

// debug

void Tensor::printShape() const {
    std::cout << "(";

    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];

        if (i + 1 < shape_.size()) {
            std::cout << ", ";
        }
    }

    std::cout << ")\n";
}

// row major strides
void Tensor::calculateStrides() {
    strides_.resize(shape_.size());

    if (shape_.empty()) {
        return;
    }

    strides_.back() = 1;

    for (size_t i = shape_.size() - 1; i > 0; --i) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
}