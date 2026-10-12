#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <random>
#include <vector>

enum class Device { CPU, CUDA };

class Tensor {
public:

    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(std::initializer_list<size_t> shape);
    Tensor(const std::vector<size_t>& shape, float value);

    // shape
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }
    size_t numel() const { return numel_; }
    size_t size() const { return numel_; }
    size_t dim(size_t axis) const;

    // device
    Device device() const { return device_; }
    Tensor to(Device target) const;   // returns a shallow copy if already on `target`

    // data access (valid for the tensor's own device)
    float* data() { return storage_ ? storage_->host.data() : nullptr; }
    const float* data() const { return storage_ ? storage_->host.data() : nullptr; }

    // element access: host (CPU) tensors only
    float& operator[](size_t index) {
        assert(device_ == Device::CPU);
        return storage_->host[index];
    }

    const float& operator[](size_t index) const {
        assert(device_ == Device::CPU);
        return storage_->host[index];
    }

    template <typename... Indices>
    float& operator()(Indices... indices) {
        assert(device_ == Device::CPU);
        return storage_->host[offset_of(static_cast<size_t>(indices)...)];
    }

    template <typename... Indices>
    const float& operator()(Indices... indices) const {
        assert(device_ == Device::CPU);
        return storage_->host[offset_of(static_cast<size_t>(indices)...)];
    }

    // fill
    void fill(float value);
    void zero();

    // random
    static void seed(uint32_t s);
    static std::mt19937& rng();
    void randomNormal(float mean, float stddev);
    void randomNormal(float mean, float stddev, uint32_t seed);

    // statistics
    float sum() const;
    float max() const;
    float min() const;

    // reshape
    void reshape(const std::vector<size_t>& newShape);          // changes this handle's shape only
    void reshape(std::initializer_list<size_t> newShape);
    Tensor reshaped(const std::vector<size_t>& new_shape) const;   // view: shares data
    Tensor reshaped(std::initializer_list<size_t> new_shape) const;

    // deep copy
    Tensor clone() const;

    // debug
    void printShape() const;

private:
    struct Storage {
        std::vector<float> host;
        // TODO: a device pointer, freed in the destructor
    };

    std::shared_ptr<Storage> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t numel_ = 0;
    Device device_ = Device::CPU;

    void calculateStrides();

    // flat index from n indices
    template <typename... Indices>
    size_t offset_of(Indices... indices) const {
        assert(sizeof...(Indices) == shape_.size());
        const size_t idx[] = {static_cast<size_t>(indices)...};
        size_t result = 0;
        for (size_t i = 0; i < sizeof...(Indices); ++i) {
            assert(idx[i] < shape_[i]);
            result += idx[i] * strides_[i];
        }
        return result;
    }
};