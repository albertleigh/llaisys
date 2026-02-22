#include "tensor.hpp"

#include "../ops/rearrange/op.hpp"
#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    int ndim = static_cast<int>(this->ndim());
    if (ndim == 0) {
        return true;
    }

    size_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (static_cast<size_t>(this->strides()[i]) != expected_stride) {
            return false;
        }
        expected_stride *= this->shape()[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim = this->ndim();
    if (ndim != order.size()) {
        EXCEPTION_INVALID_ARGUMENT("Permute order size does not match tensor dimension");
    }

    std::vector<bool> used(ndim, false);
    for (auto o : order) {
        if (o > ndim - 1) {
            EXCEPTION_INVALID_ARGUMENT("Permute order contains dimension larger than tensor dimension");
        }
        if (used[o]) {
            EXCEPTION_INVALID_ARGUMENT("Permute order contains duplicate dimension");
        }
        used[o] = true;
    }

    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = this->shape()[order[i]];
        new_strides[i] = this->strides()[order[i]];
    }

    TensorMeta meta{this->dtype(), std::move(new_shape), std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    if (!isContiguous()) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    if (new_numel != this->numel()) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    size_t ndim_new = shape.size();
    std::vector<ptrdiff_t> strides_new(ndim_new);
    size_t stride_new = 1;
    for (size_t i = 1; i <= ndim_new; i++) {
        strides_new[ndim_new - i] = stride_new;
        stride_new *= shape[ndim_new - i];
    }

    TensorMeta meta{this->dtype(), shape, std::move(strides_new)};

    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= this->ndim()) {
        EXCEPTION_INVALID_ARGUMENT("Slice dimension is larger than tensor dimension");
    }

    if (start >= end) {
        EXCEPTION_INVALID_ARGUMENT("Slice start is larger than end");
    }

    if (end > this->shape()[dim]) {
        EXCEPTION_INVALID_ARGUMENT("Slice end is larger than tensor dimension");
    }

    std::vector<size_t> shape_new = this->shape();
    std::vector<ptrdiff_t> strides_new = this->strides();

    shape_new[dim] = end - start;

    size_t offset_new = _offset + start * strides()[dim] * elementSize();

    TensorMeta meta{this->dtype(), std::move(shape_new), std::move(strides_new)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, offset_new));
}

void Tensor::load(const void *src_) {
    if (this->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(), src_, this->numel() * this->elementSize(), LLAISYS_MEMCPY_H2D);
    } else {
        std::memcpy(this->data(), src_, this->numel() * this->elementSize());
    }
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    ASSERT(this->deviceType() == LLAISYS_DEVICE_CPU, "Only CPU tensor can be incontiguous");
    auto result = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());
    ops::rearrange(result, std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset)));
    return result;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    if (isContiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device < 0) {
        device = this->deviceId();
    }

    // Already on target device
    if (this->deviceType() == device_type && this->deviceId() == device) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    // Ensure contiguous layout before cross-device copy
    auto src = this->isContiguous()
        ? std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset))
        : this->contiguous();

    auto dst = Tensor::create(src->shape(), src->dtype(), device_type, device);
    size_t nbytes = src->numel() * src->elementSize();

    llaisysDeviceType_t src_device = src->deviceType();

    if (src_device == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst->data(), src->data(), nbytes);
    } else if (src_device == LLAISYS_DEVICE_CPU) {
        // H2D
        core::context().setDevice(device_type, device);
        core::context().runtime().api()->memcpy_sync(
            dst->data(), src->data(), nbytes, LLAISYS_MEMCPY_H2D);
    } else if (device_type == LLAISYS_DEVICE_CPU) {
        // D2H
        core::context().setDevice(src_device, src->deviceId());
        core::context().runtime().api()->memcpy_sync(
            dst->data(), src->data(), nbytes, LLAISYS_MEMCPY_D2H);
    } else {
        // D2D
        core::context().setDevice(src_device, src->deviceId());
        core::context().runtime().api()->memcpy_sync(
            dst->data(), src->data(), nbytes, LLAISYS_MEMCPY_D2D);
    }

    return dst;
}

} // namespace llaisys
