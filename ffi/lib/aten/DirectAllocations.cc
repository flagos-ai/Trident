// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
#include <ATen/DLConvertor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <utility>
#include <vector>

namespace {

// Zero-DLPack TensorObj wrapper owning an at::Tensor directly.
// Avoids at::toDLPack()'s ATenDLMTensor heap allocation and extra refcount
// bounce while presenting an exact standard DLTensor / kTVMFFITensor ABI.
class ATenTensorObj final : public tvm::ffi::TensorObj {
public:
  explicit ATenTensorObj(at::Tensor tensor) : tensor_(std::move(tensor)) {
    this->data = tensor_.data_ptr();
    this->device = at::torchDeviceToDLDevice(tensor_.device());
    this->ndim = static_cast<int32_t>(tensor_.dim());
    this->dtype = at::getDLDataType(tensor_);
    this->shape = const_cast<int64_t *>(tensor_.sizes().data());
    this->strides = const_cast<int64_t *>(tensor_.strides().data());
    this->byte_offset = 0;
  }

  const at::Tensor &tensor() const { return tensor_; }

private:
  at::Tensor tensor_;
};

static void (*gATenTensorObjDeleter)(void *, int) = nullptr;

inline bool isATenTensorObj(const tvm::ffi::TensorObj *obj) {
  return gATenTensorObjDeleter != nullptr &&
         reinterpret_cast<const TVMFFIObject *>(obj)->deleter ==
             gATenTensorObjDeleter;
}

inline void storeTensor(at::Tensor &&tensor, TVMFFIAny *output) {
  auto obj = tvm::ffi::make_object<ATenTensorObj>(std::move(tensor));
  if (gATenTensorObjDeleter == nullptr) {
    gATenTensorObjDeleter =
        reinterpret_cast<const TVMFFIObject *>(obj.get())->deleter;
  }
  *output = tvm::ffi::detail::AnyUnsafe::MoveAnyToTVMFFIAny(
      tvm::ffi::Any(tvm::ffi::Tensor(std::move(obj))));
}

inline at::TensorOptions
makeFastOptions(int32_t scalarType, uint64_t packedDevice, int32_t pinMemory) {
  at::TensorOptions opts;
  if (scalarType >= 0)
    opts = opts.dtype(static_cast<at::ScalarType>(scalarType));
  if (packedDevice != 0) {
    DLDevice dl{};
    std::memcpy(&dl, &packedDevice, sizeof(dl));
    opts = opts.device(at::dlDeviceToTorchDevice(dl.device_type, dl.device_id));
  }
  if (pinMemory >= 0)
    opts = opts.pinned_memory(pinMemory != 0);
  return opts;
}

tvm::ffi::AnyView view(int32_t tag, int64_t payload) {
  TVMFFIAny raw{};
  raw.type_index = tag;
  raw.v_int64 = payload;
  return tvm::ffi::AnyView::CopyFromTVMFFIAny(raw);
}

template <class T> std::optional<T> enumOption(tvm::ffi::AnyView value) {
  if (value == nullptr)
    return std::nullopt;
  return static_cast<T>(value.cast<int64_t>());
}

std::vector<int64_t> dimensions(void *object) {
  auto array = view(kTVMFFIArray, reinterpret_cast<int64_t>(object))
                   .cast<tvm::ffi::Array<tvm::ffi::Any>>();
  std::vector<int64_t> dims;
  dims.reserve(array.size());
  for (const auto &v : array)
    dims.push_back(v.cast<int64_t>());
  return dims;
}

at::TensorOptions buildOptions(tvm::ffi::AnyView dtype,
                               tvm::ffi::AnyView layout,
                               tvm::ffi::AnyView device,
                               tvm::ffi::AnyView pin) {
  std::optional<at::ScalarType> scalar;
  if (dtype != nullptr) {
    if (auto dl = dtype.try_cast<DLDataType>()) {
      if (dl->lanes != 1)
        throw std::runtime_error("Aten dtype must have exactly one lane");
      scalar = at::toScalarType(*dl);
    } else {
      scalar = static_cast<at::ScalarType>(dtype.cast<int64_t>());
    }
  }
  std::optional<at::Device> nativeDevice;
  if (device != nullptr) {
    auto dl = device.cast<DLDevice>();
    nativeDevice = at::dlDeviceToTorchDevice(dl.device_type, dl.device_id);
  }
  return at::TensorOptions()
      .dtype(scalar)
      .layout(enumOption<at::Layout>(layout))
      .device(nativeDevice)
      .pinned_memory(pin == nullptr ? std::nullopt
                                    : std::optional<bool>(pin.cast<bool>()));
}

} // namespace

// Fast register-only C ABIs (<= 6 arguments, zero stack spilling).
extern "C" int32_t trident_c_allocate_empty(int64_t rank, const int64_t *sizes,
                                            int32_t scalarType,
                                            uint64_t packedDevice,
                                            int32_t pinMemory,
                                            TVMFFIAny *output) {
  *output = TVMFFIAny{};
  TVM_FFI_SAFE_CALL_BEGIN();
  storeTensor(at::empty(at::IntArrayRef(sizes, rank),
                        makeFastOptions(scalarType, packedDevice, pinMemory)),
              output);
  TVM_FFI_SAFE_CALL_END();
}

extern "C" int32_t trident_c_allocate_zeros(int64_t rank, const int64_t *sizes,
                                            int32_t scalarType,
                                            uint64_t packedDevice,
                                            int32_t pinMemory,
                                            TVMFFIAny *output) {
  *output = TVMFFIAny{};
  TVM_FFI_SAFE_CALL_BEGIN();
  storeTensor(at::zeros(at::IntArrayRef(sizes, rank),
                        makeFastOptions(scalarType, packedDevice, pinMemory)),
              output);
  TVM_FFI_SAFE_CALL_END();
}

extern "C" int32_t trident_c_allocate_empty_like(void *selfObject,
                                                 int32_t scalarType,
                                                 int32_t pinMemory,
                                                 TVMFFIAny *output) {
  *output = TVMFFIAny{};
  TVM_FFI_SAFE_CALL_BEGIN();
  const auto *selfTensor = static_cast<const tvm::ffi::TensorObj *>(selfObject);
  // Contiguous CUDA tensors never carry pinned memory or non-strided layouts:
  // read DLTensor metadata directly without ToDLPack/fromDLPack heap bounce.
  if (selfTensor->device.device_type == kDLCUDA && pinMemory <= 0 &&
      selfTensor->dtype.lanes == 1 && tvm::ffi::IsContiguous(*selfTensor)) {
    at::ScalarType st = scalarType >= 0
                            ? static_cast<at::ScalarType>(scalarType)
                            : at::toScalarType(selfTensor->dtype);
    auto opts = at::TensorOptions().dtype(st).device(
        at::kCUDA, selfTensor->device.device_id);
    storeTensor(
        at::empty(at::IntArrayRef(selfTensor->shape, selfTensor->ndim), opts),
        output);
  } else {
    auto self = at::fromDLPack(selfTensor->ToDLPack());
    if (scalarType < 0 && pinMemory < 0) {
      storeTensor(at::empty_like(self), output);
    } else {
      storeTensor(
          at::empty_like(self, makeFastOptions(scalarType, 0, pinMemory)),
          output);
    }
  }
  TVM_FFI_SAFE_CALL_END();
}

extern "C" int32_t trident_c_allocate_zeros_like(void *selfObject,
                                                 int32_t scalarType,
                                                 int32_t pinMemory,
                                                 TVMFFIAny *output) {
  *output = TVMFFIAny{};
  TVM_FFI_SAFE_CALL_BEGIN();
  const auto *selfTensor = static_cast<const tvm::ffi::TensorObj *>(selfObject);
  if (selfTensor->device.device_type == kDLCUDA && pinMemory <= 0 &&
      selfTensor->dtype.lanes == 1 && tvm::ffi::IsContiguous(*selfTensor)) {
    at::ScalarType st = scalarType >= 0
                            ? static_cast<at::ScalarType>(scalarType)
                            : at::toScalarType(selfTensor->dtype);
    auto opts = at::TensorOptions().dtype(st).device(
        at::kCUDA, selfTensor->device.device_id);
    storeTensor(
        at::zeros(at::IntArrayRef(selfTensor->shape, selfTensor->ndim), opts),
        output);
  } else {
    auto self = at::fromDLPack(selfTensor->ToDLPack());
    if (scalarType < 0 && pinMemory < 0) {
      storeTensor(at::zeros_like(self), output);
    } else {
      storeTensor(
          at::zeros_like(self, makeFastOptions(scalarType, 0, pinMemory)),
          output);
    }
  }
  TVM_FFI_SAFE_CALL_END();
}

extern "C" int32_t trident_c_view(void *selfObject, int64_t rank,
                                  const int64_t *sizes, TVMFFIAny *output) {
  *output = TVMFFIAny{};
  TVM_FFI_SAFE_CALL_BEGIN();
  const auto *selfTensor = static_cast<const tvm::ffi::TensorObj *>(selfObject);
  // Zero-cost no-op view when shape already matches identically.
  if (selfTensor->ndim == rank) {
    bool same = true;
    for (int64_t i = 0; i < rank; ++i) {
      if (sizes[i] != selfTensor->shape[i]) {
        same = false;
        break;
      }
    }
    if (same) {
      TVMFFIObjectIncRef(selfObject);
      output->type_index = kTVMFFITensor;
      output->zero_padding = 0;
      output->v_obj = static_cast<TVMFFIObject *>(selfObject);
      return 0;
    }
  }
  at::IntArrayRef shape(sizes, rank);
  if (isATenTensorObj(selfTensor)) {
    storeTensor(
        static_cast<const ATenTensorObj *>(selfTensor)->tensor().view(shape),
        output);
  } else {
    auto self = at::fromDLPack(selfTensor->ToDLPack());
    storeTensor(self.view(shape), output);
  }
  TVM_FFI_SAFE_CALL_END();
}

extern "C" int32_t trident_c_view_array(void *selfObject, void *sizeObject,
                                        TVMFFIAny *output) {
  *output = TVMFFIAny{};
  TVM_FFI_SAFE_CALL_BEGIN();
  auto dims = dimensions(sizeObject);
  return trident_c_view(selfObject, static_cast<int64_t>(dims.size()),
                        dims.data(), output);
  TVM_FFI_SAFE_CALL_END();
}

// General fallback C ABI: all inputs borrow the caller's objects; only the
// successful output owns a new reference. Failure leaves output None and sets
// the TVM FFI thread error.
extern "C" int32_t
trident_c_allocate(int32_t kind, int64_t rank, const int64_t *sizes,
                   const int64_t *strides, void *selfObject, void *sizeObject,
                   void *strideObject, int32_t dtypeTag, int64_t dtypeBits,
                   int32_t layoutTag, int64_t layoutBits, int32_t deviceTag,
                   int64_t deviceBits, int32_t pinTag, int64_t pinBits,
                   int32_t memoryTag, int64_t memoryBits, TVMFFIAny *output) {
  *output = TVMFFIAny{};
  TVM_FFI_SAFE_CALL_BEGIN();
  auto dtype = view(dtypeTag, dtypeBits);
  auto layout = view(layoutTag, layoutBits);
  auto device = view(deviceTag, deviceBits);
  auto pin = view(pinTag, pinBits);
  auto memory = view(memoryTag, memoryBits);
  at::Tensor tensor;
  if (kind == 2 || kind == 4) {
    auto self = at::fromDLPack(
        static_cast<const tvm::ffi::TensorObj *>(selfObject)->ToDLPack());
    bool defaults = (dtype == nullptr && layout == nullptr &&
                     device == nullptr && pin == nullptr && memory == nullptr);
    if (kind == 2) {
      tensor =
          defaults
              ? at::empty_like(self)
              : at::empty_like(self, buildOptions(dtype, layout, device, pin),
                               enumOption<at::MemoryFormat>(memory));
    } else {
      tensor =
          defaults
              ? at::zeros_like(self)
              : at::zeros_like(self, buildOptions(dtype, layout, device, pin),
                               enumOption<at::MemoryFormat>(memory));
    }
  } else {
    std::vector<int64_t> sizeStorage, strideStorage;
    if (rank < 0) {
      sizeStorage = dimensions(sizeObject);
      sizes = sizeStorage.data();
      rank = static_cast<int64_t>(sizeStorage.size());
    }
    at::IntArrayRef shape(sizes, rank);
    auto options = buildOptions(dtype, layout, device, pin);
    if (kind == 0) {
      tensor = at::empty(shape, options, enumOption<at::MemoryFormat>(memory));
    } else if (kind == 1) {
      if (strideObject) {
        strideStorage = dimensions(strideObject);
        strides = strideStorage.data();
        if (strideStorage.size() != static_cast<size_t>(rank))
          throw std::runtime_error("size/stride rank mismatch");
      }
      tensor =
          at::empty_strided(shape, at::IntArrayRef(strides, rank), options);
    } else if (kind == 3) {
      tensor = at::zeros(shape, options);
    } else {
      throw std::runtime_error("unsupported direct allocation kind");
    }
  }
  storeTensor(std::move(tensor), output);
  TVM_FFI_SAFE_CALL_END();
}
