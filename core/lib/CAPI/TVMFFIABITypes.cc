//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFIABITypes.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h" // NOLINT(misc-include-cleaner)
#include <cstdint>                // NOLINT(misc-include-cleaner)
#include <llvm/ADT/ArrayRef.h>    // NOLINT(misc-include-cleaner)
#include <llvm/ADT/SmallVector.h> // NOLINT(misc-include-cleaner)
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h> // NOLINT(misc-include-cleaner)
#include <mlir/Support/LLVM.h>

#define DEFINE_GENERIC_TYPE(Name)                                              \
  MlirType mlirTVMFFIABI##Name##TypeGet(MlirContext context) {                 \
    return wrap(trident::tvm_ffi::abi::Name##Type::get(unwrap(context)));      \
  }                                                                            \
  MlirStringRef mlirTVMFFIABI##Name##TypeGetName(void) {                       \
    return wrap(trident::tvm_ffi::abi::Name##Type::name);                      \
  }                                                                            \
  MlirTypeID mlirTVMFFIABI##Name##TypeGetTypeID(void) {                        \
    return wrap(trident::tvm_ffi::abi::Name##Type::getTypeID());               \
  }                                                                            \
  bool mlirTypeIsATVMFFIABI##Name##Type(MlirType type) {                       \
    return mlir::isa<trident::tvm_ffi::abi::Name##Type>(unwrap(type));         \
  }

DEFINE_GENERIC_TYPE(Any)
DEFINE_GENERIC_TYPE(Array)
DEFINE_GENERIC_TYPE(Bool)
DEFINE_GENERIC_TYPE(DType)
DEFINE_GENERIC_TYPE(Device)
DEFINE_GENERIC_TYPE(Exception)
DEFINE_GENERIC_TYPE(Float)
DEFINE_GENERIC_TYPE(Function)
DEFINE_GENERIC_TYPE(Int)
DEFINE_GENERIC_TYPE(None)
DEFINE_GENERIC_TYPE(Object)
DEFINE_GENERIC_TYPE(PackArguments)
DEFINE_GENERIC_TYPE(RawStr)
DEFINE_GENERIC_TYPE(SmallStr)
DEFINE_GENERIC_TYPE(Str)
DEFINE_GENERIC_TYPE(Tensor)

#undef DEFINE_GENERIC_TYPE
