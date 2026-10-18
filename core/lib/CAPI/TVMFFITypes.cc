//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

#define DEFINE_GENERIC_TYPE(Name)                                              \
  MlirType mlirTVMFFI##Name##TypeGet(MlirContext context) {                    \
    return wrap(trident::tvm_ffi::Name##Type::get(unwrap(context)));           \
  }                                                                            \
  MlirStringRef mlirTVMFFI##Name##TypeGetName(void) {                          \
    return wrap(trident::tvm_ffi::Name##Type::name);                           \
  }                                                                            \
  MlirTypeID mlirTVMFFI##Name##TypeGetTypeID(void) {                           \
    return wrap(trident::tvm_ffi::Name##Type::getTypeID());                    \
  }                                                                            \
  bool mlirTypeIsATVMFFI##Name##Type(MlirType type) {                          \
    return mlir::isa<trident::tvm_ffi::Name##Type>(unwrap(type));              \
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
DEFINE_GENERIC_TYPE(RawStr)
DEFINE_GENERIC_TYPE(SmallStr)
DEFINE_GENERIC_TYPE(Str)
DEFINE_GENERIC_TYPE(Tensor)

#undef DEFINE_GENERIC_TYPE

MlirType mlirTVMFFIResultTypeGetChecked(MlirLocation location,
                                        MlirType payload) {
  const mlir::Location loc = unwrap(location);
  const mlir::Type type = unwrap(payload);
  if (!type) {
    mlir::emitError(loc) << "Result requires one payload type";
    return MlirType{nullptr};
  }
  if (type.getContext() != loc.getContext()) {
    mlir::emitError(loc)
        << "Result payload must belong to the construction context";
    return MlirType{nullptr};
  }
  return wrap(
      trident::tvm_ffi::ResultType::getChecked(loc, loc.getContext(), type));
}

MlirStringRef mlirTVMFFIResultTypeGetName(void) {
  return wrap(trident::tvm_ffi::ResultType::name);
}

MlirType mlirTVMFFIResultTypeGetPayloadType(MlirType type) {
  return wrap(
      mlir::cast<trident::tvm_ffi::ResultType>(unwrap(type)).getPayloadType());
}

MlirTypeID mlirTVMFFIResultTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::ResultType::getTypeID());
}

bool mlirTypeIsATVMFFIResultType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::ResultType>(unwrap(type));
}

MlirType mlirTVMFFIUnionTypeGetChecked(MlirLocation location, intptr_t numTypes,
                                       const MlirType *types) {
  const mlir::Location loc = unwrap(location);
  llvm::SmallVector<mlir::Type> members;
  members.reserve(numTypes);
  for (intptr_t index = 0; index < numTypes; ++index) {
    const mlir::Type member = unwrap(types[index]);
    if (member.getContext() != loc.getContext()) {
      mlir::emitError(loc)
          << "union members must belong to the construction context";
      return MlirType{nullptr};
    }
    members.push_back(member);
  }
  return wrap(trident::tvm_ffi::UnionType::getChecked(
      loc, loc.getContext(), llvm::ArrayRef<mlir::Type>(members)));
}

MlirStringRef mlirTVMFFIUnionTypeGetName(void) {
  return wrap(trident::tvm_ffi::UnionType::name);
}

intptr_t mlirTVMFFIUnionTypeGetNumTypes(MlirType type) {
  return static_cast<intptr_t>(
      mlir::cast<trident::tvm_ffi::UnionType>(unwrap(type)).getTypes().size());
}

MlirType mlirTVMFFIUnionTypeGetType(MlirType type, intptr_t index) {
  return wrap(
      mlir::cast<trident::tvm_ffi::UnionType>(unwrap(type)).getTypes()[index]);
}

MlirTypeID mlirTVMFFIUnionTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::UnionType::getTypeID());
}

bool mlirTypeIsATVMFFIUnionType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::UnionType>(unwrap(type));
}
