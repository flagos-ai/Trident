//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFIAPITypes.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFITypes.h"
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
  MlirType mlirTVMFFIAPI##Name##TypeGet(MlirContext context) {                 \
    return wrap(trident::tvm_ffi::api::Name##Type::get(unwrap(context)));      \
  }                                                                            \
  MlirStringRef mlirTVMFFIAPI##Name##TypeGetName(void) {                       \
    return wrap(trident::tvm_ffi::api::Name##Type::name);                      \
  }                                                                            \
  MlirTypeID mlirTVMFFIAPI##Name##TypeGetTypeID(void) {                        \
    return wrap(trident::tvm_ffi::api::Name##Type::getTypeID());               \
  }                                                                            \
  bool mlirTypeIsATVMFFIAPI##Name##Type(MlirType type) {                       \
    return mlir::isa<trident::tvm_ffi::api::Name##Type>(unwrap(type));         \
  }

DEFINE_GENERIC_TYPE(Any)
DEFINE_GENERIC_TYPE(Exception)
DEFINE_GENERIC_TYPE(Function)
DEFINE_GENERIC_TYPE(Str)
DEFINE_GENERIC_TYPE(Tensor)

#undef DEFINE_GENERIC_TYPE

MlirType mlirTVMFFIAPIArrayTypeGetChecked(MlirLocation location,
                                          MlirType element) {
  mlir::Location const loc = unwrap(location);
  mlir::Type const type = unwrap(element);
  if (!type || type.getContext() != loc.getContext()) {
    mlir::emitError(loc)
        << "array element must belong to the construction context";
    return MlirType{nullptr};
  }
  trident::tvm_ffi::TVMFFICABIType const elementType =
      mlir::dyn_cast<trident::tvm_ffi::TVMFFICABIType>(type);
  if (!elementType) {
    mlir::emitError(loc) << "array element must be a TVM FFI C ABI type";
    return MlirType{nullptr};
  }
  return wrap(trident::tvm_ffi::api::ArrayType::getChecked(
      loc, loc.getContext(), elementType));
}

MlirType mlirTVMFFIAPIArrayTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<trident::tvm_ffi::api::ArrayType>(unwrap(type))
                  .getElementType());
}

MlirStringRef mlirTVMFFIAPIArrayTypeGetName(void) {
  return wrap(trident::tvm_ffi::api::ArrayType::name);
}

MlirTypeID mlirTVMFFIAPIArrayTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::api::ArrayType::getTypeID());
}

bool mlirTypeIsATVMFFIAPIArrayType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::api::ArrayType>(unwrap(type));
}

MlirType mlirTVMFFIAPITupleTypeGetChecked(MlirLocation location,
                                          intptr_t numTypes,
                                          const MlirType *types) {
  mlir::Location const loc = unwrap(location);
  if (numTypes < 0 || (numTypes > 0 && !types)) {
    mlir::emitError(loc) << "tuple requires a valid member array";
    return MlirType{nullptr};
  }
  llvm::SmallVector<trident::tvm_ffi::TVMFFICABIType> members;
  members.reserve(numTypes);
  for (intptr_t index = 0; index < numTypes; ++index) {
    const mlir::Type member = unwrap(types[index]);
    if (!member || member.getContext() != loc.getContext()) {
      mlir::emitError(loc)
          << "tuple members must belong to the construction context";
      return MlirType{nullptr};
    }
    trident::tvm_ffi::TVMFFICABIType const cabi =
        mlir::dyn_cast<trident::tvm_ffi::TVMFFICABIType>(member);
    if (!cabi) {
      mlir::emitError(loc) << "tuple elements must be TVM FFI C ABI types";
      return MlirType{nullptr};
    }
    members.push_back(cabi);
  }
  return wrap(trident::tvm_ffi::api::TupleType::getChecked(
      loc, loc.getContext(),
      llvm::ArrayRef<trident::tvm_ffi::TVMFFICABIType>(members)));
}

MlirStringRef mlirTVMFFIAPITupleTypeGetName(void) {
  return wrap(trident::tvm_ffi::api::TupleType::name);
}

intptr_t mlirTVMFFIAPITupleTypeGetNumTypes(MlirType type) {
  return static_cast<intptr_t>(
      mlir::cast<trident::tvm_ffi::api::TupleType>(unwrap(type))
          .getTypes()
          .size());
}

MlirType mlirTVMFFIAPITupleTypeGetType(MlirType type, intptr_t index) {
  return wrap(mlir::cast<trident::tvm_ffi::api::TupleType>(unwrap(type))
                  .getTypes()[index]);
}

MlirTypeID mlirTVMFFIAPITupleTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::api::TupleType::getTypeID());
}

bool mlirTypeIsATVMFFIAPITupleType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::api::TupleType>(unwrap(type));
}

MlirType mlirTVMFFIAPIUnionTypeGetChecked(MlirLocation location,
                                          intptr_t numTypes,
                                          const MlirType *types) {
  mlir::Location const loc = unwrap(location);
  if (numTypes < 0 || (numTypes > 0 && !types)) {
    mlir::emitError(loc) << "union requires a valid member array";
    return MlirType{nullptr};
  }
  llvm::SmallVector<trident::tvm_ffi::TVMFFICABIType> members;
  members.reserve(numTypes);
  for (intptr_t index = 0; index < numTypes; ++index) {
    const mlir::Type member = unwrap(types[index]);
    if (!member || member.getContext() != loc.getContext()) {
      mlir::emitError(loc)
          << "union members must belong to the construction context";
      return MlirType{nullptr};
    }
    trident::tvm_ffi::TVMFFICABIType const cabi =
        mlir::dyn_cast<trident::tvm_ffi::TVMFFICABIType>(member);
    if (!cabi) {
      mlir::emitError(loc) << "union members must be TVM FFI C ABI types";
      return MlirType{nullptr};
    }
    members.push_back(cabi);
  }
  return wrap(trident::tvm_ffi::api::UnionType::getChecked(
      loc, loc.getContext(),
      llvm::ArrayRef<trident::tvm_ffi::TVMFFICABIType>(members)));
}

MlirStringRef mlirTVMFFIAPIUnionTypeGetName(void) {
  return wrap(trident::tvm_ffi::api::UnionType::name);
}

intptr_t mlirTVMFFIAPIUnionTypeGetNumTypes(MlirType type) {
  return static_cast<intptr_t>(
      mlir::cast<trident::tvm_ffi::api::UnionType>(unwrap(type))
          .getTypes()
          .size());
}

MlirType mlirTVMFFIAPIUnionTypeGetType(MlirType type, intptr_t index) {
  return wrap(mlir::cast<trident::tvm_ffi::api::UnionType>(unwrap(type))
                  .getTypes()[index]);
}

MlirTypeID mlirTVMFFIAPIUnionTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::api::UnionType::getTypeID());
}

bool mlirTypeIsATVMFFIAPIUnionType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::api::UnionType>(unwrap(type));
}
